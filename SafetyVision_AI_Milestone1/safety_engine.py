import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any, Optional

# Exact classes from models/best.pt
MODEL_CLASSES = {
    0: 'helmet',
    1: 'gloves',
    2: 'vest',
    3: 'boots',
    4: 'goggles',
    5: 'none',
    6: 'Person',
    7: 'no_helmet',
    8: 'no_goggle',
    9: 'no_gloves',
    10: 'no_boots'
}

PPE_COMPLIANT_CLASSES = {"helmet", "gloves", "vest", "boots", "goggles"}
EXPLICIT_VIOLATION_CLASSES = {
    "no_helmet", "no_goggle", "no_gloves", "no_boots",
    # Aliases/fallbacks if model outputs variant strings or spatial engine flags missing PPE
    "no_hardhat", "no_vest", "missing_helmet", "missing_vest", "missing_gloves", "missing_boots", "missing_goggles"
}

SEVERITY_MAPPING = {
    "no_helmet": "high",
    "no_hardhat": "high",
    "missing_helmet": "high",
    "no_vest": "high",
    "missing_vest": "high",
    "no_goggle": "medium",
    "missing_goggles": "medium",
    "no_boots": "medium",
    "missing_boots": "medium",
    "no_gloves": "low",
    "missing_gloves": "low",
}


def is_violation_class(class_name: str) -> bool:
    """Checks if a detected class name directly indicates missing PPE."""
    norm = "_".join(class_name.strip().lower().replace("-", " ").split())
    return norm in EXPLICIT_VIOLATION_CLASSES


def get_violation_severity(violation_type: str) -> str:
    """Returns severity level ('critical', 'high', 'medium', 'low') based on safety impact."""
    norm = "_".join(violation_type.strip().lower().replace("-", " ").split())
    return SEVERITY_MAPPING.get(norm, "medium")


def determine_shift(dt: Optional[datetime] = None) -> str:
    """
    Categorizes timestamp into standard industrial shifts:
    - Morning Shift: 06:00 to 14:00
    - Evening Shift: 14:00 to 22:00
    - Night Shift:   22:00 to 06:00
    """
    if dt is None:
        dt = datetime.now()
    hour = dt.hour
    if 6 <= hour < 14:
        return "Morning Shift (06:00-14:00)"
    elif 14 <= hour < 22:
        return "Evening Shift (14:00-22:00)"
    else:
        return "Night Shift (22:00-06:00)"


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def box_intersection_area(b1: List[float], b2: List[float]) -> float:
    """Calculates absolute overlap area between two [x1, y1, x2, y2] boxes."""
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    return max(0.0, xB - xA) * max(0.0, yB - yA)


def box_area(b: List[float]) -> float:
    """Returns area of a [x1, y1, x2, y2] box."""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


# =========================================================================
# PERSON-CENTRIC SPATIAL PPE COMPLIANCE & VIOLATION ENGINE
# =========================================================================
def analyze_person_ppe_compliance(
    raw_detections: List[Dict[str, Any]],
    required_ppe_list: Optional[List[str]] = None,
    min_person_height: float = 30.0
) -> List[Dict[str, Any]]:
    """
    Industrial-grade Person-Centric Spatial PPE Compliance Engine.
    
    1. Detects every worker ('Person') in the scene.
    2. Dynamically partitions each worker's bounding box into anatomical zones:
       - Head Region: [x1 - 15%w, y1 - 25%h, x2 + 15%w, y1 + 38%h]
       - Torso Region: [x1 - 10%w, y1 + 15%h, x2 + 10%w, y1 + 72%h]
       - Feet Region: [x1 - 15%w, y1 + 65%h, x2 + 15%w, y2 + 15%h]
       - Hands Region: [x1 - 25%w, y1 + 25%h, x2 + 25%w, y1 + 85%h]
       - Eyes Region: [x1 - 10%w, y1 + 05%h, x2 + 10%w, y1 + 35%h]
    3. Spatially associates detected PPE gear (helmet, vest, boots, gloves, goggles)
       with the corresponding worker using spatial containment and intersection ratios.
    4. Evaluates compliance against mandatory required PPE (default: Helmet + Vest).
    5. Pinpoints workers with missing gear, tagging them with high-visibility red
       violations ('missing_helmet', 'missing_vest'), generating actionable incident tickets.
    """
    if required_ppe_list is None:
        required_ppe_list = ["helmet", "vest"]
    
    # Normalize required gear list
    req_set = {r.strip().lower() for r in required_ppe_list if r.strip()}
    
    # Separate detections into workers, positive gear, explicit negative classes, and others
    workers = []
    gear_items = []
    explicit_viol_items = []
    other_items = []

    for d in raw_detections:
        cls_lower = d["class"].strip().lower()
        if cls_lower == "person":
            # Check minimum size to filter distant blurred background artifacts
            h = d["y2"] - d["y1"]
            if h >= min_person_height:
                workers.append(d)
        elif cls_lower in PPE_COMPLIANT_CLASSES:
            gear_items.append(d)
        elif is_violation_class(cls_lower):
            explicit_viol_items.append(d)
        elif cls_lower != "none":
            other_items.append(d)

    # Output detection list
    processed_detections: List[Dict[str, Any]] = []

    # Map gear items to workers
    # To prevent 1 helmet/vest from satisfying multiple workers, assign each gear to best matching worker
    worker_gear_map: Dict[int, Dict[str, Dict[str, Any]]] = {i: {} for i in range(len(workers))}
    worker_explicit_map: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(workers))}

    for g in gear_items:
        g_box = [g["x1"], g["y1"], g["x2"], g["y2"]]
        g_area = box_area(g_box)
        if g_area <= 0:
            continue
        g_type = g["class"].strip().lower()

        best_worker_idx = None
        best_overlap_score = 0.0

        for i, w in enumerate(workers):
            w_box = [w["x1"], w["y1"], w["x2"], w["y2"]]
            pw = w_box[2] - w_box[0]
            ph = w_box[3] - w_box[1]

            # Define specific target zone based on gear type
            if g_type == "helmet":
                zone = [w_box[0] - 0.15 * pw, w_box[1] - 0.25 * ph, w_box[2] + 0.15 * pw, w_box[1] + 0.38 * ph]
            elif g_type == "vest":
                zone = [w_box[0] - 0.10 * pw, w_box[1] + 0.15 * ph, w_box[2] + 0.10 * pw, w_box[1] + 0.72 * ph]
            elif g_type == "boots":
                zone = [w_box[0] - 0.15 * pw, w_box[1] + 0.65 * ph, w_box[2] + 0.15 * pw, w_box[3] + 0.15 * ph]
            elif g_type == "gloves":
                zone = [w_box[0] - 0.25 * pw, w_box[1] + 0.25 * ph, w_box[2] + 0.25 * pw, w_box[1] + 0.85 * ph]
            elif g_type == "goggles":
                zone = [w_box[0] - 0.10 * pw, w_box[1] + 0.05 * ph, w_box[2] + 0.10 * pw, w_box[1] + 0.35 * ph]
            else:
                zone = w_box

            inter_area = box_intersection_area(g_box, zone)
            overlap_ratio = inter_area / g_area
            # Check general worker body overlap as fallback
            body_inter = box_intersection_area(g_box, w_box)
            body_ratio = body_inter / g_area

            score = max(overlap_ratio, body_ratio * 0.75)
            if score >= 0.20 and score > best_overlap_score:
                best_overlap_score = score
                best_worker_idx = i

        if best_worker_idx is not None:
            # Associate gear with worker
            worker_gear_map[best_worker_idx][g_type] = g

    # Map explicit negative classes (e.g. no_helmet, no_goggle) to workers if overlapping
    for ex in explicit_viol_items:
        ex_box = [ex["x1"], ex["y1"], ex["x2"], ex["y2"]]
        ex_area = box_area(ex_box)
        best_worker_idx = None
        best_score = 0.0

        for i, w in enumerate(workers):
            w_box = [w["x1"], w["y1"], w["x2"], w["y2"]]
            inter = box_intersection_area(ex_box, w_box)
            ratio = inter / ex_area if ex_area > 0 else 0.0
            if ratio >= 0.25 and ratio > best_score:
                best_score = ratio
                best_worker_idx = i

        if best_worker_idx is not None:
            worker_explicit_map[best_worker_idx].append(ex)

    # Now evaluate each worker's compliance
    for i, w in enumerate(workers):
        w_box = [w["x1"], w["y1"], w["x2"], w["y2"]]
        pw = w_box[2] - w_box[0]
        ph = w_box[3] - w_box[1]
        equipped = worker_gear_map[i]
        explicit_viols = worker_explicit_map[i]
        explicit_types = {e["class"].strip().lower() for e in explicit_viols}

        missing_ppe = []
        worker_violations: List[Dict[str, Any]] = []

        # 1. Helmet Check
        if "helmet" in req_set:
            has_helmet = "helmet" in equipped and "no_helmet" not in explicit_types
            if not has_helmet:
                missing_ppe.append("helmet")
                head_box = [
                    round(max(0.0, w_box[0]), 1),
                    round(max(0.0, w_box[1]), 1),
                    round(w_box[2], 1),
                    round(min(w_box[3], w_box[1] + 0.35 * ph), 1)
                ]
                worker_violations.append({
                    "class": "no_helmet",
                    "label": "Missing Helmet (Hardhat)",
                    "violation": True,
                    "severity": "high",
                    "confidence": round(w["confidence"], 4),
                    "x1": head_box[0],
                    "y1": head_box[1],
                    "x2": head_box[2],
                    "y2": head_box[3],
                    "worker_box": w_box,
                    "is_worker_violation": True,
                    "is_ppe": False,
                    "is_person": False,
                    "worker_index": i + 1,
                    "rule": "Mandatory Head Protection (Hardhat) Required"
                })

        # 2. Safety Vest Check
        if "vest" in req_set:
            has_vest = "vest" in equipped and "no_vest" not in explicit_types
            if not has_vest:
                missing_ppe.append("vest")
                torso_box = [
                    round(max(0.0, w_box[0]), 1),
                    round(w_box[1] + 0.20 * ph, 1),
                    round(w_box[2], 1),
                    round(min(w_box[3], w_box[1] + 0.70 * ph), 1)
                ]
                worker_violations.append({
                    "class": "no_vest",
                    "label": "Missing High-Vis Safety Vest",
                    "violation": True,
                    "severity": "high",
                    "confidence": round(w["confidence"], 4),
                    "x1": torso_box[0],
                    "y1": torso_box[1],
                    "x2": torso_box[2],
                    "y2": torso_box[3],
                    "worker_box": w_box,
                    "is_worker_violation": True,
                    "is_ppe": False,
                    "is_person": False,
                    "worker_index": i + 1,
                    "rule": "Mandatory High-Visibility Safety Vest Required"
                })

        # 3. Boots Check
        if "boots" in req_set:
            has_boots = "boots" in equipped and "no_boots" not in explicit_types
            if not has_boots:
                missing_ppe.append("boots")
                feet_box = [
                    round(max(0.0, w_box[0]), 1),
                    round(w_box[1] + 0.70 * ph, 1),
                    round(w_box[2], 1),
                    round(w_box[3], 1)
                ]
                worker_violations.append({
                    "class": "no_boots",
                    "label": "Missing Safety Boots",
                    "violation": True,
                    "severity": "medium",
                    "confidence": round(w["confidence"], 4),
                    "x1": feet_box[0],
                    "y1": feet_box[1],
                    "x2": feet_box[2],
                    "y2": feet_box[3],
                    "worker_box": w_box,
                    "is_worker_violation": True,
                    "is_ppe": False,
                    "is_person": False,
                    "worker_index": i + 1,
                    "rule": "Mandatory Steel-Toe Safety Boots Required"
                })

        # 4. Gloves Check
        if "gloves" in req_set:
            has_gloves = "gloves" in equipped and "no_gloves" not in explicit_types
            if not has_gloves:
                missing_ppe.append("gloves")
                worker_violations.append({
                    "class": "no_gloves",
                    "label": "Missing Protective Gloves",
                    "violation": True,
                    "severity": "low",
                    "confidence": round(w["confidence"], 4),
                    "x1": w_box[0],
                    "y1": round(w_box[1] + 0.35 * ph, 1),
                    "x2": w_box[2],
                    "y2": round(w_box[1] + 0.85 * ph, 1),
                    "worker_box": w_box,
                    "is_worker_violation": True,
                    "is_ppe": False,
                    "is_person": False,
                    "worker_index": i + 1,
                    "rule": "Mandatory Hand Protection Required"
                })

        # 5. Goggles Check
        if "goggles" in req_set:
            has_goggles = "goggles" in equipped and "no_goggle" not in explicit_types
            if not has_goggles:
                missing_ppe.append("goggles")
                worker_violations.append({
                    "class": "no_goggle",
                    "label": "Missing Eye Protection (Goggles)",
                    "violation": True,
                    "severity": "medium",
                    "confidence": round(w["confidence"], 4),
                    "x1": w_box[0],
                    "y1": round(w_box[1] + 0.05 * ph, 1),
                    "x2": w_box[2],
                    "y2": round(w_box[1] + 0.35 * ph, 1),
                    "worker_box": w_box,
                    "is_worker_violation": True,
                    "is_ppe": False,
                    "is_person": False,
                    "worker_index": i + 1,
                    "rule": "Mandatory Eye Protection Required"
                })

        # Configure worker detection item
        is_worker_viol = len(missing_ppe) > 0
        w_copy = dict(w)
        w_copy["is_person"] = True
        w_copy["worker_index"] = i + 1
        w_copy["is_compliant"] = not is_worker_viol
        w_copy["violation"] = is_worker_viol
        w_copy["missing_ppe"] = missing_ppe
        w_copy["equipped_ppe"] = list(equipped.keys())
        w_copy["severity"] = "high" if any(m in ("helmet", "vest") for m in missing_ppe) else ("medium" if is_worker_viol else None)

        if is_worker_viol:
            w_copy["status_pill"] = f"VIOLATION: MISSING {' + '.join(m.upper() for m in missing_ppe)}"
        else:
            w_copy["status_pill"] = f"COMPLIANT WORKER ({' + '.join(r.upper() for r in req_set)} OK)"

        processed_detections.append(w_copy)
        # Add generated missing PPE violations
        processed_detections.extend(worker_violations)

    # Append any explicit negative classes that were NOT associated with a worker (standalone violations)
    assigned_explicit = set()
    for ex_list in worker_explicit_map.values():
        for ex in ex_list:
            assigned_explicit.add(id(ex))

    for ex in explicit_viol_items:
        if id(ex) not in assigned_explicit:
            ex_copy = dict(ex)
            ex_copy["violation"] = True
            ex_copy["severity"] = get_violation_severity(ex["class"])
            ex_copy["is_ppe"] = False
            ex_copy["is_person"] = False
            processed_detections.append(ex_copy)

    # Append compliant gear items (helmets, vests, etc.) for visual clarity
    for g in gear_items:
        g_copy = dict(g)
        g_copy["is_ppe"] = True
        g_copy["violation"] = False
        g_copy["severity"] = None
        g_copy["is_person"] = False
        processed_detections.append(g_copy)

    # Append any other items (e.g. tools, machines)
    for o in other_items:
        o_copy = dict(o)
        o_copy["is_ppe"] = False
        o_copy["violation"] = False
        o_copy["severity"] = None
        o_copy["is_person"] = False
        processed_detections.append(o_copy)

    return processed_detections


def analyze_image_frame(
    model,
    frame_bgr: np.ndarray,
    confidence_thresh: float = 0.35,
    required_ppe_list: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Performs YOLO inference on a single image frame and evaluates
    person-centric spatial PPE compliance.
    Returns annotated BGR frame and complete list of detection/violation dictionaries.
    """
    if required_ppe_list is None:
        required_ppe_list = ["helmet", "vest"]

    results = model.predict(source=frame_bgr, conf=confidence_thresh, verbose=False)[0]

    raw_detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            xyxy = box.xyxy[0].tolist()
            label = str(results.names[cls_id])
            score = float(box.conf[0].item())
            is_viol = is_violation_class(label)
            severity = get_violation_severity(label) if is_viol else None

            raw_detections.append({
                "class": label,
                "confidence": round(score, 4),
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
                "violation": is_viol,
                "severity": severity,
                "is_ppe": label.strip().lower() in PPE_COMPLIANT_CLASSES,
                "is_person": label.strip().lower() == "person"
            })

    # Execute Person-Centric Spatial PPE Compliance Engine
    detections = analyze_person_ppe_compliance(
        raw_detections,
        required_ppe_list=required_ppe_list,
        min_person_height=30.0
    )

    annotated_frame = draw_enterprise_annotations(frame_bgr, detections)
    return annotated_frame, detections


def save_evidence_crop(
    image_bgr: np.ndarray,
    bbox: List[float],
    output_dir: Path,
    prefix: str = "evidence"
) -> Optional[str]:
    """Saves a cropped bounding box with padding as an evidence thumbnail."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        h, w = image_bgr.shape[:2]
        pad = 20
        x1 = max(0, int(bbox[0]) - pad)
        y1 = max(0, int(bbox[1]) - pad)
        x2 = min(w, int(bbox[2]) + pad)
        y2 = min(h, int(bbox[3]) + pad)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence_file = output_dir / f"{prefix}_{now_str}.jpg"
        cv2.imwrite(str(evidence_file), crop)
        return str(evidence_file)
    except Exception:
        return None


class VideoViolationTracker:
    """
    Groups frame-by-frame missing PPE detections into cohesive incident events.
    Avoids generating hundreds of separate violation tickets for the same worker.
    """
    def __init__(self, temporal_window_seconds: float = 2.0, iou_thresh: float = 0.25):
        self.temporal_window = temporal_window_seconds
        self.iou_thresh = iou_thresh
        self.active_events = []  # List of ongoing event dicts
        self.completed_events = []

    def process_detection(
        self,
        detection: Dict[str, Any],
        timestamp_seconds: float,
        frame_bgr: np.ndarray,
        evidence_dir: Path
    ) -> Optional[str]:
        """
        Associates detection with an existing event or creates a new event group.
        Returns the assigned event_group_id.
        """
        if not detection.get("violation"):
            return None

        viol_type = detection["class"]
        box = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]

        matched_event = None
        for ev in self.active_events:
            # Check same category and within time window
            if ev["violation_type"] == viol_type:
                time_diff = timestamp_seconds - ev["last_timestamp"]
                if time_diff <= self.temporal_window:
                    iou = calculate_iou(box, ev["last_box"])
                    if iou >= self.iou_thresh or time_diff <= 0.5:
                        matched_event = ev
                        break

        if matched_event:
            # Update existing event
            matched_event["last_timestamp"] = timestamp_seconds
            matched_event["last_box"] = box
            matched_event["detection_count"] += 1
            if detection["confidence"] > matched_event["peak_confidence"]:
                matched_event["peak_confidence"] = detection["confidence"]
                # Save new peak evidence
                crop_path = save_evidence_crop(frame_bgr, box, evidence_dir, prefix=f"vid_{viol_type}")
                if crop_path:
                    matched_event["evidence_path"] = crop_path
            return matched_event["event_group_id"]
        else:
            # Create new event
            group_id = f"EVT_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"
            crop_path = save_evidence_crop(frame_bgr, box, evidence_dir, prefix=f"vid_{viol_type}")
            new_event = {
                "event_group_id": group_id,
                "violation_type": viol_type,
                "severity": detection.get("severity", "medium"),
                "start_timestamp": timestamp_seconds,
                "last_timestamp": timestamp_seconds,
                "last_box": box,
                "peak_confidence": detection["confidence"],
                "detection_count": 1,
                "evidence_path": crop_path
            }
            self.active_events.append(new_event)
            return group_id


def draw_enterprise_annotations(
    frame_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    zone_label: Optional[str] = None,
    show_hud: bool = True
) -> np.ndarray:
    """
    Renders professional commercial-grade bounding boxes, worker compliance pills,
    PPE gear accents, and HUD zone telemetry onto the image frame.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Enterprise color palette (BGR format for OpenCV)
    COLOR_VIOLATION = (68, 68, 239)    # Red #ef4444 (Non-compliant worker / violation)
    COLOR_COMPLIANT = (129, 185, 16)   # Emerald #10b981 (Compliant worker)
    COLOR_GEAR      = (190, 180, 50)   # Teal/Cyan for verified PPE gear
    COLOR_PERSON    = (199, 132, 2)    # Cyan-Blue #0284c7
    COLOR_NEUTRAL   = (139, 116, 100)  # Slate #64748b

    violations_found = False
    workers_total = 0
    workers_compliant = 0

    # Draw gear items first (so worker boxes overlay cleanly on top)
    for det in detections:
        if det.get("is_ppe") and not det.get("is_person"):
            gx1, gy1 = int(det["x1"]), int(det["y1"])
            gx2, gy2 = int(det["x2"]), int(det["y2"])
            cls_name = det.get("class", "gear").upper()
            conf = float(det.get("confidence", 0.0))

            # Subtle thin box for PPE gear
            cv2.rectangle(out, (gx1, gy1), (gx2, gy2), COLOR_GEAR, 1, cv2.LINE_AA)
            gear_lbl = f"{cls_name} {conf:.0%}"
            cv2.putText(out, gear_lbl, (gx1, max(12, gy1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GEAR, 1, cv2.LINE_AA)

    # Draw specific missing PPE zones if present
    for det in detections:
        if det.get("is_worker_violation"):
            vx1, vy1 = int(det["x1"]), int(det["y1"])
            vx2, vy2 = int(det["x2"]), int(det["y2"])
            # Draw subtle dashed red zone target
            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), COLOR_VIOLATION, 1, cv2.LINE_AA)

    # Draw workers and direct violations
    for det in detections:
        if det.get("is_person"):
            workers_total += 1
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            conf = float(det.get("confidence", 0.0))
            is_viol = bool(det.get("violation"))

            if is_viol:
                violations_found = True
                color = COLOR_VIOLATION
                missing_str = " + ".join(m.upper() for m in det.get("missing_ppe", ["PPE"]))
                label_text = f"VIOLATION: MISSING {missing_str} ({conf:.0%})"
            else:
                workers_compliant += 1
                color = COLOR_COMPLIANT
                label_text = f"COMPLIANT WORKER ({conf:.0%})"

            # Draw main worker bounding box (thick 2px)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw corner accents for high-tech HUD look
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 4:
                # Top-Left
                cv2.line(out, (x1, y1), (x1 + corner_len, y1), color, 3, cv2.LINE_AA)
                cv2.line(out, (x1, y1), (x1, y1 + corner_len), color, 3, cv2.LINE_AA)
                # Top-Right
                cv2.line(out, (x2, y1), (x2 - corner_len, y1), color, 3, cv2.LINE_AA)
                cv2.line(out, (x2, y1), (x2, y1 + corner_len), color, 3, cv2.LINE_AA)
                # Bottom-Left
                cv2.line(out, (x1, y2), (x1 + corner_len, y2), color, 3, cv2.LINE_AA)
                cv2.line(out, (x1, y2), (x1, y2 - corner_len), color, 3, cv2.LINE_AA)
                # Bottom-Right
                cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, 3, cv2.LINE_AA)
                cv2.line(out, (x2, y2), (x2, y2 - corner_len), color, 3, cv2.LINE_AA)

            # Draw label pill
            font_scale = 0.5
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pill_y1 = max(0, y1 - th - 8)
            pill_y2 = y1
            pill_x2 = min(w, x1 + tw + 12)
            cv2.rectangle(out, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
            cv2.putText(out, label_text, (x1 + 6, pill_y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        elif det.get("violation") and not det.get("is_worker_violation") and not det.get("is_person"):
            # Standalone explicit violation (e.g. no_helmet box without person)
            violations_found = True
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            cls_name = det.get("class", "violation").upper()
            conf = float(det.get("confidence", 0.0))
            color = COLOR_VIOLATION
            label_text = f"VIOLATION: {cls_name} ({conf:.0%})"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pill_y1 = max(0, y1 - th - 6)
            cv2.rectangle(out, (x1, pill_y1), (min(w, x1 + tw + 10), y1), color, -1)
            cv2.putText(out, label_text, (x1 + 5, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Top HUD Bar
    if show_hud:
        hud_bg_h = 32
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_bg_h), (15, 23, 42), -1)  # Dark slate #0f172a
        cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hud_text = f"DETECTO AI ENTERPRISE | {now_str}"
        if zone_label:
            hud_text += f" | ZONE: {zone_label}"
        if workers_total > 0:
            hud_text += f" | WORKERS: {workers_total} ({workers_compliant} COMPLIANT)"

        cv2.putText(out, hud_text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

        # Top alert banner if violations present
        if violations_found:
            alert_text = "ALERTS DETECTED"
            (atw, ath), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (w - atw - 24, 4), (w - 8, 28), COLOR_VIOLATION, -1)
            cv2.putText(out, alert_text, (w - atw - 16, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def evaluate_zone_policy(
    detections: List[Dict[str, Any]],
    required_ppe_list: List[str]
) -> List[Dict[str, Any]]:
    """
    Evaluates detections against specific PPE requirements of a Safety Zone.
    Returns list of policy violation items.
    """
    violations = []
    req_norm = {r.strip().lower() for r in required_ppe_list}

    # Evaluate worker-level missing gear
    for d in detections:
        if d.get("is_person") and d.get("violation"):
            for m in d.get("missing_ppe", []):
                if m in req_norm:
                    violations.append({
                        "rule": f"Mandatory {m.title()} Required in Zone",
                        "violation_type": f"no_{m}",
                        "confidence": d.get("confidence", 0.0),
                        "box": [d["x1"], d["y1"], d["x2"], d["y2"]],
                        "severity": get_violation_severity(f"no_{m}")
                    })
        elif d.get("violation") and not d.get("is_person"):
            cls_clean = d["class"].lower().replace("no_", "").replace("missing_", "")
            if cls_clean in req_norm:
                violations.append({
                    "rule": f"Mandatory {cls_clean.title()} Required in Zone",
                    "violation_type": d["class"],
                    "confidence": d.get("confidence", 0.0),
                    "box": [d["x1"], d["y1"], d["x2"], d["y2"]],
                    "severity": get_violation_severity(d["class"])
                })

    return violations


def benchmark_model_inference(model, sample_frame_bgr: np.ndarray, runs: int = 5) -> Dict[str, Any]:
    """
    Executes multiple inference runs to benchmark latency (P50, P90, min, max) and compute FPS.
    """
    import time
    latencies = []
    # Warmup
    model.predict(source=sample_frame_bgr, conf=0.35, verbose=False)

    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        model.predict(source=sample_frame_bgr, conf=0.35, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.9)]
    avg_lat = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    return {
        "runs": runs,
        "avg_ms": round(avg_lat, 1),
        "p50_ms": round(p50, 1),
        "p90_ms": round(p90, 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "approx_fps": round(fps, 1)
    }

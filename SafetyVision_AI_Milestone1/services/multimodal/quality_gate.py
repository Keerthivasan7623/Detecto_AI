import math
from typing import Dict, Any, List, Optional, Tuple
from services.multimodal.config_loader import load_ppe_categories_config

def calculate_box_area(box: List[float]) -> float:
    """Calculates [x1, y1, x2, y2] bounding box area."""
    if len(box) < 4:
        return 0.0
    w = max(0.0, float(box[2]) - float(box[0]))
    h = max(0.0, float(box[3]) - float(box[1]))
    return w * h

def calculate_box_centroid(box: List[float]) -> Tuple[float, float]:
    """Calculates center (cx, cy) of [x1, y1, x2, y2] box."""
    if len(box) < 4:
        return (0.0, 0.0)
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return (cx, cy)

def calculate_iou(b1: List[float], b2: List[float]) -> float:
    """Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    if len(b1) < 4 or len(b2) < 4:
        return 0.0
    xa = max(b1[0], b2[0])
    ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculates 2D distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class QualityGate:
    """
    Evaluates violation events against confidence thresholds, minimum bounding box area,
    and temporal persistence (N-of-M frame persistence).
    
    Starting defaults:
    - Confidence threshold: 0.5
    - Min bbox area: 0.0 (inactive by default)
    - Temporal persistence: 3 of 5 frames, IoU >= 0.25, centroid dist <= 50 px
    """
    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        min_bbox_area: Optional[float] = None,
        persistence_n: Optional[int] = None,
        persistence_m: Optional[int] = None,
        iou_threshold: Optional[float] = None,
        centroid_dist_threshold: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        if config is None:
            config = load_ppe_categories_config()
        self.config = config

        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else float(config.get("confidence_threshold", 0.5))
        )
        self.min_bbox_area = (
            min_bbox_area
            if min_bbox_area is not None
            else float(config.get("min_bbox_area", 0.0))
        )

        pers_cfg = config.get("temporal_persistence", {})
        self.persistence_n = persistence_n if persistence_n is not None else int(pers_cfg.get("n", 3))
        self.persistence_m = persistence_m if persistence_m is not None else int(pers_cfg.get("m", 5))
        self.iou_threshold = iou_threshold if iou_threshold is not None else float(pers_cfg.get("iou_threshold", 0.25))
        self.centroid_dist_threshold = (
            centroid_dist_threshold
            if centroid_dist_threshold is not None
            else float(pers_cfg.get("centroid_dist_threshold", 50.0))
        )

    def evaluate_static_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single event for confidence and bbox area.
        Skips bbox area check for synthetic no_vest events.
        """
        reasons = []
        conf = float(event.get("confidence", 0.0))
        if conf < self.confidence_threshold:
            reasons.append(f"Confidence {conf:.4f} < threshold {self.confidence_threshold:.4f}")

        # Synthetic no_vest check
        vtype = str(event.get("violation_type", "")).lower().replace("-", "_")
        is_synthetic = bool(event.get("is_synthetic")) or ("no_vest" in vtype)

        if not is_synthetic and self.min_bbox_area > 0.0:
            area = calculate_box_area(event.get("bbox", []))
            if area < self.min_bbox_area:
                reasons.append(f"Bbox area {area:.1f} px^2 < minimum {self.min_bbox_area:.1f} px^2")

        evaluated = dict(event)
        if reasons:
            evaluated["quality_status"] = "low_confidence"
            evaluated["rejection_reasons"] = reasons
            evaluated["passed_gate"] = False
        else:
            evaluated["quality_status"] = "potential_violation"
            evaluated["rejection_reasons"] = []
            evaluated["passed_gate"] = True

        return evaluated

    def evaluate_frame_sequence(
        self,
        frame_detections: List[List[Dict[str, Any]]],
        camera_id: str = "CAM-01",
        zone: str = "Z-01"
    ) -> List[Dict[str, Any]]:
        """
        Evaluates N-of-M persistence across a sequence of M frames.
        
        Args:
            frame_detections: List of detection lists, one list per frame (length <= M).
            camera_id: camera context.
            zone: zone context.
            
        Returns:
            List of evaluated structured events. Clusters meeting >= N occurrences
            are marked 'potential_violation'; clusters with < N occurrences are
            marked 'low_confidence'.
        """
        from services.multimodal.event_builder import build_event_from_detection

        # Cluster detections across frames by category and spatial closeness
        clusters: List[Dict[str, Any]] = []

        for frame_idx, frame_dets in enumerate(frame_detections):
            for det in frame_dets:
                # Basic confidence check before tracking
                conf = float(det.get("confidence", 0.0))
                box = [float(det.get("x1", 0)), float(det.get("y1", 0)), float(det.get("x2", 0)), float(det.get("y2", 0))]
                vtype = str(det.get("class") or det.get("class_name") or "").strip().lower().replace("-", "_")
                vcat = self.config.get("category_mapping", {}).get(vtype, "General")

                matched = None
                for c in clusters:
                    if c["camera_id"] == camera_id and c["zone"] == zone and c["violation_category"] == vcat:
                        last_box = c["last_box"]
                        iou = calculate_iou(box, last_box)
                        dist = euclidean_distance(calculate_box_centroid(box), calculate_box_centroid(last_box))
                        if iou >= self.iou_threshold or dist <= self.centroid_dist_threshold:
                            matched = c
                            break

                if matched is not None:
                    matched["occurrences"].append(frame_idx)
                    matched["last_box"] = box
                    if conf > matched["peak_confidence"]:
                        matched["peak_confidence"] = conf
                        matched["best_detection"] = det
                else:
                    clusters.append({
                        "camera_id": camera_id,
                        "zone": zone,
                        "violation_type": vtype,
                        "violation_category": vcat,
                        "occurrences": [frame_idx],
                        "last_box": box,
                        "peak_confidence": conf,
                        "best_detection": det
                    })

        results = []
        for c in clusters:
            best_det = dict(c["best_detection"])
            best_det["confidence"] = c["peak_confidence"]
            event = build_event_from_detection(
                detection=best_det,
                camera_id=c["camera_id"],
                zone=c["zone"],
                config=self.config
            )

            # Evaluate static properties (conf & bbox)
            evaluated = self.evaluate_static_event(event)
            unique_frames_count = len(set(c["occurrences"]))

            # Evaluate temporal persistence
            if evaluated["passed_gate"]:
                if unique_frames_count < self.persistence_n:
                    evaluated["quality_status"] = "low_confidence"
                    evaluated["passed_gate"] = False
                    evaluated["rejection_reasons"].append(
                        f"Temporal persistence failed: {unique_frames_count} of {len(frame_detections)} frames (requires >= {self.persistence_n})"
                    )
                else:
                    evaluated["quality_status"] = "potential_violation"
                    evaluated["passed_gate"] = True

            evaluated["persistence_frames_observed"] = unique_frames_count
            results.append(evaluated)

        return results

import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from services.multimodal.config_loader import load_ppe_categories_config

def generate_event_id(dt: Optional[datetime] = None) -> str:
    """
    Generates a unique structured event ID with timestamp and full uuid4 hex.
    Format: EVT-YYYYMMDD-HHMMSS-<32_char_hex>
    """
    if dt is None:
        dt = datetime.now()
    timestamp_str = dt.strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid.uuid4().hex
    return f"EVT-{timestamp_str}-{unique_suffix}"

def is_legacy_event_id(event_id: str) -> bool:
    """Returns True if the event ID is a legacy backfilled ID."""
    return str(event_id).startswith("EVT-LEGACY-")

def parse_event_id(event_id: str) -> Dict[str, Any]:
    """
    Parses a new-format event ID.
    Enforces build-time rule: Legacy event_id values use the EVT-LEGACY- prefix
    and must never be parsed as new-format IDs.
    """
    if is_legacy_event_id(event_id):
        raise ValueError(
            f"Legacy event_id '{event_id}' uses 'EVT-LEGACY-' prefix and cannot be parsed as a new-format ID."
        )

    pattern = r"^EVT-(\d{8})-(\d{6})-([0-9a-f]{32})$"
    match = re.match(pattern, str(event_id))
    if not match:
        raise ValueError(f"Invalid new-format event_id: '{event_id}'")

    date_str, time_str, hex_suffix = match.groups()
    parsed_dt = datetime.strptime(f"{date_str}-{time_str}", "%Y%m%d-%H%M%S")
    return {
        "event_id": event_id,
        "timestamp": parsed_dt,
        "uuid_hex": hex_suffix,
        "is_legacy": False
    }

def resolve_evidence_snapshot(violation_type: str, explicit_path: Optional[str] = None) -> str:
    """
    Resolves evidence snapshot path. If not explicitly given, points to evidence/
    named by violation type (e.g. no_helmet, no_vest, etc.).
    """
    if explicit_path:
        return explicit_path

    norm_type = str(violation_type).strip().lower().replace("-", "_")
    candidates = [
        Path(f"evidence/{norm_type}.jpg"),
        Path(f"evidence/{norm_type}.png"),
        Path(f"outputs/evidence/{norm_type}.jpg"),
        Path(f"outputs/evidence/{norm_type}.png"),
    ]
    for c in candidates:
        if c.exists():
            return str(c).replace("\\", "/")

    return f"evidence/{norm_type}.jpg"

def build_event_from_detection(
    detection: Dict[str, Any],
    camera_id: str = "CAM-01",
    zone: str = "Z-01",
    snapshot_path: Optional[str] = None,
    quality_status: Optional[str] = None,
    timestamp: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Converts a raw detection dictionary into a structured multimodal event dictionary.
    
    Fields populated:
    - event_id: unique EVT-... identifier
    - camera_id: identifier of camera
    - zone: workspace zone (e.g. 'Z-01', 'ZONE-CRANE-04')
    - violation_category: category mapped from class label (Head, Eyes, Hands, Torso, Feet)
    - confidence: detection confidence (float)
    - bbox: [x1, y1, x2, y2]
    - timestamp: ISO 8601 string
    - quality_status: quality assessment ('potential_violation' or 'low_confidence')
    - snapshot_path: path to evidence image
    """
    if config is None:
        config = load_ppe_categories_config()

    cat_map = config.get("category_mapping", {})
    raw_class = str(detection.get("class") or detection.get("class_name") or "unknown").strip()
    norm_class = raw_class.lower().replace("-", "_")
    violation_category = cat_map.get(raw_class) or cat_map.get(norm_class) or "General"

    conf = float(detection.get("confidence", 0.0))

    # Bbox handling
    x1 = detection.get("x1")
    y1 = detection.get("y1")
    x2 = detection.get("x2")
    y2 = detection.get("y2")
    if None not in (x1, y1, x2, y2):
        bbox = [float(x1), float(y1), float(x2), float(y2)]
    else:
        # Check if worker_box exists for synthetic no_vest
        worker_box = detection.get("worker_box")
        if worker_box and len(worker_box) == 4:
            wx1, wy1, wx2, wy2 = [float(v) for v in worker_box]
            ph = max(0.0, wy2 - wy1)
            # Default unvalidated heuristic formula for torso box
            bbox = [wx1, round(wy1 + 0.20 * ph, 1), wx2, round(min(wy2, wy1 + 0.70 * ph), 1)]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]

    ts = timestamp or detection.get("timestamp") or datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    snap = resolve_evidence_snapshot(raw_class, snapshot_path or detection.get("snapshot_path") or detection.get("evidence_path"))

    event = {
        "event_id": generate_event_id(),
        "camera_id": str(camera_id),
        "zone": str(zone),
        "violation_type": raw_class,
        "violation_category": violation_category,
        "confidence": round(conf, 4),
        "bbox": bbox,
        "timestamp": ts,
        "quality_status": quality_status,
        "snapshot_path": snap,
        "is_synthetic": bool(detection.get("is_worker_violation") or norm_class in ("no_vest", "missing_vest"))
    }
    return event

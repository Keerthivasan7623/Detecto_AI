from typing import Dict, Any, Optional, List
from services.agent.safety_query import answer_safety_query
from services.multimodal.policy_matcher import PolicyMatcher
from services.multimodal.event_builder import build_event_from_detection

def answer_question(
    question: str,
    zone: Optional[str] = None,
    detection: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Exposes grounded safety Q&A (answer_safety_query) and zone-first policy matching
    through a unified user-facing function.
    
    Args:
        question: Natural language question regarding safety policy, visual detection, or guidance.
        zone: Optional explicit workplace zone identifier (e.g. "Zone Z-01", "Zone Z-02").
        detection: Optional optical detection context dictionary.
        
    Returns:
        Structured response dictionary:
        {
            "answer": str,
            "citations": List[Dict[str, Any]],  # with keys: document, page, section, row
            "grounded": bool,
            "status": str,
            "detection_summary": Optional[Dict[str, Any]]
        }
    """
    policy_matcher = PolicyMatcher()

    target_zone = zone
    if detection and not target_zone:
        target_zone = detection.get("zone")

    # 1. Coordinate zone-first policy matching
    policy_match = None
    if detection is not None:
        event = build_event_from_detection(detection, zone=target_zone or "Zone Z-01")
        policy_match = policy_matcher.match_policy_for_event(event, query=question)
    elif target_zone:
        policy_match = policy_matcher.match_by_category_and_zone("PPE", target_zone)

    # 2. Call existing grounded Q&A service
    rag_result = answer_safety_query(
        question=question,
        detection=detection,
        zone=target_zone
    )

    raw_status = rag_result.get("status", "not_found")
    is_grounded = bool(rag_result.get("grounded", False))
    raw_citations = rag_result.get("citations", [])

    # If policy matcher found citations and rag_result did not, fuse them
    if not raw_citations and policy_match and policy_match.get("has_policy_match"):
        raw_citations = policy_match.get("citations", [])
        is_grounded = True
        raw_status = "answered"

    # Normalize citations with exact keys: (document, page, section, row)
    normalized_citations: List[Dict[str, Any]] = []
    if is_grounded and raw_citations and raw_status not in ("not_found", "out_of_scope"):
        for c in raw_citations:
            doc = c.get("document") or c.get("file_name") or ""
            sec = c.get("section") or ""
            page = c.get("page")
            row = c.get("row")
            if page is None and isinstance(c.get("row_or_page"), int) and "row" not in str(c.get("section", "")).lower():
                page = c.get("row_or_page")
            if row is None and isinstance(c.get("row_or_page"), int) and "row" in str(c.get("section", "")).lower():
                row = c.get("row_or_page")

            normalized_citations.append({
                "document": doc,
                "page": page,
                "section": sec,
                "row": row
            })

    # Format answer text and handle out-of-scope / not-found queries
    if raw_status in ("not_found", "out_of_scope") or not is_grounded or not normalized_citations:
        status = "not_found"
        grounded = False
        normalized_citations = []
        answer_text = "Not found in the knowledge base."
    else:
        status = "answered"
        grounded = True
        answer_text = rag_result.get("answer", "")

    # Build detection_summary and enforce Step 2 rule:
    # "Every answer that includes a detection must state its confidence and 'pending human review.'"
    detection_summary = None
    if detection is not None:
        conf = float(detection.get("confidence", 0.0))
        detection_summary = {
            "class": detection.get("class") or detection.get("class_name") or "unknown",
            "confidence": conf,
            "zone": target_zone or detection.get("zone"),
            "camera_id": detection.get("camera_id"),
            "timestamp": detection.get("timestamp"),
            "review_status": "pending human review"
        }

        det_class = detection_summary["class"]
        notice = f"\n\n[Detection Context: {det_class} | Confidence: {conf:.2f} ({conf:.0%}) | Status: pending human review]"
        if "pending human review" not in answer_text.lower():
            answer_text = answer_text + notice

    return {
        "answer": answer_text,
        "citations": normalized_citations,
        "grounded": grounded,
        "status": status,
        "detection_summary": detection_summary
    }

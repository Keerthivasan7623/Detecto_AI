from typing import Any, Dict, List, Optional, Set
from database import get_connection, log_audit_action, record_review_decision, get_rejected_event_ids
from services.multimodal.policy_matcher import PolicyMatcher


def get_review_queue(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves safety events pending human review.
    Filter criteria:
    - quality_status = 'potential_violation'
    - event_id IS NOT NULL and event_id != ''
    - event_id NOT IN (SELECT event_id FROM review_decisions)
    - Low-confidence events (confidence < 0.5) are strictly excluded.
    
    Each item is enriched with confidence, policy_citation, and snapshot_path.
    """
    limit = max(1, min(int(limit), 200))
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, camera_id, zone_id, zone, violation_type, violation_category,
                   confidence, bounding_box, snapshot_path, timestamp, event_id,
                   quality_status, status, resolution_status
            FROM safety_events
            WHERE quality_status = 'potential_violation'
              AND event_id IS NOT NULL
              AND event_id != ''
              AND (confidence IS NULL OR confidence >= 0.5)
              AND event_id NOT IN (SELECT event_id FROM review_decisions)
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    queue_items = [dict(r) for r in rows]
    
    # Enrich with policy citation via PolicyMatcher
    pm = None
    for item in queue_items:
        # Guarantee confidence and snapshot_path are present
        if item.get("confidence") is None:
            item["confidence"] = 0.0
        else:
            item["confidence"] = float(item["confidence"])
        if item.get("snapshot_path") is None:
            item["snapshot_path"] = ""

        citation_str = "Standard EHS Safety Policy"
        try:
            if pm is None:
                pm = PolicyMatcher()
            match_res = pm.match_policy_for_event(item)
            cits = match_res.get("citations", [])
            item["citations"] = cits
            if cits:
                c0 = cits[0]
                doc = c0.get("document", "EHS Policy")
                sec = c0.get("section", "General")
                pg = c0.get("page", 1)
                citation_str = f"{doc} (Sec: {sec}, p.{pg})"
        except Exception:
            item["citations"] = []

        item["policy_citation"] = citation_str

    return queue_items


def approve_event(
    event_id: str,
    reviewer_user_id: int,
    note: str = "",
    role: str = "admin"
) -> Dict[str, Any]:
    """
    Approves a safety event in human review.
    - Validates reviewer_user_id (must be a positive integer).
    - Validates role (standard users cannot approve).
    - Writes one record into review_decisions.
    - Leaves violations.review_status unchanged.
    - Audits decision with log_audit_action.
    """
    if reviewer_user_id is None or not isinstance(reviewer_user_id, int) or reviewer_user_id <= 0:
        raise ValueError("reviewer_user_id must be a positive integer")
    if role == "standard":
        raise PermissionError("Standard user does not have permission to approve safety events")
    if not event_id or not str(event_id).strip():
        raise ValueError("event_id must be a non-empty string")

    clean_event_id = str(event_id).strip()
    decision_id = record_review_decision(
        event_id=clean_event_id,
        reviewer_user_id=reviewer_user_id,
        decision="approved",
        note=note or ""
    )

    log_audit_action(
        user_id=reviewer_user_id,
        action="review_decision_approved",
        target_type="safety_event",
        target_id=None,
        details=f"Approved event {clean_event_id}. Note: {note}"
    )

    return {
        "decision_id": decision_id,
        "event_id": clean_event_id,
        "reviewer_user_id": reviewer_user_id,
        "decision": "approved",
        "note": note,
        "status": "success"
    }


def reject_event(
    event_id: str,
    reviewer_user_id: int,
    note: str = "",
    role: str = "admin"
) -> Dict[str, Any]:
    """
    Rejects a safety event in human review.
    - Validates reviewer_user_id (must be a positive integer).
    - Validates role (standard users cannot reject).
    - Writes one record into review_decisions.
    - Leaves violations.review_status unchanged.
    - Audits decision with log_audit_action.
    """
    if reviewer_user_id is None or not isinstance(reviewer_user_id, int) or reviewer_user_id <= 0:
        raise ValueError("reviewer_user_id must be a positive integer")
    if role == "standard":
        raise PermissionError("Standard user does not have permission to reject safety events")
    if not event_id or not str(event_id).strip():
        raise ValueError("event_id must be a non-empty string")

    clean_event_id = str(event_id).strip()
    decision_id = record_review_decision(
        event_id=clean_event_id,
        reviewer_user_id=reviewer_user_id,
        decision="rejected",
        note=note or ""
    )

    log_audit_action(
        user_id=reviewer_user_id,
        action="review_decision_rejected",
        target_type="safety_event",
        target_id=None,
        details=f"Rejected event {clean_event_id}. Note: {note}"
    )

    return {
        "decision_id": decision_id,
        "event_id": clean_event_id,
        "reviewer_user_id": reviewer_user_id,
        "decision": "rejected",
        "note": note,
        "status": "success"
    }


def filter_report_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters report input items so that rejected events are excluded.
    Reports exclude only items with decision = 'rejected' in review_decisions.
    Approved items and unreviewed historical items are included.
    """
    rejected_eids = get_rejected_event_ids()

    filtered = []
    for item in items:
        eid = str(item.get("event_id") or "")
        if eid and eid in rejected_eids:
            continue
        if item.get("decision") == "rejected" or item.get("review_decision") == "rejected":
            continue
        filtered.append(item)
    return filtered


def get_reportable_events(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Retrieves safety events eligible for formal reporting.
    Only approved events and unreviewed historical events flow into reports.
    Reports exclude any event where decision = 'rejected'.
    Pending unreviewed events remain in the review queue until approved.
    """
    limit = max(1, min(int(limit), 1000))
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT s.*, rd.decision AS review_decision, rd.reviewer_user_id, rd.decided_at, rd.note AS review_notes
            FROM safety_events s
            LEFT JOIN review_decisions rd ON rd.event_id = s.event_id
            WHERE (rd.decision = 'approved' OR (rd.decision IS NULL AND s.event_id LIKE 'EVT-LEGACY-%'))
              AND (rd.decision IS NULL OR rd.decision != 'rejected')
              AND s.quality_status = 'potential_violation'
            ORDER BY s.id DESC LIMIT ?
        """, (limit,)).fetchall()

    events = []
    for r in rows:
        d = dict(r)
        dec = d.get("review_decision")
        if dec == "approved":
            d["review_status"] = "approved"
        elif not dec:
            d["review_status"] = "unreviewed historical"
        else:
            d["review_status"] = dec

        # Ensure compatibility with reporting generator fields
        if "created_at" not in d or not d["created_at"]:
            d["created_at"] = d.get("timestamp") or ""
        if "violation_id" not in d or not d["violation_id"]:
            d["violation_id"] = d.get("event_id") or d.get("id")
        events.append(d)

    return events

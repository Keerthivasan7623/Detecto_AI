"""
Detecto AI - Milestone 3: 5-Node Grounded Safety Workflow
Orchestrates Query Analysis, Visual Retrieval, Document Retrieval, Generation,
and Strict Grounding Validation using LangGraph StateGraph.

Rules enforced:
- Offline, deterministic execution (no LLM).
- Reuses existing functions: classify_safety_intent, answer_safety_query, answer_question, PolicyMatcher.
- Leaves services/agent/graph.py unchanged.
- Node 4 only restates text from retrieved chunks (plus detection confidence & pending human review).
- Audit logs store counts and summaries, not full chunk text.
- Low-confidence events are strictly excluded from visual retrieval.
"""

import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from database import get_safety_events, log_audit_action
from services.agent.safety_query import classify_safety_intent, answer_safety_query
from services.multimodal.policy_matcher import PolicyMatcher
from services.multimodal.qa_service import answer_question

INTENT_MAPPING = {
    "policy_inquiry": "policy",
    "policy": "policy",
    "audit_check": "visual",
    "visual": "visual",
    "investigation": "combined",
    "combined": "combined",
    "remediation": "combined",
    "guidance": "guidance",
    "out_of_scope": "out_of_scope"
}

class SafetyWorkflowState(TypedDict, total=False):
    question: str
    zone: Optional[str]
    detection: Optional[Dict[str, Any]]
    intent: str
    visual_events: List[Dict[str, Any]]
    detection_summary: Optional[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    candidate_answer: str
    final_answer: str
    status: str
    grounded: bool
    user_id: Optional[int]

# -------------------------------------------------------------
# NODE 1: QUERY ANALYSIS
# -------------------------------------------------------------
def query_analysis_node(state: SafetyWorkflowState) -> Dict[str, Any]:
    """Classifies intent using Task 1 canonical classifier and isolates zone context."""
    question = state.get("question", "").strip()
    detection = state.get("detection")

    raw_intent = classify_safety_intent(question, detection)
    intent = INTENT_MAPPING.get(raw_intent, "policy")

    # Extract or preserve zone
    zone = state.get("zone")
    if not zone and detection and detection.get("zone"):
        zone = detection.get("zone")
    if not zone:
        q_low = question.lower()
        if "z-01" in q_low or "zone a" in q_low or "fabrication" in q_low:
            zone = "Zone Z-01"
        elif "z-02" in q_low or "zone b" in q_low or "chemical" in q_low:
            zone = "Zone Z-02"
        elif "z-03" in q_low or "zone c" in q_low or "loading" in q_low:
            zone = "Zone Z-03"
        elif "z-04" in q_low or "zone d" in q_low or "crane" in q_low:
            zone = "Zone Z-04"

    # Log counts and summaries to audit trail (no full chunk text)
    log_audit_action(
        user_id=state.get("user_id"),
        action="WORKFLOW_NODE_QUERY_ANALYSIS",
        target_type="langgraph_workflow",
        details=json.dumps({
            "intent": intent,
            "has_zone": bool(zone),
            "has_detection": bool(detection)
        })
    )

    return {
        "intent": intent,
        "zone": zone
    }

def route_after_query_analysis(state: SafetyWorkflowState) -> str:
    """Conditional router based on canonical intent."""
    intent = state.get("intent", "policy")
    if intent == "out_of_scope":
        return "out_of_scope"
    elif intent in ("visual", "combined"):
        return "visual_retrieval"
    else:  # policy or guidance
        return "document_retrieval"

# -------------------------------------------------------------
# NODE 2: VISUAL RETRIEVAL
# -------------------------------------------------------------
def visual_retrieval_node(state: SafetyWorkflowState) -> Dict[str, Any]:
    """Fetches qualifying events from safety_events (quality_status = 'potential_violation' only)."""
    zone = state.get("zone")
    detection = state.get("detection")

    # Fetch potential_violation events only (low-confidence excluded)
    events = get_safety_events(
        zone=zone,
        quality_status="potential_violation",
        limit=10
    )

    # Build detection_summary if detection dict passed
    det_summary = None
    if detection is not None:
        conf = float(detection.get("confidence", 0.0))
        det_summary = {
            "class": detection.get("class") or detection.get("class_name") or "unknown",
            "confidence": conf,
            "zone": zone or detection.get("zone"),
            "camera_id": detection.get("camera_id"),
            "timestamp": detection.get("timestamp"),
            "review_status": "pending human review"
        }

    # Log summary to audit trail
    log_audit_action(
        user_id=state.get("user_id"),
        action="WORKFLOW_NODE_VISUAL_RETRIEVAL",
        target_type="langgraph_workflow",
        details=json.dumps({
            "retrieved_events_count": len(events),
            "has_detection_summary": bool(det_summary)
        })
    )

    return {
        "visual_events": events,
        "detection_summary": det_summary
    }

def route_after_visual_retrieval(state: SafetyWorkflowState) -> str:
    """Routes combined queries to document retrieval, visual-only directly to generation."""
    intent = state.get("intent", "visual")
    if intent == "combined":
        return "document_retrieval"
    return "generation"

# -------------------------------------------------------------
# NODE 3: DOCUMENT RETRIEVAL
# -------------------------------------------------------------
def document_retrieval_node(state: SafetyWorkflowState) -> Dict[str, Any]:
    """Calls answer_question and PolicyMatcher for zone-first matches."""
    question = state.get("question", "")
    zone = state.get("zone")
    detection = state.get("detection")
    intent = state.get("intent", "policy")

    # Call unified qa_service (which calls answer_safety_query & PolicyMatcher)
    qa_res = answer_question(
        question=question,
        zone=zone,
        detection=detection
    )

    citations = qa_res.get("citations", [])

    # If guidance intent: enforce exclusion of mandatory matrix/penalty titles
    if intent == "guidance" and citations:
        citations = [
            c for c in citations
            if not any(k in (c.get("section") or "").upper() for k in ["MATRIX", "MANDATORY", "PENALT"])
        ]

    # Log counts and cited document basenames to audit trail (no full text)
    log_audit_action(
        user_id=state.get("user_id"),
        action="WORKFLOW_NODE_DOCUMENT_RETRIEVAL",
        target_type="langgraph_workflow",
        details=json.dumps({
            "citations_count": len(citations),
            "cited_documents": list(set(c.get("document", "") for c in citations if c.get("document")))
        })
    )

    return {
        "citations": citations,
        "candidate_answer": qa_res.get("answer", ""),
        "status": qa_res.get("status", "not_found")
    }

# -------------------------------------------------------------
# NODE 4: GENERATION
# -------------------------------------------------------------
def generation_node(state: SafetyWorkflowState) -> Dict[str, Any]:
    """
    Builds the answer from the context only, using the deterministic template.
    Restates text from retrieved chunks only. No new requirement sentences.
    Every answer that includes a detection must state its confidence and 'pending human review'.
    """
    intent = state.get("intent", "policy")
    detection = state.get("detection")
    det_summary = state.get("detection_summary")
    citations = state.get("citations", [])
    candidate_answer = state.get("candidate_answer", "")

    # For visual-only queries without doc retrieval
    if intent == "visual" and not candidate_answer:
        events = state.get("visual_events", [])
        if events:
            ev = events[0]
            candidate_answer = (
                f"Camera event recorded in {ev.get('zone', 'Zone')}: "
                f"Violation {ev.get('violation_type')} flagged with optical confidence {ev.get('confidence', 0.0):.2f}."
            )
        else:
            candidate_answer = "No optical violation events recorded for this context."

    # Enforce detection confidence & pending human review
    if (detection or det_summary) and "pending human review" not in candidate_answer.lower():
        conf = float(detection.get("confidence", 0.0)) if detection else float(det_summary.get("confidence", 0.0))
        d_class = detection.get("class", "object") if detection else det_summary.get("class", "object")
        candidate_answer += f"\n\n[Detection Context: {d_class} | Confidence: {conf:.2f} | Status: pending human review]"

    # Log generation summary to audit trail
    log_audit_action(
        user_id=state.get("user_id"),
        action="WORKFLOW_NODE_GENERATION",
        target_type="langgraph_workflow",
        details=json.dumps({
            "answer_length": len(candidate_answer),
            "citations_used": len(citations)
        })
    )

    return {
        "candidate_answer": candidate_answer
    }

# -------------------------------------------------------------
# NODE 5: VALIDATION
# -------------------------------------------------------------
def validation_node(state: SafetyWorkflowState) -> Dict[str, Any]:
    """
    Validates requirement grounding against retrieved chunks.
    If ungrounded or citations empty (and not a visual observation), returns
    'Not found in the knowledge base.'
    """
    intent = state.get("intent", "policy")
    citations = state.get("citations", [])
    candidate = state.get("candidate_answer", "")

    is_valid = True
    if intent in ("policy", "guidance", "combined"):
        if not citations or "not found in the knowledge base" in candidate.lower():
            is_valid = False

    if is_valid and candidate:
        final_answer = candidate
        status = "answered"
        grounded = True
    else:
        final_answer = "Not found in the knowledge base."
        status = "not_found"
        grounded = False
        citations = []

    # Log validation result to audit trail
    log_audit_action(
        user_id=state.get("user_id"),
        action="WORKFLOW_NODE_VALIDATION",
        target_type="langgraph_workflow",
        details=json.dumps({
            "validation_passed": is_valid,
            "final_status": status,
            "grounded": grounded
        })
    )

    return {
        "final_answer": final_answer,
        "status": status,
        "grounded": grounded,
        "citations": citations
    }

# -------------------------------------------------------------
# GRAPH COMPILATION
# -------------------------------------------------------------
def build_safety_qa_workflow() -> StateGraph:
    """Constructs and compiles the 5-node LangGraph workflow."""
    workflow = StateGraph(SafetyWorkflowState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("visual_retrieval", visual_retrieval_node)
    workflow.add_node("document_retrieval", document_retrieval_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("validation", validation_node)

    workflow.add_edge(START, "query_analysis")

    workflow.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "out_of_scope": END,
            "visual_retrieval": "visual_retrieval",
            "document_retrieval": "document_retrieval"
        }
    )

    workflow.add_conditional_edges(
        "visual_retrieval",
        route_after_visual_retrieval,
        {
            "document_retrieval": "document_retrieval",
            "generation": "generation"
        }
    )

    workflow.add_edge("document_retrieval", "generation")
    workflow.add_edge("generation", "validation")
    workflow.add_edge("validation", END)

    return workflow.compile()

# Singleton compiled app
_WORKFLOW_APP = None

def get_safety_qa_workflow_app():
    global _WORKFLOW_APP
    if _WORKFLOW_APP is None:
        _WORKFLOW_APP = build_safety_qa_workflow()
    return _WORKFLOW_APP

def run_safety_workflow(
    question: str,
    zone: Optional[str] = None,
    detection: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes the 5-node LangGraph grounded workflow.
    """
    app = get_safety_qa_workflow_app()
    initial_state: SafetyWorkflowState = {
        "question": question,
        "zone": zone,
        "detection": detection,
        "user_id": user_id
    }
    result = app.invoke(initial_state)

    # In out_of_scope immediate termination
    if result.get("intent") == "out_of_scope":
        return {
            "intent": "out_of_scope",
            "status": "out_of_scope",
            "answer": "This inquiry is outside the scope of the workplace safety knowledge base and PPE monitoring system.",
            "citations": [],
            "grounded": False,
            "detection_summary": None
        }

    return {
        "intent": result.get("intent"),
        "status": result.get("status", "answered"),
        "answer": result.get("final_answer") or result.get("candidate_answer", ""),
        "citations": result.get("citations", []),
        "grounded": bool(result.get("grounded", False)),
        "detection_summary": result.get("detection_summary")
    }

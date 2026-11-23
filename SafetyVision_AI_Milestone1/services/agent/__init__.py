"""
Detecto AI - Autonomous Multimodal EHS Investigation Agent Package
Milestone 3: LangGraph Agentic Orchestrator, Computer Vision Telemetry Fusion,
Vector Store Document Intelligence, Dual-Level History Management, and Automated Remediations.
"""

from services.agent.tools import (
    query_visual_events,
    search_safety_policies,
    create_incident_audit_ticket,
    dispatch_emergency_alert,
    visual_inspector_tool,
    doc_retriever_tool,
    incident_logger_tool,
    supervisor_alert_tool
)

from services.agent.graph import (
    AgentState,
    build_incident_agent_graph,
    run_incident_investigation_agent,
    intent_and_parameter_parser,
    visual_correlator,
    policy_cross_referencer,
    multimodal_synthesizer,
    remediation_dispatcher,
    # Aliases
    build_visiondesk_graph,
    run_visiondesk_agent,
    classify_intent,
    visual_inspection_node,
    policy_matching_node,
    multimodal_synthesis_node,
    action_dispatcher_node
)

from services.agent.safety_query import (
    answer_safety_query,
    classify_safety_intent,
)

from services.agent.investigation_workflow import (
    InvestigationState,
    build_investigation_graph,
    run_safety_investigation,
    get_investigation_workflow,
    load_risk_framework
)

__all__ = [
    "answer_safety_query",
    "classify_safety_intent",
    "query_visual_events",
    "search_safety_policies",
    "create_incident_audit_ticket",
    "dispatch_emergency_alert",
    "visual_inspector_tool",
    "doc_retriever_tool",
    "incident_logger_tool",
    "supervisor_alert_tool",
    "AgentState",
    "build_incident_agent_graph",
    "run_incident_investigation_agent",
    "intent_and_parameter_parser",
    "visual_correlator",
    "policy_cross_referencer",
    "multimodal_synthesizer",
    "remediation_dispatcher",
    "build_visiondesk_graph",
    "run_visiondesk_agent",
    "classify_intent",
    "visual_inspection_node",
    "policy_matching_node",
    "multimodal_synthesis_node",
    "action_dispatcher_node",
    "InvestigationState",
    "build_investigation_graph",
    "run_safety_investigation",
    "get_investigation_workflow",
    "load_risk_framework"
]

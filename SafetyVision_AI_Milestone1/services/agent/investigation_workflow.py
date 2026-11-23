"""
SafetyVision AI / Detecto AI - Milestone 3: 6-Node LangGraph AI Safety Investigation
Orchestrates Incident Analysis, Visual Evidence Detection, Safety Policy Retrieval,
Risk Assessment Matrix Calculation, CAPA Corrective Action Planning, and Formal Report Generation.

Architecture:
  Node 1: Incident Analysis Agent (NLP keyword cues, hazard categorization, mandatory PPE identification)
  Node 2: Visual Evidence Analysis Agent (YOLOv8 inference, PPE compliance, missing/unavailable resilience)
  Node 3: Safety Policy Retrieval Agent (RAG vector store search, citation grounding, regulatory gap detection)
  Node 4: Risk Assessment Agent (Severity x Likelihood matrix evaluation, uncertainty annotation)
  Node 5: Corrective Actions Agent (Hierarchy of Controls, grounded citations vs. general best practices)
  Node 6: Report Generation Agent (Formal investigation markdown report, SQLite persistence, audit trail)
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any, Optional

import cv2
import numpy as np
from langgraph.graph import StateGraph, START, END

from database import (
    create_investigation,
    update_investigation,
    add_investigation_stage,
    add_investigation_citation,
    log_audit_action,
)
from services.document_intelligence.rag_service import answer_safety_policy
from safety_engine import analyze_image_frame

logger = logging.getLogger("SafetyInvestigationWorkflow")

# ==========================================
# RISK FRAMEWORK CONFIGURATION LOADER
# ==========================================
DEFAULT_RISK_FRAMEWORK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "risk_framework.json"
)

_CACHED_FRAMEWORK: Optional[Dict[str, Any]] = None


def load_risk_framework(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads risk framework configuration from JSON with fallback defaults."""
    global _CACHED_FRAMEWORK
    path = custom_path or DEFAULT_RISK_FRAMEWORK_PATH

    if not custom_path and _CACHED_FRAMEWORK is not None:
        return _CACHED_FRAMEWORK

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                framework = json.load(f)
                if not custom_path:
                    _CACHED_FRAMEWORK = framework
                return framework
        except Exception as e:
            logger.warning(f"Error loading risk framework from {path}: {e}")

    # Fallback default embedded framework
    fallback = {
        "hazard_categories": {
            "Working at Heights": {
                "mandatory_ppe": ["safety_harness", "helmet", "safety_boots"],
                "severity_baseline": 4,
                "keywords": ["heights", "scaffold", "ladder", "fall", "harness", "elevated", "roof", "platform"]
            },
            "Hot Work & Welding": {
                "mandatory_ppe": ["welding_mask", "face_shield", "flame_resistant_gloves", "safety_boots"],
                "severity_baseline": 3,
                "keywords": ["welding", "hot work", "torch", "spark", "cutting", "flame", "brazing", "grinding"]
            },
            "Electrical Safety": {
                "mandatory_ppe": ["dielectric_gloves", "arc_flash_shield", "insulated_boots", "helmet"],
                "severity_baseline": 4,
                "keywords": ["electrical", "electric", "voltage", "switchgear", "panel", "shock", "loto", "breaker"]
            },
            "Heavy Machinery & Struck-By": {
                "mandatory_ppe": ["high_visibility_vest", "helmet", "steel_toe_boots"],
                "severity_baseline": 3,
                "keywords": ["machinery", "crane", "forklift", "excavator", "traffic", "struck", "vehicle", "load"]
            },
            "Chemical & Hazardous Materials": {
                "mandatory_ppe": ["respirator", "chemical_goggles", "nitrile_gloves", "chemical_apron"],
                "severity_baseline": 3,
                "keywords": ["chemical", "fumes", "toxic", "acid", "solvent", "dust", "vapor", "respirator", "spill"]
            },
            "General Construction & PPE": {
                "mandatory_ppe": ["helmet", "vest", "safety_boots"],
                "severity_baseline": 2,
                "keywords": ["construction", "site", "worker", "helmet", "vest", "boots", "ppe", "housekeeping"]
            }
        },
        "severity_scale": {
            "1": {"label": "Minor", "score": 1},
            "2": {"label": "Moderate", "score": 2},
            "3": {"label": "Serious", "score": 3},
            "4": {"label": "Severe", "score": 4},
            "5": {"label": "Catastrophic", "score": 5}
        },
        "likelihood_scale": {
            "1": {"label": "Rare", "score": 1},
            "2": {"label": "Unlikely", "score": 2},
            "3": {"label": "Possible", "score": 3},
            "4": {"label": "Likely", "score": 4},
            "5": {"label": "Almost Certain", "score": 5}
        },
        "risk_matrix": {
            "thresholds": {
                "Low": {"min": 1, "max": 4, "color": "#16a34a", "action": "Routine control monitoring and standard toolbox briefing."},
                "Medium": {"min": 5, "max": 9, "color": "#d97706", "action": "Supervisory review, corrective action within 24 hours, mandatory PPE audit."},
                "High": {"min": 10, "max": 15, "color": "#ea580c", "action": "Immediate job pause or halt until control measures, permit re-check, and PPE compliance are verified."},
                "Critical": {"min": 16, "max": 25, "color": "#dc2626", "action": "IMMEDIATE STOP WORK ORDER. Evacuate zone, notify EHS Director, lock out hazards, full formal inquiry."}
            }
        },
        "uncertainty_guidelines": {
            "rules": [
                "If visual evidence is missing or marked unavailable, flag visual uncertainty and base findings strictly on verbal narrative and indexed policies.",
                "If detected bounding box confidence is between 0.35 and 0.50, flag detection ambiguity and recommend on-site manual supervisor validation.",
                "If policy similarity score is below 0.20, explicitly document a policy retrieval gap rather than fabricating compliance clauses."
            ]
        }
    }
    return fallback


# ==========================================
# YOLO MODEL SINGLETON LOADER
# ==========================================
_CACHED_YOLO_MODEL = None


def get_yolo_model(model_path: str = "models/best.pt"):
    """Loads and caches YOLO model instance for visual investigation node."""
    global _CACHED_YOLO_MODEL
    if _CACHED_YOLO_MODEL is not None:
        return _CACHED_YOLO_MODEL

    # Resolve model path relative to project root
    if not os.path.isabs(model_path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        resolved_path = os.path.join(root, model_path)
    else:
        resolved_path = model_path

    if os.path.exists(resolved_path):
        try:
            from ultralytics import YOLO
            _CACHED_YOLO_MODEL = YOLO(resolved_path)
            return _CACHED_YOLO_MODEL
        except Exception as e:
            logger.warning(f"Failed to load YOLO model from {resolved_path}: {e}")
            return None
    return None


# ==========================================
# WORKFLOW STATE DEFINITION
# ==========================================
class InvestigationState(TypedDict, total=False):
    # Primary Inputs
    incident_id: str
    title: str
    incident_description: str
    incident_category: Optional[str]
    visual_evidence_path: Optional[str]
    repository_scope: Optional[str]
    user_id: Optional[int]
    custom_risk_framework_path: Optional[str]

    # Node Output Payloads
    incident_analysis: Dict[str, Any]
    visual_evidence: Dict[str, Any]
    policy_context: List[Dict[str, Any]]
    policy_answer_summary: Optional[str]
    risk_assessment: Dict[str, Any]
    corrective_actions: Dict[str, Any]
    report: str

    # Execution & Telemetry Tracking
    stages: Dict[str, Dict[str, Any]]
    errors: List[str]
    investigation_id: Optional[int]


# ==========================================
# NODE 1: INCIDENT ANALYSIS AGENT
# ==========================================
def incident_analysis_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Parses the incident title and narrative description using rule-based NLP.
    Categorizes the hazard type, identifies severity cues, and extracts expected PPE.
    """
    t0 = time.time()
    stages = dict(state.get("stages", {}))
    errors = list(state.get("errors", []))

    description = (state.get("incident_description") or "").strip()
    title = (state.get("title") or "").strip()
    full_text = f"{title} {description}".lower()

    try:
        framework = load_risk_framework(state.get("custom_risk_framework_path"))
        categories_dict = framework.get("hazard_categories", {})

        # Keyword-based category classification
        category_scores: Dict[str, int] = {}
        for cat_name, cat_data in categories_dict.items():
            score = 0
            keywords = cat_data.get("keywords", [])
            for kw in keywords:
                if kw.lower() in full_text:
                    score += 1
            category_scores[cat_name] = score

        explicit_cat = state.get("incident_category")
        if explicit_cat and explicit_cat in categories_dict:
            chosen_category = explicit_cat
        else:
            best_cat = max(category_scores.items(), key=lambda x: x[1])
            chosen_category = best_cat[0] if best_cat[1] > 0 else "General Construction & PPE"

        cat_info = categories_dict.get(chosen_category, {})
        mandatory_ppe = cat_info.get("mandatory_ppe", ["helmet", "vest", "safety_boots"])
        baseline_severity = cat_info.get("severity_baseline", 2)

        # Severity cue extraction
        severity_cue_lexicon = [
            "fall", "scaffold without", "no railing", "unprotected edge", "high voltage",
            "live wire", "unanchored", "no harness", "without helmet", "without vest",
            "fire", "explosion", "spark", "fumes", "collapsed", "fatal", "injury",
            "chemical spill", "struck by", "crane collapse", "caught between"
        ]
        identified_cues = [cue for cue in severity_cue_lexicon if cue in full_text]

        # Entity cues
        entity_lexicon = [
            "scaffolding", "ladder", "crane", "forklift", "excavator", "welding torch",
            "grinder", "breaker panel", "chemical drum", "safety harness", "tether",
            "guardrail", "lifeline", "compressor", "tank", "generator"
        ]
        identified_entities = [ent for ent in entity_lexicon if ent in full_text]

        analysis_output = {
            "identified_category": chosen_category,
            "baseline_severity": baseline_severity,
            "mandatory_ppe": mandatory_ppe,
            "severity_cues": identified_cues,
            "extracted_entities": identified_entities,
            "word_count": len(description.split()),
        }

        exec_ms = int((time.time() - t0) * 1000)
        stages["Incident Analysis"] = {
            "status": "ok",
            "execution_time_ms": exec_ms,
            "output": analysis_output,
            "error": ""
        }

        return {
            "incident_category": chosen_category,
            "incident_analysis": analysis_output,
            "stages": stages,
            "errors": errors
        }

    except Exception as exc:
        logger.error(f"Incident Analysis Node failed: {exc}", exc_info=True)
        exec_ms = int((time.time() - t0) * 1000)
        errors.append(f"Incident Analysis error: {str(exc)}")
        stages["Incident Analysis"] = {
            "status": "failed",
            "execution_time_ms": exec_ms,
            "output": {},
            "error": str(exc)
        }
        return {
            "incident_category": "General Construction & PPE",
            "incident_analysis": {
                "identified_category": "General Construction & PPE",
                "baseline_severity": 2,
                "mandatory_ppe": ["helmet", "vest"],
                "severity_cues": [],
                "extracted_entities": []
            },
            "stages": stages,
            "errors": errors
        }


# ==========================================
# NODE 2: VISUAL EVIDENCE ANALYSIS AGENT
# ==========================================
def visual_evidence_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Executes YOLOv8 object and PPE detection on the uploaded visual evidence.
    Gracefully handles missing, invalid, or corrupted images without crashing.
    """
    t0 = time.time()
    stages = dict(state.get("stages", {}))
    errors = list(state.get("errors", []))
    image_path = state.get("visual_evidence_path")
    incident_id = state.get("incident_id", "INV-UNKNOWN")

    if not image_path or not str(image_path).strip():
        exec_ms = int((time.time() - t0) * 1000)
        stages["Visual Evidence Analysis"] = {
            "status": "unavailable",
            "execution_time_ms": exec_ms,
            "output": {"reason": "No visual evidence file provided. Operating in narrative-only mode."},
            "error": ""
        }
        return {
            "visual_evidence": {
                "status": "unavailable",
                "reason": "No image provided",
                "detections": [],
                "violations": [],
                "persons_count": 0,
                "annotated_image_path": None
            },
            "stages": stages,
            "errors": errors
        }

    # Resolve path
    if not os.path.isabs(image_path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_image_path = os.path.join(root, image_path)
    else:
        full_image_path = image_path

    if not os.path.exists(full_image_path):
        exec_ms = int((time.time() - t0) * 1000)
        stages["Visual Evidence Analysis"] = {
            "status": "unavailable",
            "execution_time_ms": exec_ms,
            "output": {"reason": f"Visual evidence file not found at '{image_path}'."},
            "error": ""
        }
        return {
            "visual_evidence": {
                "status": "unavailable",
                "reason": f"File not found: {image_path}",
                "detections": [],
                "violations": [],
                "persons_count": 0,
                "annotated_image_path": None
            },
            "stages": stages,
            "errors": errors
        }

    try:
        frame_bgr = cv2.imread(full_image_path)
        if frame_bgr is None or frame_bgr.size == 0:
            exec_ms = int((time.time() - t0) * 1000)
            stages["Visual Evidence Analysis"] = {
                "status": "unavailable",
                "execution_time_ms": exec_ms,
                "output": {"reason": "Image file could not be decoded by OpenCV."},
                "error": ""
            }
            return {
                "visual_evidence": {
                    "status": "unavailable",
                    "reason": "Failed to decode image",
                    "detections": [],
                    "violations": [],
                    "persons_count": 0,
                    "annotated_image_path": None
                },
                "stages": stages,
                "errors": errors
            }

        # Run YOLO inference
        model = get_yolo_model()
        if model is None:
            exec_ms = int((time.time() - t0) * 1000)
            stages["Visual Evidence Analysis"] = {
                "status": "failed",
                "execution_time_ms": exec_ms,
                "output": {"reason": "YOLO model weights (best.pt) could not be loaded."},
                "error": "Model missing"
            }
            return {
                "visual_evidence": {
                    "status": "failed",
                    "reason": "YOLO model not loaded",
                    "detections": [],
                    "violations": [],
                    "persons_count": 0,
                    "annotated_image_path": None
                },
                "stages": stages,
                "errors": errors + ["YOLO model weights missing or unreadable"]
            }

        annotated_bgr, detections = analyze_image_frame(model, frame_bgr, confidence_thresh=0.35)

        # Categorize detections
        violations = [d for d in detections if d.get("violation") or d.get("is_violation")]
        persons = [d for d in detections if d.get("is_person") or d.get("class", "").lower() == "person"]

        # Persist annotated evidence image
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(root, "outputs", "investigations")
        os.makedirs(output_dir, exist_ok=True)
        annotated_filename = f"{incident_id}_annotated.jpg"
        annotated_save_path = os.path.join(output_dir, annotated_filename)
        cv2.imwrite(annotated_save_path, annotated_bgr)
        rel_annotated_path = f"outputs/investigations/{annotated_filename}"

        visual_output = {
            "status": "ok",
            "total_detections": len(detections),
            "persons_count": len(persons),
            "violations_count": len(violations),
            "violations": violations,
            "raw_detections": detections,
            "annotated_image_path": rel_annotated_path,
            "source_image_path": image_path
        }

        exec_ms = int((time.time() - t0) * 1000)
        stages["Visual Evidence Analysis"] = {
            "status": "ok",
            "execution_time_ms": exec_ms,
            "output": visual_output,
            "error": ""
        }

        return {
            "visual_evidence": visual_output,
            "stages": stages,
            "errors": errors
        }

    except Exception as exc:
        logger.error(f"Visual Evidence Node failed: {exc}", exc_info=True)
        exec_ms = int((time.time() - t0) * 1000)
        errors.append(f"Visual Evidence error: {str(exc)}")
        stages["Visual Evidence Analysis"] = {
            "status": "failed",
            "execution_time_ms": exec_ms,
            "output": {},
            "error": str(exc)
        }
        return {
            "visual_evidence": {
                "status": "failed",
                "reason": str(exc),
                "detections": [],
                "violations": [],
                "persons_count": 0,
                "annotated_image_path": None
            },
            "stages": stages,
            "errors": errors
        }


# ==========================================
# NODE 3: SAFETY POLICY RETRIEVAL AGENT
# ==========================================
def policy_retrieval_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Queries the RAG Safety Document vector repository using incident cues.
    Retrieves grounded citations and transparently flags policy repository gaps.
    """
    t0 = time.time()
    stages = dict(state.get("stages", {}))
    errors = list(state.get("errors", []))

    category = state.get("incident_category") or "General Construction & PPE"
    scope = state.get("repository_scope") or "All Categories"
    if scope == "All Categories":
        scope_filter = None
    else:
        scope_filter = scope

    try:
        # Build search query targeting the identified hazard and violations
        incident_analysis = state.get("incident_analysis", {})
        cues = incident_analysis.get("severity_cues", [])
        cue_str = " ".join(cues[:2]) if cues else ""

        query = f"What are the mandatory PPE and safety requirements for {category} {cue_str}?"

        rag_result = answer_safety_policy(
            query=query,
            category=scope_filter,
            relevance_threshold=0.20
        )

        citations = rag_result.get("citations", [])
        answer_text = rag_result.get("answer", "")
        mode = rag_result.get("mode", "Extractive")

        # Format citations
        formatted_citations = []
        for cit in citations:
            formatted_citations.append({
                "document_id": cit.get("document_id"),
                "file_name": cit.get("file_name", "Safety_Policy"),
                "page_number": int(cit.get("page_number") or 1),
                "row_number": int(cit.get("row_number") or 0),
                "section_title": cit.get("section_title", "General Policy"),
                "citation": cit.get("citation", f"{cit.get('file_name')} (p.{cit.get('page_number', 1)})"),
                "snippet": cit.get("snippet", ""),
                "score": float(cit.get("score") or 0.0)
            })

        has_grounding = len(formatted_citations) > 0

        exec_ms = int((time.time() - t0) * 1000)
        stages["Safety Policy Retrieval"] = {
            "status": "ok",
            "execution_time_ms": exec_ms,
            "output": {
                "query": query,
                "scope_applied": scope,
                "citations_count": len(formatted_citations),
                "grounded": has_grounding,
                "answering_mode": mode
            },
            "error": ""
        }

        return {
            "policy_context": formatted_citations,
            "policy_answer_summary": answer_text,
            "stages": stages,
            "errors": errors
        }

    except Exception as exc:
        logger.error(f"Policy Retrieval Node failed: {exc}", exc_info=True)
        exec_ms = int((time.time() - t0) * 1000)
        errors.append(f"Policy Retrieval error: {str(exc)}")
        stages["Safety Policy Retrieval"] = {
            "status": "failed",
            "execution_time_ms": exec_ms,
            "output": {"citations_count": 0},
            "error": str(exc)
        }
        return {
            "policy_context": [],
            "policy_answer_summary": "Policy retrieval could not be executed due to an internal error.",
            "stages": stages,
            "errors": errors
        }


# ==========================================
# NODE 4: RISK ASSESSMENT AGENT
# ==========================================
def risk_assessment_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Evaluates incident risk using the matrix in config/risk_framework.json.
    Computes Severity x Likelihood, maps to Risk Levels, and identifies uncertainties.
    """
    t0 = time.time()
    stages = dict(state.get("stages", {}))
    errors = list(state.get("errors", []))

    try:
        framework = load_risk_framework(state.get("custom_risk_framework_path"))
        incident_analysis = state.get("incident_analysis", {})
        visual_evidence = state.get("visual_evidence", {})
        policy_context = state.get("policy_context", [])

        category = incident_analysis.get("identified_category", "General Construction & PPE")
        cat_info = framework.get("hazard_categories", {}).get(category, {})
        baseline_severity = cat_info.get("severity_baseline", 2)

        # 1. Determine Severity (1 to 5)
        severity = baseline_severity
        cues = incident_analysis.get("severity_cues", [])
        if any(c in cues for c in ["fall", "fatal", "unprotected edge", "high voltage", "explosion"]):
            severity = max(severity, 4)
        elif any(c in cues for c in ["fire", "spark", "fumes", "chemical spill"]):
            severity = max(severity, 3)

        # 2. Determine Likelihood (1 to 5)
        # Visual evidence provides objective field confirmation of active unsafe condition
        violations = visual_evidence.get("violations", [])
        visual_status = visual_evidence.get("status")

        if visual_status == "ok" and len(violations) > 0:
            likelihood = 4  # Likely / actively observable non-compliance
            if severity >= 4:
                likelihood = 5  # Imminent catastrophic failure condition
        elif visual_status == "ok" and visual_evidence.get("persons_count", 0) > 0 and len(violations) == 0:
            likelihood = 2  # Unlikely / workers fully equipped
        else:
            # Narrative-only mode or unavailable image
            if len(cues) > 0:
                likelihood = 3  # Possible
            else:
                likelihood = 2  # Low baseline probability

        severity = max(1, min(5, int(severity)))
        likelihood = max(1, min(5, int(likelihood)))
        risk_score = severity * likelihood

        # Map to Risk Level
        thresholds = framework.get("risk_matrix", {}).get("thresholds", {})
        if risk_score <= thresholds.get("Low", {}).get("max", 4):
            risk_level = "Low"
            rec_action = thresholds.get("Low", {}).get("action", "Routine control monitoring.")
        elif risk_score <= thresholds.get("Medium", {}).get("max", 9):
            risk_level = "Medium"
            rec_action = thresholds.get("Medium", {}).get("action", "Supervisory review within 24 hours.")
        elif risk_score <= thresholds.get("High", {}).get("max", 15):
            risk_level = "High"
            rec_action = thresholds.get("High", {}).get("action", "Immediate job pause until controls verified.")
        else:
            risk_level = "Critical"
            rec_action = thresholds.get("Critical", {}).get("action", "IMMEDIATE STOP WORK ORDER.")

        # Regulatory standards cross-referencing
        reg_map = {
            "Working at Heights": ["OSHA 1926.501 (Duty to have fall protection)", "OSHA 1926.502 (Fall protection systems)"],
            "Hot Work & Welding": ["OSHA 1910.252 (General requirements for welding & cutting)", "NFPA 51B"],
            "Electrical Safety": ["OSHA 1910.333 (Selection and use of work practices)", "NFPA 70E Standard for Electrical Safety"],
            "Heavy Machinery & Struck-By": ["OSHA 1926.602 (Material handling equipment)", "OSHA 1926.201 (Signaling and flagging)"],
            "Chemical & Hazardous Materials": ["OSHA 1910.1200 (Hazard Communication Standard)", "OSHA 1910.134 (Respiratory Protection)"],
            "General Construction & PPE": ["OSHA 1926.100 (Head protection)", "OSHA 1926.28 (Personal protective equipment)"]
        }
        regulatory_standards = reg_map.get(category, ["OSHA General Duty Clause Section 5(a)(1)"])

        # Uncertainty guidelines evaluation
        uncertainty_notes = []
        if visual_status != "ok":
            uncertainty_notes.append("Visual evidence was unavailable; risk calculation was derived entirely from the incident narrative.")
        elif visual_status == "ok":
            low_conf_violations = [v for v in violations if float(v.get("confidence", 1.0)) < 0.50]
            if low_conf_violations:
                uncertainty_notes.append(f"{len(low_conf_violations)} visual detection(s) have confidence below 0.50. On-site supervisor verification is advised.")

        if not policy_context:
            uncertainty_notes.append("No directly matching indexed company policy was found; risk was benchmarked against standard statutory guidelines.")

        risk_output = {
            "severity": severity,
            "severity_label": framework.get("severity_scale", {}).get(str(severity), {}).get("label", "Moderate"),
            "likelihood": likelihood,
            "likelihood_label": framework.get("likelihood_scale", {}).get(str(likelihood), {}).get("label", "Possible"),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_action": rec_action,
            "regulatory_references": regulatory_standards,
            "uncertainty_notes": uncertainty_notes
        }

        exec_ms = int((time.time() - t0) * 1000)
        stages["Risk Assessment"] = {
            "status": "ok",
            "execution_time_ms": exec_ms,
            "output": risk_output,
            "error": ""
        }

        return {
            "risk_assessment": risk_output,
            "stages": stages,
            "errors": errors
        }

    except Exception as exc:
        logger.error(f"Risk Assessment Node failed: {exc}", exc_info=True)
        exec_ms = int((time.time() - t0) * 1000)
        errors.append(f"Risk Assessment error: {str(exc)}")
        stages["Risk Assessment"] = {
            "status": "failed",
            "execution_time_ms": exec_ms,
            "output": {"risk_level": "Medium", "risk_score": 6},
            "error": str(exc)
        }
        return {
            "risk_assessment": {
                "severity": 2,
                "severity_label": "Moderate",
                "likelihood": 3,
                "likelihood_label": "Possible",
                "risk_score": 6,
                "risk_level": "Medium",
                "recommended_action": "Standard supervisory review.",
                "regulatory_references": ["OSHA General Duty Clause"],
                "uncertainty_notes": ["Risk assessment completed using default fallback parameters."]
            },
            "stages": stages,
            "errors": errors
        }


# ==========================================
# NODE 5: CORRECTIVE ACTIONS AGENT
# ==========================================
def corrective_actions_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Synthesizes corrective and preventive actions (CAPA) based on the Hierarchy of Controls.
    Strictly distinguishes policy-grounded actions (with document/page citations) from general best practices.
    """
    t0 = time.time()
    stages = dict(state.get("stages", {}))
    errors = list(state.get("errors", []))

    try:
        incident_analysis = state.get("incident_analysis", {})
        risk_assessment = state.get("risk_assessment", {})
        policy_context = state.get("policy_context", [])
        category = incident_analysis.get("identified_category", "General Construction & PPE")
        risk_level = risk_assessment.get("risk_level", "Medium")

        grounded_actions = []
        general_actions = []

        # Ground actions in indexed policy citations
        if policy_context:
            for cit in policy_context[:3]:
                fname = cit.get("file_name", "Policy")
                pnum = cit.get("page_number", 1)
                sect = cit.get("section_title", "General")
                snippet = cit.get("snippet", "").strip()

                short_snippet = snippet[:120] + "..." if len(snippet) > 120 else snippet
                action_text = f"Enforce compliance with mandatory protocol: {short_snippet}"
                grounded_actions.append({
                    "action": action_text,
                    "citation": f"[Source: {fname}, Page {pnum}, Section: {sect}]",
                    "source_type": "Grounded Company Policy",
                    "file_name": fname,
                    "page_number": pnum,
                    "section_title": sect
                })

        # Category-specific Hierarchy of Controls actions
        hierarchy_controls = {
            "Working at Heights": {
                "engineering": "Install certified guardrail barriers and perimeter toe-boards on all elevated work decks >1.8m.",
                "administrative": "Verify valid Working at Heights Permit-to-Work (PTW) and inspect anchorage points before shift commencement.",
                "ppe": "Mandatory donning of EN 361 full-body harness with shock-absorbing lanyard and helmet with chin strap fastened."
            },
            "Hot Work & Welding": {
                "engineering": "Deploy portable fume extraction hoods and flame-retardant welding screens around the hot work zone.",
                "administrative": "Establish dedicated 30-minute post-welding fire watch and clear flammable materials within 10 meters.",
                "ppe": "Provide auto-darkening welding helmets (shade 10-12), split-cowhide leather gloves, and flame-retardant jacket."
            },
            "Electrical Safety": {
                "engineering": "Install lockable isolation switches and residual current devices (RCD) on temporary supply feeds.",
                "administrative": "Execute formal Lockout / Tagout (LOTO) procedure with individual padlock and test-before-touch verification.",
                "ppe": "Equip personnel with Class 0 dielectric gloves, 8 cal/cm² arc-flash face shield, and non-conductive safety footwear."
            },
            "Heavy Machinery & Struck-By": {
                "engineering": "Erect physical pedestrian-vehicle segregation barriers and illuminated designated walkways.",
                "administrative": "Implement mandatory blind-spot horn sounding protocol and assign dedicated qualified spotters/banksmen.",
                "ppe": "Enforce Class 3 high-visibility vests with reflective striping and steel-toe composite penetration-resistant boots."
            },
            "Chemical & Hazardous Materials": {
                "engineering": "Operate continuous forced-air ventilation and keep certified eyewash/safety shower stations within 10 seconds reach.",
                "administrative": "Mandate Safety Data Sheet (SDS) briefing prior to decanting and maintain chemical spill containment kits.",
                "ppe": "Issue fitted ABEK1 chemical respirators, splash goggles, and nitrile gauntlets."
            },
            "General Construction & PPE": {
                "engineering": "Maintain clear, unobstructed designated walkways and install non-slip walking surfaces.",
                "administrative": "Conduct daily pre-shift 5-minute safety toolbox talks reinforcing zero-tolerance PPE compliance.",
                "ppe": "Enforce 100% hard hat and high-visibility vest compliance across all active site perimeters."
            }
        }

        controls = hierarchy_controls.get(category, hierarchy_controls["General Construction & PPE"])

        general_actions.append({
            "control_type": "Engineering Control",
            "action": controls["engineering"],
            "source_type": "General Industry Best Practice (Not from indexed documents)"
        })
        general_actions.append({
            "control_type": "Administrative Control",
            "action": controls["administrative"],
            "source_type": "General Industry Best Practice (Not from indexed documents)"
        })
        general_actions.append({
            "control_type": "Personal Protective Equipment",
            "action": controls["ppe"],
            "source_type": "General Industry Best Practice (Not from indexed documents)"
        })

        # Timeline categorization
        immediate_action = "STOP WORK ORDER: Suspend hazardous task immediately until supervisor inspection" if risk_level in ("Critical", "High") else "Issue verbal correction and immediate PPE rectification."
        corrective_output = {
            "immediate_action": immediate_action,
            "grounded_actions": grounded_actions,
            "general_actions": general_actions,
            "hierarchy_summary": {
                "engineering": controls["engineering"],
                "administrative": controls["administrative"],
                "ppe": controls["ppe"]
            }
        }

        exec_ms = int((time.time() - t0) * 1000)
        stages["Corrective Actions"] = {
            "status": "ok",
            "execution_time_ms": exec_ms,
            "output": corrective_output,
            "error": ""
        }

        return {
            "corrective_actions": corrective_output,
            "stages": stages,
            "errors": errors
        }

    except Exception as exc:
        logger.error(f"Corrective Actions Node failed: {exc}", exc_info=True)
        exec_ms = int((time.time() - t0) * 1000)
        errors.append(f"Corrective Actions error: {str(exc)}")
        stages["Corrective Actions"] = {
            "status": "failed",
            "execution_time_ms": exec_ms,
            "output": {},
            "error": str(exc)
        }
        return {
            "corrective_actions": {
                "immediate_action": "Supervisor intervention required.",
                "grounded_actions": [],
                "general_actions": [
                    {
                        "control_type": "General Best Practice",
                        "action": "Conduct immediate safety briefing and inspect PPE compliance.",
                        "source_type": "General Industry Best Practice (Not from indexed documents)"
                    }
                ]
            },
            "stages": stages,
            "errors": errors
        }


# ==========================================
# NODE 6: REPORT GENERATION AGENT
# ==========================================
def report_generation_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Assembles the formal comprehensive markdown investigation report.
    Persists investigation, stages, and citations to the SQLite database.
    """
    t0 = time.time()
    stages = dict(state.get("stages", {}))
    errors = list(state.get("errors", []))

    incident_id = state.get("incident_id") or f"INV-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    title = state.get("title") or "Workplace Safety Incident Investigation"
    description = state.get("incident_description") or "No description provided."
    category = state.get("incident_category") or "General Construction & PPE"
    image_path = state.get("visual_evidence_path") or ""
    scope = state.get("repository_scope") or "All Categories"
    user_id = state.get("user_id")

    incident_analysis = state.get("incident_analysis", {})
    visual_evidence = state.get("visual_evidence", {})
    policy_context = state.get("policy_context", [])
    policy_summary = state.get("policy_answer_summary") or ""
    risk_assessment = state.get("risk_assessment", {})
    corrective_actions = state.get("corrective_actions", {})

    risk_level = risk_assessment.get("risk_level", "Medium")
    risk_score = risk_assessment.get("risk_score", 6)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Mark stage 6 ok before writing report to include in workflow summary table
    exec_ms_pre = int((time.time() - t0) * 1000)
    stages["Report Generation"] = {
        "status": "ok",
        "execution_time_ms": exec_ms_pre,
        "output": {"incident_id": incident_id},
        "error": ""
    }

    # Build Markdown Report
    lines = [
        f"# DETECTO AI / SAFETYVISION AI - FORMAL SAFETY INVESTIGATION REPORT",
        f"**Incident Reference:** `{incident_id}`  ",
        f"**Investigation Title:** {title}  ",
        f"**Generated:** {timestamp_str} | **Assessor:** Autonomous Multimodal AI Agent  ",
        f"**Repository Scope:** {scope} | **Primary Hazard Category:** {category}  ",
        f"**Assessed Enterprise Risk Rating:** **{risk_level.upper()}** (Score: {risk_score}/25)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        f"An autonomous multimodal safety investigation was executed for incident `{incident_id}`. "
        f"The reported operational context involved **{category}**. "
        f"Evaluation of narrative evidence and computer vision telemetry yielded an assessed risk level of **{risk_level}** "
        f"(Severity: {risk_assessment.get('severity_label', 'Moderate')}, Likelihood: {risk_assessment.get('likelihood_label', 'Possible')}). "
        f"{corrective_actions.get('immediate_action', 'Supervisory oversight required.')}",
        "",
        "## 2. Incident Description & Operating Narrative",
        f"> {description}",
        "",
        f"- **Identified Equipment / Entities:** {', '.join(incident_analysis.get('extracted_entities', [])) or 'None explicitly identified'}",
        f"- **Observed Severity Cues:** {', '.join(incident_analysis.get('severity_cues', [])) or 'Standard operational activity'}",
        f"- **Mandatory PPE Expected for Category:** {', '.join(incident_analysis.get('mandatory_ppe', []))}",
        "",
        "## 3. Visual Intelligence & Telemetry Evidence",
    ]

    vis_status = visual_evidence.get("status", "unavailable")
    if vis_status == "ok":
        violations = visual_evidence.get("violations", [])
        lines.append(f"- **Visual Telemetry Status:** Successfully processed with YOLOv8.")
        lines.append(f"- **Total Objects Detected:** {visual_evidence.get('total_detections', 0)}")
        lines.append(f"- **Personnel Count:** {visual_evidence.get('persons_count', 0)}")
        lines.append(f"- **Active Violations Identified:** {len(violations)}")
        if visual_evidence.get("annotated_image_path"):
            lines.append(f"- **Annotated Evidence Crop:** `{visual_evidence.get('annotated_image_path')}`")

        if violations:
            lines.append("")
            lines.append("| Detected Class | Confidence | Severity | Coordinates (x1, y1, x2, y2) |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for v in violations:
                box_str = f"({v.get('x1', 0)}, {v.get('y1', 0)}, {v.get('x2', 0)}, {v.get('y2', 0)})"
                lines.append(f"| `{v.get('class')}` | {v.get('confidence', 0.0):.2f} | {v.get('severity', 'Medium')} | {box_str} |")
    elif vis_status == "unavailable":
        lines.append(f"> **Visual Telemetry Note:** Visual evidence is marked **UNAVAILABLE** ({visual_evidence.get('reason', 'None provided')}). "
                     f"In accordance with system safeguards, the investigation proceeded using narrative and document evidence.")
    else:
        lines.append(f"> **Visual Telemetry Alert:** Visual processing encountered an error: `{visual_evidence.get('reason', 'Processing failure')}`.")

    lines.extend([
        "",
        "## 4. Grounded Safety Policy & Regulatory Context",
    ])

    if policy_context:
        lines.append("The following corporate safety clauses were retrieved from the indexed knowledge base:")
        lines.append("")
        lines.append("| Document Name | Page | Section | Excerpt Snippet | Relevance Score |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for cit in policy_context:
            snip = cit.get("snippet", "").replace("\n", " ")[:90] + "..."
            lines.append(f"| `{cit.get('file_name')}` | p.{cit.get('page_number', 1)} | {cit.get('section_title', 'General')} | {snip} | {cit.get('score', 0.0):.2f} |")
        lines.append("")
        if policy_summary:
            lines.append(f"**Knowledge Base Synthesis:** {policy_summary}")
    else:
        lines.append("> **Policy Grounding Disclosure:** No matching company safety documents were found in the indexed repository for this incident context. "
                     "Safety requirements were cross-referenced against statutory safety regulations.")

    lines.extend([
        "",
        "## 5. Enterprise Risk Matrix Evaluation",
        "",
        "| Evaluation Metric | Assessed Value | Technical Basis |",
        "| :--- | :--- | :--- |",
        f"| **Severity Rating** | **{risk_assessment.get('severity', 2)}/5** ({risk_assessment.get('severity_label', 'Moderate')}) | Evaluated against baseline hazard severity and potential bodily harm |",
        f"| **Likelihood Rating** | **{risk_assessment.get('likelihood', 3)}/5** ({risk_assessment.get('likelihood_label', 'Possible')}) | Derived from presence of active violations and field narrative |",
        f"| **Composite Risk Score** | **{risk_score}/25** | Calculated as Severity × Likelihood |",
        f"| **Risk Categorization** | **{risk_level.upper()}** | Defined by Enterprise Risk Framework threshold matrix |",
        "",
        f"**Governing Regulatory Standards:** {', '.join(risk_assessment.get('regulatory_references', []))}",
        "",
        "## 6. Evidentiary Uncertainty & Boundary Disclosures",
    ])

    uncertainties = risk_assessment.get("uncertainty_notes", [])
    if uncertainties:
        for unc in uncertainties:
            lines.append(f"- [!] {unc}")
    else:
        lines.append("- Visual and documentary evidence achieved high confidence. No significant evidentiary gaps detected.")

    lines.extend([
        "",
        "## 7. Corrective & Preventive Action Plan (CAPA)",
        f"**Immediate Containment:** {corrective_actions.get('immediate_action', 'Initiate supervisory review.')}",
        "",
        "### A. Grounded Actions (Directly Derived from Indexed Documents)",
    ])

    grounded_acts = corrective_actions.get("grounded_actions", [])
    if grounded_acts:
        for ga in grounded_acts:
            lines.append(f"- **{ga.get('action')}**  \n  *{ga.get('citation')}*")
    else:
        lines.append("- *(No direct company policy citations available for auto-grounded actions. Rely on statutory controls below.)*")

    lines.extend([
        "",
        "### B. Hierarchy of Controls (Standard EHS Engineering Practice)",
    ])
    general_acts = corrective_actions.get("general_actions", [])
    for ga in general_acts:
        lines.append(f"- **[{ga.get('control_type')}]** {ga.get('action')}  \n  *({ga.get('source_type')})*")

    lines.extend([
        "",
        "## 8. Multi-Agent Workflow Execution Log",
        "",
        "| Stage Name | Agent Execution Status | Latency (ms) | Output Summary |",
        "| :--- | :--- | :--- | :--- |"
    ])

    for stage_name, sinfo in stages.items():
        st_status = sinfo.get("status", "ok").upper()
        ms = sinfo.get("execution_time_ms", 0)
        lines.append(f"| {stage_name} | `{st_status}` | {ms} ms | Completed successfully |")

    lines.extend([
        "",
        "---",
        "*SafetyVision AI Autonomous Multi-Agent Investigation System. Generated automatically for EHS audit compliance.*"
    ])

    full_report_md = "\n".join(lines)

    # Persist record in SQLite Database
    try:
        inv_id = create_investigation(
            incident_id=incident_id,
            title=title,
            incident_description=description,
            incident_category=category,
            visual_evidence_path=image_path,
            repository_scope=scope,
            risk_level=risk_level,
            status="Completed",
            final_report=full_report_md,
            user_id=user_id
        )

        # Persist each stage
        for stage_name, sdata in stages.items():
            add_investigation_stage(
                investigation_id=inv_id,
                stage_name=stage_name,
                status=sdata.get("status", "ok"),
                output_json=sdata.get("output", {}),
                execution_time_ms=sdata.get("execution_time_ms", 0),
                error_message=sdata.get("error", "")
            )

        # Persist citations
        for cit in policy_context:
            add_investigation_citation(
                investigation_id=inv_id,
                file_name=cit.get("file_name", "Policy"),
                page_number=cit.get("page_number", 1),
                row_number=cit.get("row_number", 0),
                section_title=cit.get("section_title", ""),
                citation=cit.get("citation", ""),
                snippet=cit.get("snippet", ""),
                document_id=cit.get("document_id")
            )

        # Audit Log
        log_audit_action(
            action="INVESTIGATION_COMPLETED",
            user_id=user_id,
            target_type="investigation",
            target_id=inv_id,
            details=json.dumps({
                "incident_id": incident_id,
                "risk_level": risk_level,
                "hazard_category": category,
                "stages_count": len(stages)
            })
        )

    except Exception as db_exc:
        logger.error(f"Failed to persist investigation to database: {db_exc}", exc_info=True)
        errors.append(f"Database persistence error: {str(db_exc)}")
        inv_id = None

    exec_ms = int((time.time() - t0) * 1000)
    stages["Report Generation"] = {
        "status": "ok",
        "execution_time_ms": exec_ms,
        "output": {"incident_id": incident_id, "investigation_id": inv_id},
        "error": ""
    }

    return {
        "report": full_report_md,
        "investigation_id": inv_id,
        "incident_id": incident_id,
        "stages": stages,
        "errors": errors
    }


# ==========================================
# GRAPH BUILDER & ENTRYPOINT
# ==========================================
def build_investigation_graph():
    """Constructs and compiles the 6-Node LangGraph StateGraph."""
    graph = StateGraph(InvestigationState)

    graph.add_node("incident_analysis", incident_analysis_node)
    graph.add_node("visual_evidence", visual_evidence_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("corrective_actions", corrective_actions_node)
    graph.add_node("report_generation", report_generation_node)

    graph.add_edge(START, "incident_analysis")
    graph.add_edge("incident_analysis", "visual_evidence")
    graph.add_edge("visual_evidence", "policy_retrieval")
    graph.add_edge("policy_retrieval", "risk_assessment")
    graph.add_edge("risk_assessment", "corrective_actions")
    graph.add_edge("corrective_actions", "report_generation")
    graph.add_edge("report_generation", END)

    return graph.compile()


_INVESTIGATION_APP = None


def get_investigation_workflow():
    """Returns singleton compiled LangGraph application."""
    global _INVESTIGATION_APP
    if _INVESTIGATION_APP is None:
        _INVESTIGATION_APP = build_investigation_graph()
    return _INVESTIGATION_APP


def run_safety_investigation(
    incident_description: str,
    title: Optional[str] = None,
    visual_evidence_path: Optional[str] = None,
    repository_scope: str = "All Categories",
    incident_category: Optional[str] = None,
    user_id: Optional[int] = None,
    incident_id: Optional[str] = None,
    image_path: Optional[str] = None,
    scope_filter: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main invocation entrypoint for the 6-Node LangGraph Safety Investigation Workflow.
    Executes all 6 nodes sequentially with full state management, persistence, and audit logging.
    Supports image_path and scope_filter keyword aliases for flexible invocation.
    """
    if visual_evidence_path is None and image_path is not None:
        visual_evidence_path = image_path
    if scope_filter is not None:
        repository_scope = scope_filter or "All Categories"

    app = get_investigation_workflow()

    gen_incident_id = incident_id or f"INV-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    gen_title = title or f"Incident Investigation - {gen_incident_id}"

    initial_state: InvestigationState = {
        "incident_id": gen_incident_id,
        "title": gen_title,
        "incident_description": incident_description,
        "incident_category": incident_category,
        "visual_evidence_path": visual_evidence_path,
        "repository_scope": repository_scope,
        "user_id": user_id,
        "stages": {},
        "errors": []
    }

    final_state = app.invoke(initial_state)
    return final_state

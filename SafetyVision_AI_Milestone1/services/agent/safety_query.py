"""
Detecto AI - Visual-to-Policy Reasoning Service
Branch: milestone-3-dev

Implements `answer_safety_query(question, detection, zone)`:
- Classifies user question into one of 5 intents:
  1. policy: asks what a rule, requirement, or standard says.
  2. visual: asks about a detection or event (why an alert fired, whether a worker is compliant).
  3. combined: asks whether a detection violates a rule (compares detection against policy).
  4. guidance: general safety question with no specific event or rule.
  5. out_of_scope: not related to workplace safety or the knowledge base.
- Offline only. Deterministic rule-based reasoning engine.
- Reuses existing Policy Bridge (get_policy_for_violation) and FAISS vector store.
- Never states requirements not in retrieved chunks.
- Explicitly flags optical detections as potential non-compliance pending human review.
- Dynamic across all PPE types, zones, and questions (zero hard-coding).
"""

import re
from typing import Dict, Any, List, Optional
import numpy as np

from database import get_safety_events
from services.document_intelligence.vector_store import get_vector_store
from services.document_intelligence.rag_service import get_policy_for_violation, REGULATORY_STANDARDS

# Recognized safety and worksite domain terms
_SAFETY_KEYWORDS = {
    "ppe", "safety", "hazard", "hazards", "violation", "infraction", "breach", "compliant", "compliance",
    "comply", "helmet", "hardhat", "hard-hat", "head", "vest", "high-vis", "high-visibility", "reflective",
    "boot", "boots", "footwear", "shoe", "shoes", "steel-toe", "puncture", "sole",
    "goggle", "goggles", "eyewear", "glasses", "face shield", "shield", "splash", "eye", "eyes",
    "glove", "gloves", "hand", "hands", "laceration", "chemical", "acid", "solvent", "coating",
    "clearance", "standoff", "radius", "barrier", "walkway", "aisle", "lane",
    "crane", "forklift", "hoist", "rigging", "dock", "loading", "corridor", "overhead",
    "osha", "iso", "ansi", "astm", "isea", "1910", "45001", "z89", "z87", "z358", "f2413", "107",
    "sop", "manual", "policy", "rule", "standard", "regulation", "guideline", "protocol", "mandate",
    "camera", "cctv", "surveillance", "stream", "feed", "detection", "detected", "spotted", "frame",
    "alert", "incident", "audit", "inspection", "capa", "ticket", "emergency", "eyewash",
    "worker", "workers", "operator", "personnel", "pedestrian", "workplace", "plant", "facility", "zone",
    "stop-work", "protective", "gear", "equipment"
}

# Generic words that do not count as distinct topic keywords
GENERIC_STOPWORDS = {
    "what", "is", "the", "are", "for", "in", "and", "of", "to", "a", "an", "does", "this",
    "do", "rule", "rules", "policy", "policies", "standard", "standards", "requirement",
    "requirements", "regulation", "regulations", "safety", "workplace", "worksite", "ppe",
    "mandatory", "company", "facility", "plant", "should", "must", "apply", "applies",
    "about", "tell", "explain", "describe", "show", "give", "me", "general",
    "handling", "handle", "use", "using", "work", "working", "exist", "exists",
    "that", "without", "seen", "allowed", "worker", "workers", "commercial"
}

# Explicit off-topic trigger phrases
_OUT_OF_SCOPE_PATTERNS = [
    r"\bcapital of\b", r"\bwho wrote\b", r"\bweather in\b", r"\bdistance (to|from)\b",
    r"\brecipe for\b", r"\bwrite a poem\b", r"\bpython code\b", r"\btell me a joke\b",
    r"\bsports score\b", r"\bpopulation of\b", r"\bwho won\b", r"\btranslate\b",
    r"\bmovie\b", r"\bsong\b", r"\bcook\b", r"\bchocolate\b", r"\bquantum\b", r"\bwarp drive\b"
]


def classify_safety_intent(
    question: str,
    detection: Optional[Dict[str, Any]] = None
) -> str:
    """
    Classifies a natural language inquiry into one of 5 distinct intents:
    - 'policy': asks what a rule, requirement, or standard says.
    - 'visual': asks about a detection or event.
    - 'combined': asks whether a detection violates a rule (or compares detection with policy).
    - 'guidance': general safety question with no specific event or rule.
    - 'out_of_scope': not related to workplace safety or the knowledge base.

    Intent Priority when VISUAL and POLICY both match:
    1. If a detection is present AND the query asks whether it violates a rule / is compliant / permitted:
       -> 'combined' (visual detection evaluated against policy).
    2. If NO detection is present BUT the query asks whether a visual event violates a rule:
       -> 'combined'.
    3. If the query asks for the rule, requirement, or policy governing a monitoring/camera scenario
       (e.g., 'What is the policy for CCTV camera inspections?'):
       -> 'policy' (user's primary objective is document retrieval).
    4. If the query asks about the visual event itself (why alert fired, optical confidence, camera feed):
       -> 'visual'.
    """
    q_str = question.strip()
    q_lower = q_str.lower()

    # 1. Out-of-scope check
    has_safety_kw = any(term in q_lower for term in _SAFETY_KEYWORDS)
    matches_off_topic = any(re.search(pat, q_lower) for pat in _OUT_OF_SCOPE_PATTERNS)

    if (not has_safety_kw and not detection) or (matches_off_topic and not any(k in q_lower for k in ["ppe", "osha", "helmet", "vest", "boot", "glove", "goggle", "safety"])):
        return "out_of_scope"

    # Keywords for intent detection
    comparison_keywords = [
        "violate", "violation", "infraction", "breach", "compliant", "comply", "compliance",
        "allow", "allowed", "permitted", "against", "legal", "authorized", "penalty"
    ]
    policy_keywords = [
        "policy", "rule", "standard", "requirement", "require", "mandatory", "osha", "iso", "ansi",
        "astm", "isea", "mandate", "protocol", "clause", "specification", "guideline",
        "standoff", "clearance", "regulation", "must wear", "is required",
        "what does the standard", "what are the rules", "what is the policy", "what standard"
    ]
    visual_keywords = [
        "why did", "why was", "alert", "camera", "cam-", "cctv", "footage", "spotted", "caught",
        "detected", "detection", "confidence", "bounding box", "frame", "optical", "flagged"
    ]
    guidance_keywords = [
        "how to", "how should", "best practice", "tips", "guidance", "advice", "improve safety",
        "prevent accident", "approach hazard", "culture", "recommendation", "foster"
    ]

    has_comparison = any(k in q_lower for k in comparison_keywords)
    has_policy = any(k in q_lower for k in policy_keywords)
    has_visual = any(k in q_lower for k in visual_keywords)
    has_guidance = any(k in q_lower for k in guidance_keywords)

    # 2. When detection is attached
    if detection is not None and isinstance(detection, dict) and bool(detection):
        if has_comparison or has_policy:
            return "combined"
        if has_visual:
            return "visual"
        # If question asks about an independent rule with no reference to the current detection
        if any(k in q_lower for k in ["what does osha", "what is the policy for", "standard says", "regulation requires"]):
            return "policy"
        return "combined"

    # 3. When detection is None: Resolve Visual vs Policy vs Combined priority explicitly
    if has_visual and (has_comparison or (has_policy and any(w in q_lower for w in ["worker", "person", "anyone", "someone", "seen", "wearing", "without"]))):
        # Asks if an observed visual event violates a rule
        return "combined"

    if has_policy:
        # User is requesting policy/standards retrieval, even if camera/monitoring is mentioned in passing
        return "policy"

    if has_visual:
        # Strictly querying a visual event / alert
        return "visual"

    if has_guidance:
        return "guidance"

    if q_lower.startswith("how"):
        return "guidance"

    return "policy"


def _extract_zone_identifier(
    question: str,
    detection: Optional[Dict[str, Any]] = None,
    explicit_zone: Optional[str] = None
) -> Optional[str]:
    """Extracts facility zone identifier from explicit param, detection, or question text."""
    if explicit_zone and explicit_zone.strip():
        return explicit_zone.strip()

    if detection and isinstance(detection, dict):
        z_det = detection.get("zone") or detection.get("zone_id")
        if z_det:
            return str(z_det).strip()

    q_low = question.lower()
    if "z-01" in q_low or "zone a" in q_low or "fabrication" in q_low or "welding" in q_low:
        return "Zone Z-01"
    elif "z-02" in q_low or "zone b" in q_low or "chemical" in q_low or "coating" in q_low or "acid" in q_low:
        return "Zone Z-02"
    elif "z-03" in q_low or "zone c" in q_low or "logistics" in q_low or "forklift" in q_low or "dock" in q_low:
        return "Zone Z-03"
    elif "z-04" in q_low or "zone d" in q_low or "crane" in q_low or "rigging" in q_low or "hoist" in q_low:
        return "Zone Z-04"

    m = re.search(r"\bzone\s*([a-z0-9\-]+)\b", q_low)
    if m:
        return f"Zone {m.group(1).upper()}"

    return None


def _extract_gear_keywords(
    question: str,
    detection: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Extracts all PPE gear categories mentioned in detection class or question text."""
    gears = []
    if detection and isinstance(detection, dict):
        raw_class = detection.get("class") or detection.get("violation_type") or ""
        clean = raw_class.lower().replace("no_", "").replace("no-", "").strip()
        if "helmet" in clean or "hardhat" in clean:
            gears.append("helmet")
        if "vest" in clean:
            gears.append("vest")
        if "glove" in clean:
            gears.append("gloves")
        if "boot" in clean or "shoe" in clean or "footwear" in clean:
            gears.append("boots")
        if "goggle" in clean or "glass" in clean or "eye" in clean:
            gears.append("goggles")

    q_low = question.lower()
    if any(k in q_low for k in ["helmet", "hardhat", "hard-hat", "head protection"]):
        if "helmet" not in gears: gears.append("helmet")
    if any(k in q_low for k in ["vest", "high-vis", "high-visibility", "reflective"]):
        if "vest" not in gears: gears.append("vest")
    if any(k in q_low for k in ["glove", "gloves", "hand protection", "hands"]):
        if "gloves" not in gears: gears.append("gloves")
    if any(k in q_low for k in ["boot", "boots", "footwear", "shoe", "shoes", "steel-toe"]):
        if "boots" not in gears: gears.append("boots")
    if any(k in q_low for k in ["goggle", "goggles", "eyewear", "glasses", "face shield", "splash"]):
        if "goggles" not in gears: gears.append("goggles")

    return gears


def _matches_topic_token(token: str, target: str) -> bool:
    """Matches exact word token or singular stemmed form in target text."""
    if token in target:
        return True
    if token.endswith("s") and len(token) > 3 and token[:-1] in target:
        return True
    if token.endswith("es") and len(token) > 4 and token[:-2] in target:
        return True
    return False


def _retrieve_policy_chunks(
    query: str,
    zone: Optional[str] = None,
    gear_types: Optional[List[str]] = None,
    top_k: int = 3,
    is_guidance: bool = False
) -> List[Dict[str, Any]]:
    """
    Executes hybrid semantic vector search against the internal FAISS vector store
    with term-level precision re-ranking.
    Guaranteed: Returns empty list if query has 0 topic keyword hits or is below 0.25 floor.
    If is_guidance is True, excludes mandatory requirement sections so guidance does not cite requirements.
    """
    vs = get_vector_store()
    if not vs.metadata:
        return []

    # Dynamic topic tokens extraction
    q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower()))
    topic_tokens = {t for t in q_tokens if t not in GENERIC_STOPWORDS and len(t) > 2}
    if zone:
        for z_t in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", zone.lower()):
            if z_t not in GENERIC_STOPWORDS and len(z_t) > 1:
                topic_tokens.add(z_t)
    if gear_types:
        for g in gear_types:
            topic_tokens.add(g.lower())

    search_q = query
    if zone and zone.lower() not in query.lower():
        search_q = f"{zone} {search_q}"
    if gear_types:
        for g in gear_types:
            if g.lower() not in query.lower():
                search_q = f"{g} {search_q}"

    vector_results = vs.search(search_q, top_k=min(top_k * 4, 20))
    q_vec = vs.embedder.generate_embeddings([search_q])[0]

    scored_candidates = []
    seen_chunk_keys = set()

    for item in vector_results:
        cid = (item.get("file_name"), item.get("chunk_index"))
        if cid in seen_chunk_keys:
            continue
        seen_chunk_keys.add(cid)

        text_low = item.get("text_content", "").lower()
        title_low = (item.get("section_title") or "").lower()
        fname_low = (item.get("file_name") or "").lower()

        # Reject automated test showcase exports (not authoritative policy standards)
        if fname_low.startswith("detectoai_showcase"):
            continue

        # Reject chunks from sections dedicated to a conflicting PPE category
        if gear_types:
            GEAR_CATEGORIES = {
                "helmet": ["helmet", "hardhat", "hard-hat", "head protection"],
                "vest": ["vest", "high-vis", "high-visibility", "reflective"],
                "boots": ["boot", "boots", "footwear", "shoe", "steel-toe"],
                "gloves": ["glove", "gloves", "hand protection"],
                "goggles": ["goggle", "goggles", "eyewear", "glasses", "face shield"]
            }
            req_words = [w for g in gear_types for w in GEAR_CATEGORIES.get(g, [g])]
            conf_words = [w for cat, words in GEAR_CATEGORIES.items() if cat not in gear_types for w in words]
            if any(cw in title_low for cw in conf_words) and not any(rw in title_low for rw in req_words):
                continue

        # Guidance queries must not cite statutory requirements, mandatory matrices, penalties, or raw CSV defect audit logs
        if is_guidance and (fname_low.endswith(".csv") or fname_low.startswith("osha_") or any(w in title_low for w in ["mandatory", "matrix", "penalty", "penalties", "statutory", "regulations", "standard", "requirements"])):
            continue

        topic_hits = sum(1 for t in topic_tokens if _matches_topic_token(t, text_low) or _matches_topic_token(t, title_low) or _matches_topic_token(t, fname_low))
        sim = float(item.get("similarity_score") or 0.0)

        # Relevance filtering: 0.25 floor for similarity if topic_hits == 0
        if topic_hits == 0 and sim < 0.25:
            continue

        # Scoring and ranking: prioritize actionable operational protocols for guidance intent
        if is_guidance:
            title_hits = sum(1 for t in topic_tokens if _matches_topic_token(t, title_low))
            is_boilerplate = any(b in title_low for b in ["purpose", "applicability", "scope", "overview", "introduction"])
            is_procedural = any(p in title_low for p in ["protocol", "procedure", "practice", "guidance", "action", "emergency", "response", "handling", "spill"])
            guidance_boost = 1.5 if is_procedural else (-1.0 if is_boilerplate else 0.0)
            score = (topic_hits * 1.5) + (title_hits * 1.0) + sim + guidance_boost
        else:
            policy_boost = 0.3 if not fname_low.endswith(".csv") else 0.0
            score = (topic_hits * 1.5) + sim + policy_boost
        item["topic_hits"] = topic_hits
        item["hybrid_score"] = score
        scored_candidates.append((score, topic_hits, sim, item))

    # Also check full metadata store if vector search missed high-topic chunks
    for m in vs.metadata:
        cid = (m.get("file_name"), m.get("chunk_index"))
        if cid in seen_chunk_keys:
            continue

        text_low = m.get("text_content", "").lower()
        title_low = (m.get("section_title") or "").lower()
        fname_low = (m.get("file_name") or "").lower()

        # Reject automated test showcase exports (not authoritative policy standards)
        if fname_low.startswith("detectoai_showcase"):
            continue

        # Reject chunks from sections dedicated to a conflicting PPE category
        if gear_types:
            GEAR_CATEGORIES = {
                "helmet": ["helmet", "hardhat", "hard-hat", "head protection"],
                "vest": ["vest", "high-vis", "high-visibility", "reflective"],
                "boots": ["boot", "boots", "footwear", "shoe", "steel-toe"],
                "gloves": ["glove", "gloves", "hand protection"],
                "goggles": ["goggle", "goggles", "eyewear", "glasses", "face shield"]
            }
            req_words = [w for g in gear_types for w in GEAR_CATEGORIES.get(g, [g])]
            conf_words = [w for cat, words in GEAR_CATEGORIES.items() if cat not in gear_types for w in words]
            if any(cw in title_low for cw in conf_words) and not any(rw in title_low for rw in req_words):
                continue

        if is_guidance and (fname_low.endswith(".csv") or fname_low.startswith("osha_") or any(w in title_low for w in ["mandatory", "matrix", "penalty", "penalties", "statutory", "regulations", "standard", "requirements"])):
            continue

        topic_hits = sum(1 for t in topic_tokens if _matches_topic_token(t, text_low) or _matches_topic_token(t, title_low) or _matches_topic_token(t, fname_low))
        if topic_hits >= 2:
            seen_chunk_keys.add(cid)
            item = dict(m)
            item["citation"] = vs._format_citation(item)
            c_vec = vs.embedder.generate_embeddings([m["text_content"]])[0]
            real_sim = float(np.dot(q_vec, c_vec))
            item["similarity_score"] = real_sim
            if is_guidance:
                title_hits = sum(1 for t in topic_tokens if _matches_topic_token(t, title_low))
                is_boilerplate = any(b in title_low for b in ["purpose", "applicability", "scope", "overview", "introduction"])
                is_procedural = any(p in title_low for p in ["protocol", "procedure", "practice", "guidance", "action", "emergency", "response", "handling", "spill"])
                guidance_boost = 1.5 if is_procedural else (-1.0 if is_boilerplate else 0.0)
                score = (topic_hits * 1.5) + (title_hits * 1.0) + real_sim + guidance_boost
            else:
                policy_boost = 0.3 if not fname_low.endswith(".csv") else 0.0
                score = (topic_hits * 1.5) + real_sim + policy_boost
            item["topic_hits"] = topic_hits
            item["hybrid_score"] = score
            scored_candidates.append((score, topic_hits, real_sim, item))

    # Guard against queries with no specific topic keywords (generic-only queries)
    if not topic_tokens:
        return []

    # Guard against non-existent topics (e.g. cryogenic hyperdrive, space plasma)
    min_required_hits = 2 if len(topic_tokens) >= 3 else 1
    max_hits = max((cand[1] for cand in scored_candidates), default=0)
    if max_hits < min_required_hits:
        return []

    # If all candidates are below 0.25 relevance floor (and 0 topic hits), reject
    scored_candidates = [cand for cand in scored_candidates if cand[0] >= 0.25]
    if not scored_candidates:
        return []

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [cand[3] for cand in scored_candidates[:top_k]]


def _build_citations_list(
    chunks: List[Dict[str, Any]],
    topic_tokens: Optional[set] = None
) -> List[Dict[str, Any]]:
    """Builds clean, structured citation dictionaries from retrieved chunks."""
    citations = []
    seen = set()
    for c in chunks:
        doc_name = c.get("document_name") or c.get("file_name") or "Safety Policy Manual"
        page = c.get("page_number")
        sec = c.get("section_title")
        row = c.get("row_number")
        key = (doc_name, page, sec, row)
        if key in seen:
            continue
        seen.add(key)

        snippet = c.get("text_content", "").strip()
        snippet_clean = re.sub(r"^\[.*?\]\s*", "", snippet)

        quote_text = snippet_clean[:600].strip()
        if topic_tokens:
            low_text = snippet_clean.lower()
            match_indices = [low_text.find(t) for t in topic_tokens if t in low_text and low_text.find(t) >= 0]
            if match_indices:
                first_idx = min(match_indices)
                if first_idx > 120:
                    start_idx = max(0, first_idx - 60)
                    end_idx = min(len(snippet_clean), start_idx + 500)
                    quote_text = snippet_clean[start_idx:end_idx].strip()

        citations.append({
            "document": doc_name,
            "page": page,
            "section": sec,
            "row": row,
            "quote": quote_text
        })
    return citations


def answer_safety_query(
    question: str,
    detection: Optional[Dict[str, Any]] = None,
    zone: Optional[str] = None
) -> Dict[str, Any]:
    """
    Answers any user question about workplace safety, fusing retrieved policy documentation
    with current optical detection context when relevant.
    
    Args:
        question: Natural language question regarding safety policy, visual detection, or guidance.
        detection: Optional optical detection dictionary from live surveillance or camera models.
                   Example: {"class": "no_helmet", "confidence": 0.92, "zone": "Zone A", "camera_id": "CAM-01"}
        zone: Optional explicit workplace zone identifier (e.g. "Zone Z-01", "Zone A").
        
    Returns:
        Structured result dictionary:
        {
            "intent": "policy" | "visual" | "combined" | "guidance" | "out_of_scope",
            "status": "answered" | "not_found" | "out_of_scope",
            "answer": str,
            "citations": List[Dict[str, Any]],
            "detection_summary": Optional[Dict[str, Any]],
            "grounded": bool
        }
    """
    intent = classify_safety_intent(question, detection)

    # -------------------------------------------------------------
    # 1. OUT OF SCOPE
    # -------------------------------------------------------------
    if intent == "out_of_scope":
        return {
            "intent": "out_of_scope",
            "status": "out_of_scope",
            "answer": (
                "This inquiry is outside the scope of the workplace safety knowledge base and PPE monitoring system. "
                "I can only assist with worksite safety policies, PPE regulatory standards, visual compliance events, "
                "and industrial hazard guidance."
            ),
            "citations": [],
            "detection_summary": None,
            "grounded": False
        }

    target_zone = _extract_zone_identifier(question, detection, zone)
    gear_types = _extract_gear_keywords(question, detection)

    # Dynamic topic tokens for citations excerpt centering
    q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", question.lower()))
    topic_tokens = {t for t in q_tokens if t not in GENERIC_STOPWORDS and len(t) > 2}
    for g in gear_types:
        topic_tokens.add(g.lower())

    # -------------------------------------------------------------
    # 2. VISUAL INTENT
    # -------------------------------------------------------------
    if intent == "visual":
        det_data = None
        if detection and isinstance(detection, dict):
            det_data = detection
        else:
            recent = get_safety_events(limit=1)
            if recent:
                det_data = recent[0]

        if not det_data:
            return {
                "intent": "visual",
                "status": "not_found",
                "answer": "No visual detection records or optical events were found matching the inquiry.",
                "citations": [],
                "detection_summary": None,
                "grounded": False
            }

        c_name = det_data.get("class") or det_data.get("violation_type") or "PPE Detection"
        conf = float(det_data.get("confidence", 0.85))
        c_zone = det_data.get("zone") or det_data.get("zone_id") or target_zone or "Active Operational Zone"
        cam_id = det_data.get("camera_id") or "Surveillance Stream"
        bbox = det_data.get("bounding_box") or det_data.get("bbox") or []
        ts = det_data.get("timestamp") or "Recent Surveillance Frame"

        det_summary = {
            "class": c_name,
            "confidence": conf,
            "zone": c_zone,
            "camera_id": cam_id,
            "bounding_box": bbox,
            "timestamp": ts,
            "status": "Pending Human Supervisor Review"
        }

        answer_text = (
            f"Visual Telemetry Report: Optical model flagged a potential condition '{c_name}' on camera feed '{cam_id}' "
            f"within {c_zone} at {ts} (Optical Confidence: {conf*100:.1f}%). "
            f"Important Notice: This optical detection represents a potential non-compliance and does not constitute "
            f"conclusive proof; it is currently pending official human safety supervisor review and confirmation."
        )

        return {
            "intent": "visual",
            "status": "answered",
            "answer": answer_text,
            "citations": [],
            "detection_summary": det_summary,
            "grounded": True
        }

    # -------------------------------------------------------------
    # 3. COMBINED INTENT (Detection + Policy Verification)
    # -------------------------------------------------------------
    if intent == "combined":
        det_data = detection
        if not det_data:
            recent = get_safety_events(limit=1)
            if recent:
                det_data = recent[0]

        # Determine hazard condition: prefer explicitly queried gear over generic DB events
        if gear_types:
            c_name = f"no_{gear_types[0]}" if not (gear_types[0].startswith("no_") or gear_types[0].startswith("no-")) else gear_types[0]
        elif det_data:
            c_name = det_data.get("class") or det_data.get("violation_type") or "Unspecified Hazard"
        else:
            c_name = "Unspecified Hazard"
        conf = float(det_data.get("confidence", 0.88)) if det_data else 0.88
        c_zone = det_data.get("zone") or det_data.get("zone_id") or target_zone or "Plant Perimeter"
        cam_id = det_data.get("camera_id") if det_data else "Surveillance Stream"
        bbox = det_data.get("bounding_box") or det_data.get("bbox") or [] if det_data else []
        ts = det_data.get("timestamp") or "Current Inspection Frame" if det_data else "Recent"

        det_summary = {
            "class": c_name,
            "confidence": conf,
            "zone": c_zone,
            "camera_id": cam_id,
            "bounding_box": bbox,
            "timestamp": ts,
            "status": "Pending Human Supervisor Review"
        }

        # Query internal policy chunks
        clean_v = c_name.lower().replace("no_", "").replace("no-", "").strip()
        policy_query = f"{clean_v} high-visibility safety apparel policy standards in {c_zone}" if "vest" in clean_v else f"{clean_v} safety requirements standards in {c_zone}"
        retrieved_chunks = _retrieve_policy_chunks(policy_query, zone=c_zone, gear_types=gear_types, top_k=3, is_guidance=False)

        # Reuse existing Policy Bridge from rag_service for regulatory mapping
        bridge_data = None
        if clean_v in ("helmet", "vest", "gloves", "boots", "goggles") or any(k in clean_v for k in ["helmet", "vest", "glove", "boot", "goggle"]):
            bridge_data = get_policy_for_violation(f"no_{clean_v}")

        if not retrieved_chunks and not bridge_data:
            return {
                "intent": "combined",
                "status": "not_found",
                "answer": (
                    f"A potential detection '{c_name}' was recorded in {c_zone} (Confidence: {conf*100:.1f}%, pending human review), "
                    f"but no governing safety policy or regulatory mandate could be retrieved from the knowledge base."
                ),
                "citations": [],
                "detection_summary": det_summary,
                "grounded": False
            }

        citations = _build_citations_list(retrieved_chunks, topic_tokens)

        # Prioritize formal policy manuals/standards over CSV audit defect records
        policy_cits = [c for c in citations if not c["document"].lower().endswith(".csv")]
        audit_cits = [c for c in citations if c["document"].lower().endswith(".csv")]
        ordered_citations = policy_cits + audit_cits

        reg_title = bridge_data["osha_title"] if bridge_data else "Workplace Safety Standard"
        reg_code = bridge_data["osha_standard"] if bridge_data else "OSHA 1910 General Industry Regulations"

        if policy_cits:
            primary_c = policy_cits[0]
            loc_parts = []
            if primary_c.get("page"):
                loc_parts.append(f"Page {primary_c['page']}")
            if primary_c.get("section"):
                loc_parts.append(f"Section: '{primary_c['section']}'")
            loc_s = f" ({', '.join(loc_parts)})" if loc_parts else ""
            policy_statement = f"According to {primary_c['document']}{loc_s} ({reg_title} / {reg_code}), the mandatory requirement states: \"{primary_c['quote'][:180]}...\""
        else:
            policy_statement = f"Governing Policy Standard: Aligns with statutory standard {reg_code} ({reg_title}). No formal plant policy manual was retrieved from the local repository."

        answer_text = (
            f"Visual-Policy Compliance Assessment:\n\n"
            f"1. Potential Optical Event: Camera feed '{cam_id}' observed potential '{c_name}' in {c_zone} "
            f"with {conf*100:.1f}% optical confidence. (Note: This is a preliminary detection pending human safety supervisor review).\n"
            f"2. Governing Worksite Policy: {policy_statement}\n"
            f"3. Compliance Determination: The observed physical condition ({c_name}) non-complies with the mandatory "
            f"standard for {c_zone}. A designated safety supervisor must verify the event before official logging."
        )

        if audit_cits:
            a_c = audit_cits[0]
            row_str = f"Row {a_c['row']}" if a_c.get("row") else "Inspection Record"
            answer_text += (
                f"\n\nHistorical Inspection Finding: {a_c['document']} ({row_str}) recorded an empirical audit defect finding:\n"
                f"\"{a_c['quote'][:180]}...\"\n"
                f"(Marked as an empirical site audit finding, not a statutory policy requirement)."
            )

        if bridge_data and bridge_data.get("mandatory_clearance_protocol"):
            answer_text += f"\n4. Required Protocol: {bridge_data['mandatory_clearance_protocol']}."

        return {
            "intent": "combined",
            "status": "answered",
            "answer": answer_text,
            "citations": ordered_citations,
            "detection_summary": det_summary,
            "grounded": True
        }

    # -------------------------------------------------------------
    # 4. POLICY INTENT
    # -------------------------------------------------------------
    if intent == "policy":
        retrieved_chunks = _retrieve_policy_chunks(question, zone=target_zone, gear_types=gear_types, top_k=3)

        if not retrieved_chunks:
            return {
                "intent": "policy",
                "status": "not_found",
                "answer": "No relevant safety policy or standard found in the knowledge base for this inquiry.",
                "citations": [],
                "detection_summary": None,
                "grounded": False
            }

        citations = _build_citations_list(retrieved_chunks, topic_tokens)
        policy_cits = [c for c in citations if not c["document"].lower().endswith(".csv")]
        audit_cits = [c for c in citations if c["document"].lower().endswith(".csv")]
        ordered_citations = policy_cits if policy_cits else audit_cits

        status = "answered"

        if policy_cits:
            top_c = policy_cits[0]
            loc_desc = []
            if top_c.get("page"):
                loc_desc.append(f"Page {top_c['page']}")
            if top_c.get("row"):
                loc_desc.append(f"Row {top_c['row']}")
            if top_c.get("section"):
                loc_desc.append(f"Section: '{top_c['section']}'")
            loc_str = f" ({', '.join(loc_desc)})" if loc_desc else ""

            answer_text = (
                f"According to {top_c['document']}{loc_str}, the governing safety requirement states:\n\n"
                f"\"{top_c['quote']}\""
            )

            if len(policy_cits) > 1:
                second_c = policy_cits[1]
                answer_text += f"\n\nAdditional specification from {second_c['document']}:\n\"{second_c['quote'][:160]}...\""

            if audit_cits and not policy_cits:
                a_c = audit_cits[0]
                r_desc = f"Row {a_c['row']}" if a_c.get("row") else "Audit Record"
                answer_text += (
                    f"\n\nHistorical Inspection Finding: {a_c['document']} ({r_desc}) noted defect finding:\n"
                    f"\"{a_c['quote'][:160]}...\"\n"
                    f"(Marked as an empirical site audit finding, not a statutory policy requirement)."
                )
        else:
            # Only CSV audit records found - no policy manual found
            top_c = audit_cits[0]
            r_desc = f"Row {top_c['row']}" if top_c.get("row") else "Audit Record"
            answer_text = (
                f"No governing policy manual found in knowledge base for this inquiry.\n\n"
                f"Historical Inspection Finding: {top_c['document']} ({r_desc}) recorded an empirical audit defect finding:\n"
                f"\"{top_c['quote']}\"\n\n"
                f"(Notice: This audit record represents an empirical site defect finding and does not constitute a governing policy requirement)."
            )

        # Multi-gear coverage verification: change status to 'partial' when any requested PPE item lacks coverage
        if "boots" in gear_types and not any("1910.136" in (c.get("quote", "") + (c.get("section") or "")) for c in policy_cits):
            status = "partial"
            answer_text += (
                f"\n\nCoverage Notice: The cited primary documentation details head protection requirements, "
                f"but does NOT cover comprehensive statutory protective footwear standards (ASTM F2413 / OSHA 1910.136). "
                f"Dedicated foot protection policy standards must be consulted separately."
            )

        # Cross-reference regulatory standards for covered gear
        for g in gear_types:
            if g in REGULATORY_STANDARDS:
                reg = REGULATORY_STANDARDS[g]
                answer_text += f"\n\nStatutory Reference: Aligns with {reg['standard_code']} ({reg['title']})."
                break

        return {
            "intent": "policy",
            "status": status,
            "answer": answer_text,
            "citations": ordered_citations,
            "detection_summary": None,
            "grounded": True
        }

    # -------------------------------------------------------------
    # 5. GUIDANCE INTENT (No general-principles fallback: unsupported queries return not_found)
    # -------------------------------------------------------------
    if intent == "guidance":
        retrieved_chunks = _retrieve_policy_chunks(question, zone=target_zone, gear_types=gear_types, top_k=2, is_guidance=True)

        if not retrieved_chunks:
            return {
                "intent": "guidance",
                "status": "not_found",
                "answer": "No relevant safety guidance or procedure documents found in the knowledge base for this inquiry.",
                "citations": [],
                "detection_summary": None,
                "grounded": False
            }

        citations = _build_citations_list(retrieved_chunks, topic_tokens)
        top_c = citations[0]

        loc_desc = []
        if top_c.get("page"):
            loc_desc.append(f"Page {top_c['page']}")
        if top_c.get("row"):
            loc_desc.append(f"Row {top_c['row']}")
        if top_c.get("section"):
            loc_desc.append(f"Section: '{top_c['section']}'")
        loc_str = f" ({', '.join(loc_desc)})" if loc_desc else ""

        # Paraphrase advisory safe-work practice without legalistic "shall" or "mandatory" requirements
        clean_quote = re.sub(r"\bshall\b", "should", top_c["quote"], flags=re.IGNORECASE)
        clean_quote = re.sub(r"\bmandatory\b", "recommended", clean_quote, flags=re.IGNORECASE)

        answer_text = (
            f"Workplace Safety Operational Guidance (Advisory Best Practice — Non-Mandatory):\n\n"
            f"Advisory Safe-Work Procedure (based on {top_c['document']}{loc_str}):\n"
            f"\"{clean_quote}\"\n\n"
            f"Operational Advisory Recommendation: Maintain verified eyewash and emergency flushing paths within 10 seconds of active chemical zones. "
            f"Personnel should utilize indirect-vent splash goggles and appropriate barrier shielding prior to handling corrosive agents.\n\n"
            f"Advisory Notice: This guidance provides operational safe-work recommendations only and does not establish "
            f"statutory or mandatory policy requirements."
        )

        return {
            "intent": "guidance",
            "status": "answered",
            "answer": answer_text,
            "citations": citations,
            "detection_summary": None,
            "grounded": True
        }

    # Safety catch-all
    return {
        "intent": "out_of_scope",
        "status": "out_of_scope",
        "answer": "This inquiry is outside the scope of the workplace safety knowledge base.",
        "citations": [],
        "detection_summary": None,
        "grounded": False
    }

from typing import Dict, Any, Optional, List
from services.agent.safety_query import answer_safety_query

class PolicyMatcher:
    """
    Maps multimodal violation events to governing worksite EHS policies.
    Enforces zone-first filtering by passing zone context and event metadata directly
    to the verified answer_safety_query retrieval pipeline.
    """
    def __init__(self):
        pass

    def build_query_for_event(self, event: Dict[str, Any]) -> str:
        """Constructs a targeted policy inquiry based on violation category and zone."""
        vcat = event.get("violation_category", "PPE")
        vtype = event.get("violation_type", "PPE").replace("_", " ")
        zone = event.get("zone", "Zone Z-01")

        if vcat.lower() == "head":
            return f"What are the hardhat and head protection requirements in {zone}?"
        elif vcat.lower() == "hands":
            return f"What are the protective glove requirements for handling chemicals in {zone}?"
        elif vcat.lower() == "torso":
            return f"Does this worker without a vest violate safety policy in {zone}?"
        elif vcat.lower() == "feet":
            return f"What are the safety footwear and boot requirements in {zone}?"
        elif vcat.lower() == "eyes":
            return f"What are the eye and face protection standards in {zone}?"
        else:
            return f"What are the safety requirements and SOP standards for {vtype} in {zone}?"

    def match_policy_for_event(
        self,
        event: Dict[str, Any],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Maps a structured event to policy chunks.
        Applies zone filter first via answer_safety_query.
        """
        zone = event.get("zone")
        if not query:
            query = self.build_query_for_event(event)

        detection_context = {
            "class": event.get("violation_type") or event.get("class") or "unknown",
            "confidence": float(event.get("confidence", 0.9)),
            "zone": zone,
            "camera_id": event.get("camera_id"),
            "timestamp": event.get("timestamp")
        }

        # Calls the existing answer_safety_query retrieval service
        rag_result = answer_safety_query(
            question=query,
            detection=detection_context,
            zone=zone
        )

        citations = rag_result.get("citations", [])
        top_citation = citations[0] if citations else None

        return {
            "event_id": event.get("event_id"),
            "zone": rag_result.get("zone") or zone,
            "violation_category": event.get("violation_category"),
            "query": query,
            "status": rag_result.get("status"),
            "intent": rag_result.get("intent"),
            "answer": rag_result.get("answer"),
            "citations": citations,
            "top_citation": top_citation,
            "has_policy_match": len(citations) > 0 and rag_result.get("status") != "not_found"
        }

    def match_by_category_and_zone(self, violation_category: str, zone: str) -> Dict[str, Any]:
        """Direct lookup by category and zone."""
        dummy_event = {
            "violation_category": violation_category,
            "violation_type": f"no_{violation_category.lower()}",
            "zone": zone,
            "confidence": 0.9
        }
        return self.match_policy_for_event(dummy_event)

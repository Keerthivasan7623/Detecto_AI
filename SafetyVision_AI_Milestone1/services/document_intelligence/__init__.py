"""
Detecto AI - Document Intelligence Package
Milestone 2: Hybrid RAG Knowledge Base, Dual-Path Extraction, and Visual-Policy Bridge.
"""

from services.document_intelligence.uploader import (
    validate_and_store_document,
    get_status_badge,
    SUPPORTED_EXTENSIONS,
    MAX_DOCUMENT_SIZE_BYTES
)

from services.document_intelligence.extractor import (
    extract_document_text,
    clean_text_noise,
    extract_ehs_entities,
    generate_document_summary
)

from services.document_intelligence.vector_store import (
    VectorStore,
    get_vector_store,
    chunk_text_content,
    process_extracted_items_into_chunks
)

from services.document_intelligence.rag_service import (
    RAGService,
    get_rag_service,
    get_policy_for_violation,
    synthesize_lead_ehs_directive,
    ask_open_world_qa,
    answer_safety_policy,
    REGULATORY_STANDARDS
)

from services.document_intelligence.seed_data import seed_default_knowledge_base, load_sample_ehs_library

__all__ = [
    "validate_and_store_document",
    "get_status_badge",
    "SUPPORTED_EXTENSIONS",
    "MAX_DOCUMENT_SIZE_BYTES",
    "extract_document_text",
    "clean_text_noise",
    "extract_ehs_entities",
    "generate_document_summary",
    "VectorStore",
    "get_vector_store",
    "chunk_text_content",
    "process_extracted_items_into_chunks",
    "RAGService",
    "get_rag_service",
    "get_policy_for_violation",
    "synthesize_lead_ehs_directive",
    "ask_open_world_qa",
    "answer_safety_policy",
    "REGULATORY_STANDARDS",
    "seed_default_knowledge_base",
    "load_sample_ehs_library",
]

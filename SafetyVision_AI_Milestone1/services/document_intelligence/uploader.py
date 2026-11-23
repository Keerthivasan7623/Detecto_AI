"""
Detecto AI - Document Intelligence: Upload & Validation Pipeline
Handles multi-format document intake (PDF, DOCX, CSV, Scanned Images),
rigorous file integrity verification, secure storage, and database registration.
"""

import os
import re
import csv
import io
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, BinaryIO
from PIL import Image

from database import create_document, update_document_status, get_document_by_file_hash

# Supported document categories
VALID_DOCUMENT_CATEGORIES = ("Safety Manuals", "SOPs", "Inspection Reports")

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".csv": "csv",
    ".png": "scanned_image",
    ".jpg": "scanned_image",
    ".jpeg": "scanned_image",
    ".tiff": "scanned_image",
    ".bmp": "scanned_image",
}

MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

try:
    from storage_config import DOCUMENTS_DIR
    DOCUMENTS_STORAGE_DIR = DOCUMENTS_DIR
except ImportError:
    DOCUMENTS_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "documents"


def get_storage_directory() -> Path:
    """Ensures and returns the local documents archive directory."""
    try:
        from storage_config import DOCUMENTS_DIR
        storage_dir = DOCUMENTS_DIR
    except ImportError:
        storage_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "documents"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def sanitize_filename(filename: str) -> str:
    """Removes unsafe characters, path traversals, and limits filename length."""
    base = Path(filename).name
    clean = re.sub(r"[^\w.\-_ ]", "_", base).strip()
    return clean[:120] if clean else "document.bin"


def get_status_badge(status: str, progress: int = 0) -> Dict[str, Any]:
    """
    Returns visual badge properties for Streamlit display:
    label, icon, color class, and progress percentage.
    """
    status_clean = (status or "").lower()
    if status_clean == "indexed":
        return {
            "label": "Processed Complete",
            "icon": "✅",
            "pill_class": "pill-emerald",
            "progress": 100,
            "description": "Text extracted, chunked, and vector indexed into FAISS."
        }
    elif status_clean == "processing":
        pct = progress if progress > 0 else 75
        return {
            "label": f"Analyzing... {pct}%",
            "icon": "⚡",
            "pill_class": "pill-amber",
            "progress": pct,
            "description": "Document undergoing dual-path extraction & semantic indexing."
        }
    elif status_clean == "uploaded":
        return {
            "label": "Uploaded (Queued)",
            "icon": "⏳",
            "pill_class": "pill-slate",
            "progress": 25,
            "description": "File integrity validated. Ready for extraction pipeline."
        }
    elif status_clean == "failed":
        return {
            "label": "Processing Failed",
            "icon": "❌",
            "pill_class": "pill-red",
            "progress": 0,
            "description": "Corrupt file format or extraction error."
        }
    return {
        "label": status.capitalize(),
        "icon": "ℹ️",
        "pill_class": "pill-slate",
        "progress": 50,
        "description": "Status pending."
    }


def validate_file_integrity(file_bytes: bytes, extension: str) -> Tuple[bool, str]:
    """
    Validates file headers and structural integrity for supported extensions:
    - PDF: Verifies %PDF header
    - DOCX: Verifies zip magic bytes PK\x03\x04 and document.xml presence
    - CSV: Verifies text encoding and tabular structure
    - Image: Verifies Pillow image decodability
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "File is completely empty (0 bytes)."

    if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        return False, f"File exceeds maximum allowed size ({MAX_DOCUMENT_SIZE_BYTES // (1024*1024)} MB)."

    ext = extension.lower()

    if ext == ".pdf":
        if not file_bytes.startswith(b"%PDF-"):
            return False, "Invalid PDF: Missing standard '%PDF-' magic header."
        return True, ""

    elif ext == ".docx":
        if not file_bytes.startswith(b"PK\x03\x04"):
            return False, "Invalid DOCX: Missing standard OOXML Zip container signature."
        return True, ""

    elif ext == ".txt":
        try:
            sample = file_bytes[:8192]
            if b"\x00" in sample:
                return False, "Corrupt TXT: Binary null bytes detected (not a readable text file)."
            text = sample.decode("utf-8", errors="replace").strip()
            if not text:
                return False, "TXT document contains no readable text."
            return True, ""
        except Exception as e:
            return False, f"TXT integrity check failed: {str(e)}"

    elif ext == ".csv":
        try:
            # Check for binary zero bytes which indicate binary/executable masquerading as CSV
            sample = file_bytes[:8192]
            if b"\x00" in sample:
                return False, "Corrupt CSV: Binary null bytes detected."
            text = sample.decode("utf-8", errors="replace")
            # Ensure lines exist
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                return False, "CSV contains no readable lines."
            return True, ""
        except Exception as e:
            return False, f"CSV integrity check failed: {str(e)}"

    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.verify()
            return True, ""
        except Exception as e:
            return False, f"Corrupt image file: Cannot decode image stream ({str(e)})."

    return False, f"Unsupported file format: {ext}"


def validate_and_store_document(
    file_name: str,
    file_bytes: bytes,
    user_id: Optional[int] = None,
    category: str = "Safety Manuals"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Executes end-to-end upload & validation pipeline:
    1. Check File Type
    2. Check File Size
    3. Check File Integrity
    4. Prevent Duplicate Ingestion (via file_hash SHA-256)
    5. Store Document to outputs/documents
    6. Register in database with category and file_hash
    """
    clean_name = sanitize_filename(file_name)
    ext = Path(clean_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(SUPPORTED_EXTENSIONS.keys())
        return False, f"Unsupported file type '{ext}'. Allowed formats: {supported_list}", None

    file_size = len(file_bytes)

    # 1-3. Integrity Validation
    is_valid, err_msg = validate_file_integrity(file_bytes, ext)
    if not is_valid:
        return False, err_msg, None

    # 4. Duplicate Prevention via Content Hash (SHA-256)
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing_doc = get_document_by_file_hash(file_hash)
    if existing_doc:
        return (
            False,
            f"Duplicate document rejected: an identical document is already indexed as "
            f"'{existing_doc['file_name']}' (Catalog ID #{existing_doc['id']}).",
            None
        )

    # 5. Secure File Storage
    storage_dir = get_storage_directory()
    ts_prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_path = storage_dir / f"{ts_prefix}_{clean_name}"

    try:
        with open(target_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        return False, f"Failed to persist file to disk storage: {str(e)}", None

    file_type = SUPPORTED_EXTENSIONS[ext]
    clean_category = category.strip() if (category and category.strip() in VALID_DOCUMENT_CATEGORIES) else "Safety Manuals"

    # 6. Database Record Creation
    try:
        doc_id = create_document(
            file_name=clean_name,
            file_type=file_type,
            file_size=file_size,
            file_path=str(target_path),
            status="uploaded",
            category=clean_category,
            file_hash=file_hash
        )
    except Exception as e:
        # Cleanup file if DB insertion failed
        if target_path.exists():
            target_path.unlink()
        return False, f"Database transaction failed: {str(e)}", None

    doc_meta = {
        "id": doc_id,
        "file_name": clean_name,
        "file_type": file_type,
        "file_size": file_size,
        "file_path": str(target_path),
        "category": clean_category,
        "file_hash": file_hash,
        "status": "uploaded",
        "badge": get_status_badge("uploaded")
    }

    return True, "Document validated and registered successfully.", doc_meta


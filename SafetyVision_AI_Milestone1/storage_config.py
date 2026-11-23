"""SafetyVision AI / Detecto AI - Centralized Storage Configuration.
Supports Railway persistent storage (/data) and local development.

Environment Variables:
- SAFETYVISION_DATA_DIR: Path to persistent storage directory.
  On Railway: Set to /data
  Locally: Defaults to application directory (SafetyVision_AI_Milestone1)
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Union

# Application root directory (where app.py and models/best.pt reside)
APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = APP_DIR.parent

# Ensure APP_DIR is on sys.path for submodule imports
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Centralized data directory resolution
_ENV_DATA_DIR = os.environ.get("SAFETYVISION_DATA_DIR")
if _ENV_DATA_DIR and _ENV_DATA_DIR.strip():
    DATA_DIR = Path(_ENV_DATA_DIR.strip()).resolve()
    IS_PERSISTENT_VOLUME = True
else:
    DATA_DIR = APP_DIR
    IS_PERSISTENT_VOLUME = False

# SQLite database file path (WAL and SHM files will be created in this directory)
DB_PATH = DATA_DIR / "safety.db"

# Centralized outputs directory
if IS_PERSISTENT_VOLUME:
    OUTPUTS_DIR = DATA_DIR / "outputs"
else:
    OUTPUTS_DIR = APP_DIR / "outputs"

# Subdirectories for runtime generated files
ANNOTATED_DIR = OUTPUTS_DIR / "annotated"
OUTPUT_DIR = ANNOTATED_DIR  # Backward-compatible alias
REPORTS_DIR = OUTPUTS_DIR / "reports"
REPORT_DIR = REPORTS_DIR    # Backward-compatible alias
EVIDENCE_DIR = OUTPUTS_DIR / "evidence"
DOCUMENTS_DIR = OUTPUTS_DIR / "documents"
DOCUMENTS_STORAGE_DIR = DOCUMENTS_DIR  # Backward-compatible alias
INVESTIGATIONS_DIR = OUTPUTS_DIR / "investigations"
VECTOR_STORE_DIR = OUTPUTS_DIR / "vector_store"

# FAISS Vector Store persistent index & metadata file paths
INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_STORE_DIR / "chunks_metadata.json"

# Read-only assets (Never moved to persistent volume)
MODEL_PATH = APP_DIR / "models" / "best.pt"
SAMPLE_DIR = APP_DIR / "datasets" / "construction-ppe" / "images" / "test"
CONFIG_DIR = APP_DIR / "config"

ALL_STORAGE_DIRS: List[Path] = [
    DATA_DIR,
    OUTPUTS_DIR,
    ANNOTATED_DIR,
    REPORTS_DIR,
    EVIDENCE_DIR,
    DOCUMENTS_DIR,
    INVESTIGATIONS_DIR,
    VECTOR_STORE_DIR,
]


def ensure_storage_directories() -> None:
    """Creates all required persistent storage directories if they do not exist."""
    for d in ALL_STORAGE_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def init_persistent_storage() -> None:
    """
    Initializes persistent volume storage.
    Ensures directories exist.
    If running on an external persistent mount (e.g. /data) for the first time:
    - Seeds SQLite database from bundled template if persistent DB does not exist yet.
    - Seeds FAISS index and metadata if persistent vector store is unpopulated.
    - Seeds sample documents if documents directory is empty.
    Never overwrites existing files.
    """
    ensure_storage_directories()

    if not IS_PERSISTENT_VOLUME:
        return

    # Seed SQLite database if missing in persistent directory
    bundled_db = APP_DIR / "safety.db"
    if not DB_PATH.exists() and bundled_db.exists():
        try:
            shutil.copy2(bundled_db, DB_PATH)
        except Exception:
            pass

    # Seed FAISS index if missing in persistent directory
    bundled_index = APP_DIR / "outputs" / "vector_store" / "faiss_index.bin"
    bundled_meta = APP_DIR / "outputs" / "vector_store" / "chunks_metadata.json"
    if not INDEX_FILE.exists() and bundled_index.exists():
        try:
            shutil.copy2(bundled_index, INDEX_FILE)
        except Exception:
            pass
    if not METADATA_FILE.exists() and bundled_meta.exists():
        try:
            shutil.copy2(bundled_meta, METADATA_FILE)
        except Exception:
            pass

    # Seed sample documents if documents directory is empty
    bundled_docs_dir = APP_DIR / "outputs" / "documents"
    if bundled_docs_dir.exists() and bundled_docs_dir.is_dir():
        try:
            existing_docs = list(DOCUMENTS_DIR.iterdir()) if DOCUMENTS_DIR.exists() else []
            if not existing_docs:
                for item in bundled_docs_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, DOCUMENTS_DIR / item.name)
        except Exception:
            pass


def resolve_runtime_path(path_str: Optional[Union[str, Path]]) -> Optional[str]:
    """
    Resolves relative or absolute path against storage directories.
    Handles paths saved as 'outputs/evidence/xxx.jpg', '/data/safety.db', etc.
    Returns resolved absolute string path if file exists, or None.
    """
    if not path_str:
        return None
    p_str = str(path_str).strip()
    if not p_str:
        return None

    p = Path(p_str)
    if p.is_file():
        return str(p.resolve())

    # Candidate locations to inspect
    candidates = [
        DATA_DIR / p_str,
        OUTPUTS_DIR / p_str,
        EVIDENCE_DIR / p.name,
        ANNOTATED_DIR / p.name,
        DOCUMENTS_DIR / p.name,
        INVESTIGATIONS_DIR / p.name,
        APP_DIR / p_str,
        WORKSPACE_DIR / p_str,
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())

    return None


# Automatically ensure directories exist upon module load
ensure_storage_directories()

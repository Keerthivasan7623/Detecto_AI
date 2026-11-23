import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

_CACHED_CONFIG: Optional[Dict[str, Any]] = None

def get_default_config_path() -> Path:
    """Finds config/ppe_categories.json relative to project root or current working dir."""
    curr = Path(__file__).resolve().parent
    for parent in [curr, curr.parent, curr.parent.parent]:
        p = parent / "config" / "ppe_categories.json"
        if p.exists():
            return p
    return Path("config/ppe_categories.json")

def load_ppe_categories_config(config_path: Optional[Union[str, Path]] = None, reload: bool = False) -> Dict[str, Any]:
    """
    Loads and caches configuration from config/ppe_categories.json.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not reload and config_path is None:
        return _CACHED_CONFIG

    target_path = Path(config_path) if config_path else get_default_config_path()
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config_path is None:
        _CACHED_CONFIG = config

    return config

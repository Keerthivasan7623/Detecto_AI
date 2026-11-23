"""Root re-export wrapper for centralized storage configuration."""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent / "SafetyVision_AI_Milestone1"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from storage_config import *  # noqa: F401, F403

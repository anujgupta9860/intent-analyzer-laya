"""Service configuration (environment variables)."""
from __future__ import annotations

import os


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings:
    def __init__(self) -> None:
        # Directory holding one trained checkpoint per agent:
        #   <dir>/billing/calibration.json, <dir>/orders/..., ...
        self.checkpoint_dir: str = _get("ANALYZER_CHECKPOINT_DIR", "models")
        # Choice confidence below this -> needs_human + feedback logging.
        self.confidence_threshold: float = float(
            _get("ANALYZER_CONFIDENCE_THRESHOLD", "0.6"))
        # Set FEEDBACK_ENABLED=false to disable low-confidence logging
        # (e.g. for eval runs you don't want polluting the training pool).
        self.feedback_enabled: bool = _get(
            "FEEDBACK_ENABLED", "true").strip().lower() in (
            "1", "true", "yes", "on")
        self.log_level: str = _get("LOG_LEVEL", "INFO").upper()

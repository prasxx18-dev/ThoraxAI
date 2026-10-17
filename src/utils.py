"""Small shared helpers: logging, JSON I/O, seeding and environment capture."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "thoraxai"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a project logger; configures a console handler once."""
    root = logging.getLogger(LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return root.getChild(name) if name else root


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _clean_nans(obj: Any) -> Any:
    """Replace float NaN/inf with None so the JSON output is standards-compliant."""
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    return obj


def save_json(data: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_clean_nans(data), fh, indent=2, default=_json_default)
    return path


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_json_if_exists(path: Path) -> Any | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        get_logger("utils").warning("Could not read %s: %s", path, exc)
        return None


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - TF is optional for some helpers
        pass


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment_info() -> dict:
    """Record software versions for reproducibility."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for module in ("tensorflow", "keras", "pandas", "sklearn", "PIL", "streamlit"):
        try:
            mod = __import__(module)
            info[module] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[module] = None
    try:
        import tensorflow as tf

        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        info["gpus"] = []
    return info


def file_size_mb(path: Path) -> float | None:
    path = Path(path)
    return round(path.stat().st_size / (1024 * 1024), 2) if path.is_file() else None

"""Central configuration for ThoraxAI.

All filesystem locations are derived from the project root, an optional
machine-local settings file, or environment variables, so no machine-specific
path is hard-coded in the code base.

Precedence (highest first)
--------------------------
1. Environment variables:
   THORAXAI_DATA_DIR     Root folder of the extracted NIH ChestX-ray14 dataset.
   THORAXAI_METADATA_CSV Explicit path to Data_Entry_2017*.csv (optional).
   THORAXAI_IMAGE_ROOT   Explicit folder that contains the PNG images (optional).
   THORAXAI_MODELS_DIR   Where trained models are stored (default: ./models).
   THORAXAI_RESULTS_DIR  Where reports/metrics are stored (default: ./results).
2. ``local_config.json`` in the project root (git-ignored), e.g.
   {"data_dir": "C:/datasets/nih"} with optional keys metadata_csv, image_root,
   models_dir, results_dir. Relative paths are resolved against the project root.
3. Built-in defaults (data/chest_xray, models, results inside the project).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_FILE = PROJECT_ROOT / "local_config.json"


def _load_local_config(path: Optional[Path] = None) -> dict:
    path = Path(path) if path is not None else LOCAL_CONFIG_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Ignoring unreadable {path.name}: {exc}", stacklevel=2)
        return {}
    if not isinstance(data, dict):
        warnings.warn(f"Ignoring {path.name}: expected a JSON object", stacklevel=2)
        return {}
    return data


LOCAL_SETTINGS = _load_local_config()


def _setting(env_name: str, key: str, default: Optional[Path], local: Optional[dict] = None) -> Optional[Path]:
    """Resolve a path from the environment, then local_config.json, then the default."""
    local = LOCAL_SETTINGS if local is None else local
    value = os.environ.get(env_name) or local.get(key)
    if not value:
        return default
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


# module attribute -> (environment variable, local_config.json key, default)
_PATH_SETTINGS = {
    "DATA_DIR": ("THORAXAI_DATA_DIR", "data_dir", PROJECT_ROOT / "data" / "chest_xray"),
    "METADATA_CSV": ("THORAXAI_METADATA_CSV", "metadata_csv", None),
    "IMAGE_ROOT": ("THORAXAI_IMAGE_ROOT", "image_root", None),
    "MODELS_DIR": ("THORAXAI_MODELS_DIR", "models_dir", PROJECT_ROOT / "models"),
    "RESULTS_DIR": ("THORAXAI_RESULTS_DIR", "results_dir", PROJECT_ROOT / "results"),
}


def resolve_paths(local: Optional[dict] = None) -> dict[str, Optional[Path]]:
    """Current path settings from the environment and ``local_config.json``."""
    local = LOCAL_SETTINGS if local is None else local
    return {name: _setting(env, key, default, local) for name, (env, key, default) in _PATH_SETTINGS.items()}


def refresh_paths() -> dict[str, Optional[Path]]:
    """Re-read ``local_config.json`` and the environment and update the path settings.

    Values above are computed once at import. A long-running process (the
    Streamlit server) would otherwise keep using the paths that were valid when
    it started, even after ``local_config.json`` is created or edited. The app
    calls this before checking its status; command-line scripts are fresh
    processes and do not need it.
    """
    global LOCAL_SETTINGS
    LOCAL_SETTINGS = _load_local_config(LOCAL_CONFIG_FILE)
    paths = resolve_paths(LOCAL_SETTINGS)
    globals().update(paths)
    return paths


_initial = resolve_paths()
DATA_DIR: Path = _initial["DATA_DIR"]
METADATA_CSV: Optional[Path] = _initial["METADATA_CSV"]
IMAGE_ROOT: Optional[Path] = _initial["IMAGE_ROOT"]
MODELS_DIR: Path = _initial["MODELS_DIR"]
RESULTS_DIR: Path = _initial["RESULTS_DIR"]

# --------------------------------------------------------------------------- #
# Label definitions
# --------------------------------------------------------------------------- #
# The order below defines the index of each sigmoid output. It must never be
# changed for an already-trained model (the label mapping is saved with it).
LABELS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_LABELS = len(LABELS)

# "No Finding" is the dataset's marker for an image with none of the 14 labels.
# It is NOT trained as a separate output: it is encoded as an all-zero target.
NO_FINDING_TOKEN = "No Finding"

DISPLAY_NAMES: dict[str, str] = {label: label.replace("_", " ") for label in LABELS}

# Aliases accepted when parsing label strings (case-insensitive).
LABEL_ALIASES: dict[str, str] = {
    "pleural thickening": "Pleural_Thickening",
    "pleural_thickening": "Pleural_Thickening",
}

# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MIN_IMAGE_SIDE = 64
MAX_IMAGE_PIXELS = 50_000_000  # guards against decompression bombs

# --------------------------------------------------------------------------- #
# Supported architectures
# --------------------------------------------------------------------------- #
# normalization: how raw RGB values in [0, 255] are transformed. This is built
# into the model graph (see src/model.py), so training, evaluation and the app
# always share exactly the same preprocessing.
ARCHITECTURES: dict[str, dict] = {
    "densenet121": {
        "display_name": "DenseNet121",
        "normalization": "imagenet_torch",  # x/255, then (x - mean) / std
        "default_image_size": 224,
    },
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "normalization": "tf_minus1_1",  # x/127.5 - 1
        "default_image_size": 224,
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """Hyper-parameters and options for one training run."""

    architecture: str = "densenet121"
    mode: str = "development"  # "development" (subset) or "full"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 5  # head-training epochs (stage 1)
    fine_tune_epochs: int = 0  # stage 2; 0 disables fine-tuning
    fine_tune_layers: int = 40  # number of trailing backbone layers to unfreeze
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    dropout: float = 0.3
    loss: str = "weighted_bce"  # "bce" or "weighted_bce"
    pretrained: bool = True  # ImageNet weights
    seed: int = 42
    val_fraction: float = 0.1  # fraction of train/val patients used for validation
    test_fraction: float = 0.2  # only used when no official split files exist
    dev_max_images: int = 3000  # total image budget in development mode
    early_stopping_patience: int = 3
    threshold_method: str = "f1"  # "f1" or "youden"
    max_missing_fraction: float = 0.02  # abort if more images than this are missing
    resume: bool = False
    augment: bool = True
    verify_images: bool = True  # integrity-check the selected images before training
    cache_dir: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

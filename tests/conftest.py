"""Shared pytest fixtures.

IMPORTANT: every image created here is SYNTHETIC random noise used only to
exercise the software. None of it is medical data, and no metric computed on
it says anything about diagnostic performance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config  # noqa: E402


def make_synthetic_dataset(root: Path, n_patients: int = 30, images_per_patient: int = 2, size: int = 72,
                           seed: int = 0, official_lists: bool = True, nested: bool = True) -> Path:
    """Create a tiny NIH-like folder (CSV + PNGs) of synthetic noise images."""
    rng = np.random.default_rng(seed)
    image_dir = root / "images_001" / "images" if nested else root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for p in range(1, n_patients + 1):
        for f in range(images_per_patient):
            name = f"{p:08d}_{f:03d}.png"
            if k < len(config.LABELS):  # guarantee every label appears at least once
                labels = config.LABELS[k]
            else:
                chosen = [lbl for lbl in config.LABELS if rng.random() < 0.12]
                labels = "|".join(chosen) if chosen else "No Finding"
            k += 1
            arr = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(image_dir / name)
            rows.append({"Image Index": name, "Finding Labels": labels, "Follow-up #": f, "Patient ID": p,
                         "Patient Age": 50, "Patient Gender": "M", "View Position": "PA"})
    df = pd.DataFrame(rows)
    df.to_csv(root / "Data_Entry_2017.csv", index=False)
    if official_lists:
        patients = df["Patient ID"].unique()
        test_patients = set(patients[: max(2, n_patients // 5)])
        df[df["Patient ID"].isin(test_patients)]["Image Index"].to_csv(root / "test_list.txt", index=False, header=False)
        df[~df["Patient ID"].isin(test_patients)]["Image Index"].to_csv(root / "train_val_list.txt", index=False,
                                                                         header=False)
    return root


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    return make_synthetic_dataset(tmp_path / "chest_xray")


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    """Point config at a temporary local_config.json, isolated from the developer's real one.

    Returns ``write(**settings)``, which (re)writes the file and calls
    ``config.refresh_paths()`` - the same mechanism the running app uses.
    THORAXAI_* environment variables are removed so they cannot override it.
    All path attributes are restored after the test.
    """
    import json

    for env_name, _key, _default in config._PATH_SETTINGS.values():
        monkeypatch.delenv(env_name, raising=False)
    for attr in (*config._PATH_SETTINGS, "LOCAL_SETTINGS"):
        monkeypatch.setattr(config, attr, getattr(config, attr))  # record originals for restore
    cfg_file = tmp_path / "local_config.json"
    monkeypatch.setattr(config, "LOCAL_CONFIG_FILE", cfg_file)

    def write(**settings) -> Path:
        defaults = {"data_dir": tmp_path / "no_data", "models_dir": tmp_path / "models",
                    "results_dir": tmp_path / "results"}
        data = {k: str(v) for k, v in {**defaults, **settings}.items()}
        cfg_file.write_text(json.dumps(data), encoding="utf-8")
        config.refresh_paths()
        return cfg_file

    write()
    return write


class MockModel:
    """Stand-in for a Keras model: returns fixed scores, records its input."""

    def __init__(self, scores, image_size: int = 64):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.input_shape = (None, image_size, image_size, 3)
        self.last_input = None

    def __call__(self, batch, training=False):
        self.last_input = batch
        return np.repeat(self.scores[np.newaxis, :], len(batch), axis=0)


@pytest.fixture
def mock_model_factory():
    return MockModel

"""Framework-independent inference (no Streamlit imports, so it is unit-testable).

Output handling
---------------
The ThoraxAI model ends in ``Dense(14, activation="sigmoid")``, so it already
returns one independent score in [0, 1] per label (multi-label; never softmax).
``Predictor`` inspects the final layer: sigmoid outputs are used as-is, and only
a model whose final activation is linear (raw logits) gets ``sigmoid`` applied -
exactly once. ``to_percent`` converts a score to a percentage for display.

Scores are NOT calibrated probabilities (no calibration has been performed), so
the UI labels them "model-estimated likelihood". A finding is *flagged* when its
score is >= its validation-selected threshold; when nothing is flagged that is a
model status, not a confirmed normal result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

import config
from src.preprocessing import load_image, prepare_batch
from src.utils import file_size_mb, get_logger, load_json_if_exists

log = get_logger("predict")


class PredictionError(RuntimeError):
    """Raised when the model output is unusable (wrong shape, NaN, ...)."""


@dataclass
class PredictionResult:
    scores: dict[str, float]
    thresholds: dict[str, float]
    flagged: list[str]
    no_finding_predicted: bool
    inference_ms: float
    original_size: tuple[int, int]
    model_input_size: int
    thresholds_validated: bool
    notes: list[str] = field(default_factory=list)

    @property
    def percentages(self) -> dict[str, float]:
        """Model-estimated likelihood per label, in percent (uncalibrated)."""
        return {lbl: to_percent(s) for lbl, s in self.scores.items()}

    def ranked(self) -> list[tuple[str, float, float, bool]]:
        """(label, score, threshold, flagged) sorted by descending score."""
        return sorted(
            ((lbl, s, self.thresholds[lbl], lbl in self.flagged) for lbl, s in self.scores.items()),
            key=lambda r: r[1],
            reverse=True,
        )


def to_percent(score: float) -> float:
    """Convert a [0, 1] model score to a percentage rounded to 0.1.

    Values are clamped to [0, 100] (guards against float round-off); NaN/inf raise
    ``PredictionError`` rather than being displayed.
    """
    value = float(score)
    if not np.isfinite(value):
        raise PredictionError("Model produced a non-finite score.")
    return round(min(max(value, 0.0), 1.0) * 100.0, 1)


def output_activation(model: Any) -> Optional[str]:
    """Name of the final layer's activation (e.g. 'sigmoid', 'linear'), or None if unknown."""
    layers = getattr(model, "layers", None)
    if not layers:
        return None
    act = getattr(layers[-1], "activation", None)
    return getattr(act, "__name__", None)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def flag_findings(scores: Sequence[float], thresholds: Sequence[float],
                  labels: Sequence[str] = config.LABELS) -> list[str]:
    """Labels whose score >= threshold, sorted by descending score."""
    if len(scores) != len(thresholds) or len(scores) != len(labels):
        raise ValueError("scores, thresholds and labels must have equal length")
    hits = [(lbl, s) for lbl, s, t in zip(labels, scores, thresholds) if s >= t]
    return [lbl for lbl, _ in sorted(hits, key=lambda x: x[1], reverse=True)]


def available_models(models_dir: Optional[Path] = None) -> list[str]:
    """Architectures that have a trained checkpoint on disk."""
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    return [arch for arch in config.ARCHITECTURES if (models_dir / arch / "best_model.keras").is_file()]


class Predictor:
    """Wraps a trained model plus its validated thresholds."""

    def __init__(self, model: Any, thresholds: Optional[Sequence[float]] = None,
                 labels: Sequence[str] = config.LABELS, metadata: Optional[dict] = None):
        self.model = model
        self.labels = list(labels)
        self.metadata = metadata or {}
        self.thresholds_validated = thresholds is not None
        self.thresholds = list(thresholds) if thresholds is not None else [0.5] * len(self.labels)
        shape = getattr(model, "input_shape", None)
        if not shape or shape[1] is None:
            raise PredictionError("Model must declare a fixed input shape (None, H, W, 3).")
        self.image_size = int(shape[1])
        # "sigmoid" -> use outputs as-is; "linear" -> outputs are logits, apply sigmoid once.
        self.activation = output_activation(model)
        if self.activation not in (None, "sigmoid", "linear"):
            raise PredictionError(f"Unsupported output activation '{self.activation}' for multi-label scoring.")
        self.outputs_are_logits = self.activation == "linear"

    @classmethod
    def from_model_dir(cls, model_dir: Path, checkpoint: str = "best") -> "Predictor":
        from src.model import ModelNotFoundError, load_trained_model

        model_dir = Path(model_dir)
        path = model_dir / f"{checkpoint}_model.keras"
        if not path.is_file():
            raise ModelNotFoundError(
                f"No trained checkpoint at {path}. Train one with: "
                f"python scripts/train_model.py --model {model_dir.name} --mode development"
            )
        model = load_trained_model(path)
        meta = load_json_if_exists(model_dir / "model_config.json") or {}
        labels = meta.get("labels", config.LABELS)
        if list(labels) != list(config.LABELS):
            raise PredictionError("Model label order does not match this application's label mapping.")
        thr_cfg = load_json_if_exists(model_dir / "threshold_config.json")
        thresholds = [float(thr_cfg["thresholds"][lbl]) for lbl in labels] if thr_cfg else None
        meta = {**meta, "checkpoint_path": str(path), "model_size_mb": file_size_mb(path),
                "threshold_config": thr_cfg}
        return cls(model, thresholds, labels, meta)

    def predict_scores(self, batch: np.ndarray) -> np.ndarray:
        out = self.model(batch, training=False) if callable(self.model) else self.model.predict(batch, verbose=0)
        scores = np.asarray(out, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != len(self.labels):
            raise PredictionError(f"Expected model output (N, {len(self.labels)}), got {scores.shape}")
        if not np.all(np.isfinite(scores)):
            raise PredictionError("Model produced non-finite scores.")
        if self.outputs_are_logits:
            scores = _sigmoid(scores)
        elif np.any((scores < -1e-6) | (scores > 1 + 1e-6)):
            # A sigmoid model can't produce these; refuse rather than guess (e.g. wrongly assume logits).
            raise PredictionError("Model scores fall outside [0, 1]; cannot interpret them as sigmoid outputs.")
        return scores

    def predict(self, image: Image.Image | bytes, filename: Optional[str] = None,
                thresholds: Optional[Sequence[float]] = None) -> PredictionResult:
        img = image if isinstance(image, Image.Image) else load_image(image, filename=filename)
        batch = prepare_batch(img, self.image_size)
        start = time.perf_counter()
        scores = self.predict_scores(batch)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        thr = list(thresholds) if thresholds is not None else self.thresholds
        flagged = flag_findings(scores, thr, self.labels)
        notes = []
        if not self.thresholds_validated and thresholds is None:
            notes.append("No validated threshold file found; using 0.5 for every finding.")
        return PredictionResult(
            scores={lbl: float(s) for lbl, s in zip(self.labels, scores)},
            thresholds={lbl: float(t) for lbl, t in zip(self.labels, thr)},
            flagged=flagged,
            no_finding_predicted=len(flagged) == 0,
            inference_ms=round(elapsed_ms, 1),
            original_size=img.size,
            model_input_size=self.image_size,
            thresholds_validated=self.thresholds_validated and thresholds is None,
            notes=notes,
        )

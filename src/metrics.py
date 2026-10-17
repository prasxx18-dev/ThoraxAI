"""Multi-label metrics and validation-only threshold selection (NumPy/scikit-learn).

Averaging conventions
---------------------
* Per-label metrics treat each finding as an independent binary problem.
* Macro average = unweighted mean over labels where the metric is defined
  (labels with no positives in the evaluated split have undefined AUC/AP and
  are excluded from macro AUC/AP; the count of excluded labels is reported).
* Micro average = metric computed after flattening all (image, label) pairs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import config


def _safe_div(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
    }


def counts_to_metrics(c: dict[str, int]) -> dict[str, float | None]:
    precision = _safe_div(c["tp"], c["tp"] + c["fp"])
    recall = _safe_div(c["tp"], c["tp"] + c["fn"])
    specificity = _safe_div(c["tn"], c["tn"] + c["fp"])
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision is not None or recall is not None:
        f1 = 0.0
    accuracy = _safe_div(c["tp"] + c["tn"], sum(c.values()))
    return {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "accuracy": accuracy}


def apply_thresholds(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64)
    if scores.shape[-1] != thr.shape[0]:
        raise ValueError(f"Got {thr.shape[0]} thresholds for {scores.shape[-1]} outputs")
    return (scores >= thr).astype(np.int8)


def _auc_ap(y: np.ndarray, s: np.ndarray) -> tuple[float | None, float | None]:
    if y.min() == y.max():  # only one class present -> undefined
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Sequence[float],
    labels: Sequence[str] = config.LABELS,
) -> dict:
    """Per-label, macro and micro metrics at the given thresholds."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != len(labels):
        raise ValueError(f"Expected (N, {len(labels)}) arrays, got {y_true.shape}")
    y_pred = apply_thresholds(y_score, thresholds)

    per_label = {}
    for j, label in enumerate(labels):
        counts = binary_counts(y_true[:, j], y_pred[:, j])
        m = counts_to_metrics(counts)
        auc, ap = _auc_ap(y_true[:, j], y_score[:, j])
        per_label[label] = {
            **m,
            "roc_auc": auc,
            "average_precision": ap,
            "threshold": float(thresholds[j]),
            "support": int(y_true[:, j].sum()),
            "confusion": counts,
        }

    def _macro(key: str) -> float | None:
        values = [v[key] for v in per_label.values() if v[key] is not None]
        return float(np.mean(values)) if values else None

    micro_counts = binary_counts(y_true.ravel(), y_pred.ravel())
    micro = counts_to_metrics(micro_counts)
    micro_auc, micro_ap = _auc_ap(y_true.ravel(), y_score.ravel())
    undefined = [lbl for lbl, v in per_label.items() if v["roc_auc"] is None]

    return {
        "num_samples": int(y_true.shape[0]),
        "per_label": per_label,
        "macro": {k: _macro(k) for k in ("precision", "recall", "specificity", "f1", "roc_auc", "average_precision")},
        "micro": {
            "precision": micro["precision"],
            "recall": micro["recall"],
            "specificity": micro["specificity"],
            "f1": micro["f1"],
            "roc_auc": micro_auc,
            "average_precision": micro_ap,
        },
        "labels_with_undefined_auc": undefined,
        "exact_match_ratio": float(np.mean(np.all(y_pred == y_true, axis=1))),
        "hamming_loss": float(np.mean(y_pred != y_true)),
    }


def select_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "f1",
    default: float = 0.5,
) -> tuple[list[float], dict]:
    """Choose one threshold per label using VALIDATION data only.

    ``method="f1"`` maximises F1 on the precision-recall curve;
    ``method="youden"`` maximises sensitivity + specificity - 1 on the ROC curve.
    Labels with no positives (or no negatives) keep ``default``.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    thresholds: list[float] = []
    notes: dict[str, str] = {}
    for j, label in enumerate(config.LABELS[: y_true.shape[1]]):
        y, s = y_true[:, j], y_score[:, j]
        if y.min() == y.max():
            thresholds.append(default)
            notes[label] = "default (single class in validation split)"
            continue
        if method == "f1":
            precision, recall, thr = precision_recall_curve(y, s)
            f1 = np.where((precision + recall) > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
            best = int(np.argmax(f1[:-1])) if len(thr) else 0
            thresholds.append(float(thr[best]) if len(thr) else default)
            notes[label] = f"max F1={f1[best]:.3f} on validation"
        elif method == "youden":
            fpr, tpr, thr = roc_curve(y, s)
            j_stat = tpr - fpr
            best = int(np.argmax(j_stat))
            value = float(thr[best]) if np.isfinite(thr[best]) else default
            thresholds.append(float(np.clip(value, 0.0, 1.0)))
            notes[label] = f"max Youden J={j_stat[best]:.3f} on validation"
        else:
            raise ValueError(f"Unknown threshold method '{method}'")
    return thresholds, notes


def curve_data(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, dict]:
    """ROC and PR curve points per label (for plotting)."""
    out = {}
    for j, label in enumerate(config.LABELS[: y_true.shape[1]]):
        y, s = y_true[:, j], y_score[:, j]
        if y.min() == y.max():
            continue
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        out[label] = {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}
    return out

"""Metric calculations, threshold selection and missing evaluation files."""

import numpy as np
import pytest

import config
from src.metrics import apply_thresholds, binary_counts, compute_multilabel_metrics, counts_to_metrics, select_thresholds


def test_binary_counts_and_metrics():
    c = binary_counts([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    m = counts_to_metrics(c)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)


def test_undefined_metrics_are_none_not_zero():
    m = counts_to_metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 5})
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None


def test_apply_thresholds():
    out = apply_thresholds(np.array([[0.2, 0.8], [0.6, 0.4]]), [0.5, 0.5])
    assert out.tolist() == [[0, 1], [1, 0]]
    with pytest.raises(ValueError):
        apply_thresholds(np.zeros((1, 3)), [0.5])


def test_multilabel_metrics_perfect_and_undefined():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 14)) < 0.3).astype(int)
    y[:, 13] = 0  # label with no positives -> undefined AUC
    scores = y * 0.9 + 0.05
    res = compute_multilabel_metrics(y, scores, [0.5] * 14)
    assert res["per_label"]["Atelectasis"]["roc_auc"] == pytest.approx(1.0)
    assert res["per_label"]["Atelectasis"]["f1"] == pytest.approx(1.0)
    assert res["per_label"]["Hernia"]["roc_auc"] is None
    assert res["labels_with_undefined_auc"] == ["Hernia"]
    assert res["macro"]["roc_auc"] == pytest.approx(1.0)  # undefined label excluded
    assert res["micro"]["f1"] == pytest.approx(1.0)
    assert res["exact_match_ratio"] == pytest.approx(1.0)


def test_macro_vs_micro_differ_under_imbalance():
    y = np.zeros((100, 14), dtype=int)
    y[:50, 0] = 1
    y[:2, 1] = 1
    scores = np.zeros((100, 14))
    scores[:50, 0] = 0.9  # common label perfect
    scores[2:4, 1] = 0.9  # rare label all wrong
    res = compute_multilabel_metrics(y, scores, [0.5] * 14)
    assert res["micro"]["f1"] > res["macro"]["f1"]


def test_shape_validation():
    with pytest.raises(ValueError):
        compute_multilabel_metrics(np.zeros((5, 14)), np.zeros((5, 13)), [0.5] * 14)


@pytest.mark.parametrize("method", ["f1", "youden"])
def test_threshold_selection(method):
    rng = np.random.default_rng(1)
    y = (rng.random((300, 14)) < 0.2).astype(int)
    scores = np.clip(y * 0.4 + rng.random((300, 14)) * 0.5, 0, 1)
    y[:, 5] = 0
    thresholds, notes = select_thresholds(y, scores, method=method)
    assert len(thresholds) == 14 and all(0 <= t <= 1 for t in thresholds)
    assert thresholds[5] == 0.5 and "default" in notes[config.LABELS[5]]
    with pytest.raises(ValueError):
        select_thresholds(y, scores, method="magic")


def test_missing_evaluation_files(tmp_path):
    from src.evaluate import EvaluationError, compare_models, load_evaluation, load_thresholds

    assert load_evaluation("densenet121", tmp_path) is None
    assert compare_models(tmp_path) is None
    with pytest.raises(EvaluationError):
        load_thresholds(tmp_path / "densenet121")

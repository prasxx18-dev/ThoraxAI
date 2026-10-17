"""Held-out test evaluation using the split recorded at training time and the
thresholds selected on validation data. The test set is used once, here, and
never for tuning."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.dataset import DatasetError, build_image_index, discover_dataset, parse_finding_labels, require_dataset
from src.dataset import encode_labels
from src.utils import file_size_mb, get_logger, load_json, load_json_if_exists, save_json, utc_timestamp

log = get_logger("evaluate")


class EvaluationError(RuntimeError):
    pass


def load_thresholds(model_dir: Path) -> tuple[list[float], dict]:
    data = load_json_if_exists(Path(model_dir) / "threshold_config.json")
    if not data:
        raise EvaluationError(f"threshold_config.json missing in {model_dir}; re-run training.")
    return [float(data["thresholds"][lbl]) for lbl in config.LABELS], data


def evaluate_model(
    architecture: str = "densenet121",
    checkpoint: str = "best",
    split: str = "test",
    batch_size: int = 32,
    data_root: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    image_root: Optional[Path] = None,
    models_dir: Path = config.MODELS_DIR,
    results_dir: Path = config.RESULTS_DIR,
) -> dict:
    from src.metrics import compute_multilabel_metrics, curve_data
    from src.model import load_trained_model
    from src.plots import plot_confusion_matrices, plot_curves, plot_history, plot_per_label_bars
    from src.preprocessing import make_tf_dataset

    model_dir = Path(models_dir) / architecture
    run_results = Path(results_dir) / architecture
    model_path = model_dir / f"{checkpoint}_model.keras"
    if not model_path.is_file():
        from src.model import ModelNotFoundError

        raise ModelNotFoundError(
            f"No trained checkpoint at {model_path}. Train first: "
            f"python scripts/train_model.py --model {architecture} --mode development"
        )
    model_config = load_json_if_exists(model_dir / "model_config.json") or {}
    thresholds, threshold_cfg = load_thresholds(model_dir)

    split_file = run_results / "splits.csv"
    if not split_file.is_file():
        raise EvaluationError(f"{split_file} not found. Train the model first so the held-out split is recorded.")
    splits = pd.read_csv(split_file)
    part = splits[splits["split"] == split].copy()
    if part.empty:
        raise EvaluationError(f"No images assigned to split '{split}'.")

    paths = discover_dataset(data_root, metadata_csv, image_root)
    require_dataset(paths)
    index, _ = build_image_index(paths.image_root)
    part["path"] = part["image"].map(lambda n: str(index[n]) if n in index else None)
    missing = int(part["path"].isna().sum())
    if missing == len(part):
        raise DatasetError("None of the recorded test images were found on disk.")
    if missing:
        log.warning("%d test images missing on disk are excluded.", missing)
    part = part[part["path"].notna()]
    part["labels"] = part["finding_labels"].map(lambda v: parse_finding_labels(v)[0])
    y_true = encode_labels(part["labels"])

    model = load_trained_model(model_path)
    image_size = int(model.input_shape[1])
    ds = make_tf_dataset(part["path"].tolist(), y_true, image_size, batch_size, training=False)

    log.info("Evaluating %s (%s checkpoint) on %d %s images", architecture, checkpoint, len(part), split)
    start = time.perf_counter()
    scores = np.asarray(model.predict(ds, verbose=0), dtype=np.float64)
    batch_seconds = time.perf_counter() - start

    # Single-image latency (what a user of the app experiences), excluding warm-up.
    sample = next(iter(ds.unbatch().batch(1).take(1)))[0]
    model.predict(sample, verbose=0)
    t0 = time.perf_counter()
    for _ in range(5):
        model.predict(sample, verbose=0)
    single_ms = (time.perf_counter() - t0) / 5 * 1000

    metrics = compute_multilabel_metrics(y_true, scores, thresholds)
    result = {
        "generated_at": utc_timestamp(),
        "architecture": architecture,
        "checkpoint": model_path.name,
        "split": split,
        "mode": model_config.get("mode"),
        "mode_note": model_config.get("mode_note"),
        "threshold_method": threshold_cfg.get("method"),
        "thresholds_selected_on": threshold_cfg.get("selected_on"),
        "missing_images_excluded": missing,
        "model_size_mb": file_size_mb(model_path),
        "inference": {
            "throughput_ms_per_image": round(batch_seconds / len(part) * 1000, 2),
            "single_image_latency_ms": round(single_ms, 2),
            "batch_size": batch_size,
        },
        **metrics,
    }
    run_results.mkdir(parents=True, exist_ok=True)
    save_json(result, run_results / f"{split}_metrics.json")

    rows = []
    for label, m in metrics["per_label"].items():
        rows.append({"label": config.DISPLAY_NAMES[label], "support": m["support"], "threshold": m["threshold"],
                     "precision": m["precision"], "recall": m["recall"], "specificity": m["specificity"],
                     "f1": m["f1"], "roc_auc": m["roc_auc"], "average_precision": m["average_precision"],
                     **{k.upper(): v for k, v in m["confusion"].items()}})
    pd.DataFrame(rows).to_csv(run_results / ("per_label_metrics.csv" if split == "test" else f"per_label_metrics_{split}.csv"),
                              index=False)
    pred_df = pd.DataFrame(scores, columns=[f"score_{lbl}" for lbl in config.LABELS])
    pred_df.insert(0, "image", part["image"].to_numpy())
    pred_df.to_csv(run_results / f"{split}_predictions.csv", index=False)

    plot_confusion_matrices(metrics["per_label"], run_results / "confusion_matrices")
    plot_curves(curve_data(y_true, scores), metrics["per_label"], run_results / "roc_curves",
                run_results / "precision_recall_curves")
    plot_per_label_bars(metrics["per_label"], run_results / "per_label_performance.png")
    history = load_json_if_exists(run_results / "training_history.json")
    if history and history.get("history"):
        plot_history(history["history"], run_results / "training_curves.png")

    compare_models(Path(results_dir))
    log.info("Macro ROC-AUC: %s | Macro F1: %s", metrics["macro"]["roc_auc"], metrics["macro"]["f1"])
    return result


def compare_models(results_dir: Path = config.RESULTS_DIR) -> Optional[Path]:
    """Write results/model_comparison.csv from every available test_metrics.json."""
    rows = []
    for arch in config.ARCHITECTURES:
        data = load_json_if_exists(Path(results_dir) / arch / "test_metrics.json")
        if not data:
            continue
        rows.append({
            "model": config.ARCHITECTURES[arch]["display_name"],
            "mode": data.get("mode"),
            "test_images": data.get("num_samples"),
            "macro_roc_auc": data["macro"]["roc_auc"],
            "macro_pr_auc": data["macro"]["average_precision"],
            "macro_f1": data["macro"]["f1"],
            "micro_f1": data["micro"]["f1"],
            "macro_recall": data["macro"]["recall"],
            "model_size_mb": data.get("model_size_mb"),
            "latency_ms": data.get("inference", {}).get("single_image_latency_ms"),
        })
    if not rows:
        return None
    path = Path(results_dir) / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_evaluation(architecture: str, results_dir: Path = config.RESULTS_DIR) -> Optional[dict]:
    """Return saved test metrics or None (used by the app's empty states)."""
    return load_json_if_exists(Path(results_dir) / architecture / "test_metrics.json")


__all__ = ["evaluate_model", "compare_models", "load_evaluation", "load_thresholds", "EvaluationError", "load_json"]

"""Training pipeline: data preparation, two-stage transfer learning, checkpointing,
validation-only threshold selection and artifact/metadata recording.

Stage 1 trains only the new classification head (backbone frozen).
Stage 2 (optional, ``fine_tune_epochs > 0``) unfreezes the last
``fine_tune_layers`` backbone layers and continues with a much lower LR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.dataset import (
    DatasetError,
    assign_splits,
    attach_image_paths,
    build_image_index,
    check_images,
    development_subset,
    discover_dataset,
    encode_labels,
    load_metadata,
    patient_overlap,
    read_split_list,
    require_dataset,
    split_summary,
    survey_image_properties,
)
from src.utils import environment_info, get_logger, save_json, set_seeds, utc_timestamp

log = get_logger("train")


@dataclass
class PreparedData:
    df: pd.DataFrame  # rows used for this run, with "split" and "path"
    split_method: str
    excluded: dict
    image_properties: dict


def prepare_data(
    cfg: config.TrainConfig,
    data_root: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    image_root: Optional[Path] = None,
) -> PreparedData:
    """Load metadata, resolve images, create patient-level splits, apply dev subset."""
    paths = discover_dataset(data_root, metadata_csv, image_root)
    require_dataset(paths)
    log.info("Dataset: csv=%s images=%s", paths.metadata_csv, paths.image_root)

    df = load_metadata(paths.metadata_csv)
    index, duplicates = build_image_index(paths.image_root)
    df = attach_image_paths(df, index)
    missing = int((~df["exists"]).sum())
    missing_fraction = missing / max(1, len(df))
    log.info("Metadata rows: %d | images found: %d | missing: %d (%.2f%%)", len(df), len(index), missing,
             100 * missing_fraction)
    if missing_fraction > cfg.max_missing_fraction:
        raise DatasetError(
            f"{missing} of {len(df)} metadata images ({missing_fraction:.1%}) are missing on disk, exceeding "
            f"max_missing_fraction={cfg.max_missing_fraction:.1%}. Check the image root or extract all "
            "image archives. (Use --max-missing-fraction to override deliberately, e.g. for a partial download.)"
        )

    df, split_method = assign_splits(
        df,
        read_split_list(paths.train_val_list),
        read_split_list(paths.test_list),
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )
    # Splits are assigned BEFORE dropping missing files so patient grouping is
    # computed on the full metadata and cannot change with partial downloads.
    df = df[df["exists"] & (df["split"] != "unassigned")].copy()

    if cfg.mode == "development":
        df = development_subset(df, cfg.dev_max_images, seed=cfg.seed)
        log.info("Development mode: using a reproducible subset of %d images.", len(df))
    elif cfg.mode != "full":
        raise ValueError("mode must be 'development' or 'full'")

    corrupted: list = []
    if cfg.verify_images:
        log.info("Verifying integrity of %d selected images...", len(df))
        corrupted = check_images(df["path"].tolist())
        if corrupted:
            log.warning("Excluding %d corrupted images.", len(corrupted))
            df = df[~df["path"].isin({p for p, _ in corrupted})].copy()

    overlap = patient_overlap(df)
    if any(overlap.values()):
        raise DatasetError(f"Patient leakage detected between splits: {overlap}")
    for split in ("train", "val"):
        if (df["split"] == split).sum() == 0:
            raise DatasetError(f"The '{split}' split is empty; the dataset subset is too small.")

    props = survey_image_properties(df["path"].tolist(), max_images=200, seed=cfg.seed)
    if props["max_side"] and props["max_side"] < cfg.image_size:
        log.warning("Source images are at most %dpx but image_size=%d: images will be upsampled (no detail gained).",
                    props["max_side"], cfg.image_size)

    return PreparedData(
        df=df.reset_index(drop=True),
        split_method=split_method,
        image_properties=props,
        excluded={
            "missing_on_disk": missing,
            "duplicate_filenames_on_disk": len(duplicates),
            "corrupted": len(corrupted),
            "corrupted_examples": [Path(p).name for p, _ in corrupted[:20]],
        },
    )


def dry_run(
    cfg: config.TrainConfig,
    data_root: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    image_root: Optional[Path] = None,
) -> dict:
    """Check that a training run *could* start, without training or writing artifacts.

    Prepares the splits exactly as ``run_training`` would, decodes one training
    and one validation batch through the real tf.data pipeline, builds the model
    and runs a single forward pass to confirm the (batch, 14) output shape.
    No weights are updated and no scores are reported.
    """
    from src.model import build_model, compute_label_weights
    from src.preprocessing import make_tf_dataset

    set_seeds(cfg.seed)
    data = prepare_data(cfg, data_root, metadata_csv, image_root)
    df = data.df
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    y_train, y_val = encode_labels(train_df["labels"]), encode_labels(val_df["labels"])
    pos_w, _ = compute_label_weights(y_train)

    checks = {}
    for name, part, y, training in (("train", train_df, y_train, True), ("val", val_df, y_val, False)):
        ds = make_tf_dataset(part["path"].tolist(), y, cfg.image_size, cfg.batch_size, training=training,
                             augment=cfg.augment, seed=cfg.seed)
        xb, yb = next(iter(ds))
        checks[name] = {
            "batch_images": list(xb.shape),
            "batch_labels": list(yb.shape),
            "pixel_range": [float(xb.numpy().min()), float(xb.numpy().max())],
        }

    model, info = build_model(cfg.architecture, cfg.image_size, pretrained=cfg.pretrained, dropout=cfg.dropout)
    out = model(xb, training=False)
    return {
        "split_method": data.split_method,
        "splits": split_summary(df),
        "excluded": data.excluded,
        "image_properties": data.image_properties,
        "positive_label_weights": dict(zip(config.LABELS, (round(w, 2) for w in pos_w))),
        "pipeline_checks": checks,
        "model": {**info, "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                  "total_params": int(model.count_params()), "output_shape": list(out.shape)},
    }


def _history_to_dict(histories: list) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def predict_in_batches(model, dataset) -> np.ndarray:
    return np.asarray(model.predict(dataset, verbose=0), dtype=np.float64)


def run_training(
    cfg: config.TrainConfig,
    data_root: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    image_root: Optional[Path] = None,
    models_dir: Path = config.MODELS_DIR,
    results_dir: Path = config.RESULTS_DIR,
) -> dict:
    """Train a model and write all artifacts. Returns a summary dict."""
    import keras

    from src.metrics import compute_multilabel_metrics, select_thresholds
    from src.model import build_model, compile_model, compute_label_weights, load_trained_model, unfreeze_backbone
    from src.preprocessing import make_tf_dataset

    set_seeds(cfg.seed)
    keras.utils.set_random_seed(cfg.seed)
    started = time.time()
    model_dir = Path(models_dir) / cfg.architecture
    run_results = Path(results_dir) / cfg.architecture
    model_dir.mkdir(parents=True, exist_ok=True)
    run_results.mkdir(parents=True, exist_ok=True)

    data = prepare_data(cfg, data_root, metadata_csv, image_root)
    df = data.df
    splits = split_summary(df)
    for name, info in splits.items():
        log.info("Split %-5s: %6d images, %5d patients", name, info["images"], info["patients"])

    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    y_train, y_val = encode_labels(train_df["labels"]), encode_labels(val_df["labels"])
    label_weights = compute_label_weights(y_train)

    # Record the exact split so evaluation uses the same held-out test images.
    split_file = run_results / "splits.csv"
    df[["image", "split", "finding_labels"]].to_csv(split_file, index=False)

    train_ds = make_tf_dataset(train_df["path"].tolist(), y_train, cfg.image_size, cfg.batch_size, training=True,
                               augment=cfg.augment, seed=cfg.seed, cache_dir=cfg.cache_dir)
    val_ds = make_tf_dataset(val_df["path"].tolist(), y_val, cfg.image_size, cfg.batch_size, training=False)

    model, model_info = build_model(cfg.architecture, cfg.image_size, pretrained=cfg.pretrained, dropout=cfg.dropout)
    best_path, last_path = model_dir / "best_model.keras", model_dir / "last_model.keras"
    if cfg.resume and last_path.is_file():
        log.info("Resuming from %s", last_path)
        model.load_weights(last_path)
    compile_model(model, cfg.learning_rate, cfg.loss, label_weights, cfg.weight_decay)

    def _callbacks(initial_best: Optional[float] = None, append_log: bool = False):
        return [
            keras.callbacks.ModelCheckpoint(best_path, monitor="val_auc", mode="max", save_best_only=True,
                                            initial_value_threshold=initial_best, verbose=1),
            keras.callbacks.ModelCheckpoint(last_path, save_best_only=False),
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=cfg.early_stopping_patience,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1),
            keras.callbacks.CSVLogger(str(run_results / "training_log.csv"), append=append_log or cfg.resume),
        ]

    log.info("Stage 1: training classification head for up to %d epochs", cfg.epochs)
    histories = [model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=_callbacks(), verbose=2)]
    stage1_epochs = len(histories[0].history.get("loss", []))

    if cfg.fine_tune_epochs > 0:
        best_so_far = max(histories[0].history.get("val_auc", [0.0]))
        n = unfreeze_backbone(model, cfg.fine_tune_layers)
        log.info("Stage 2: fine-tuning %d backbone layers for up to %d epochs", n, cfg.fine_tune_epochs)
        compile_model(model, cfg.fine_tune_learning_rate, cfg.loss, label_weights, cfg.weight_decay)
        histories.append(
            model.fit(train_ds, validation_data=val_ds, initial_epoch=stage1_epochs,
                      epochs=stage1_epochs + cfg.fine_tune_epochs,
                      callbacks=_callbacks(best_so_far, append_log=True), verbose=2)
        )

    final_path = model_dir / "final_model.keras"
    model.save(final_path)
    training_seconds = time.time() - started

    # Threshold selection on the VALIDATION split, using the best checkpoint.
    best_model = load_trained_model(best_path) if best_path.is_file() else model
    val_scores = predict_in_batches(best_model, val_ds)
    thresholds, notes = select_thresholds(y_val, val_scores, method=cfg.threshold_method)
    val_metrics = compute_multilabel_metrics(y_val, val_scores, thresholds)

    history = _history_to_dict(histories)
    labels_mapping = {
        "labels": list(config.LABELS),
        "display_names": [config.DISPLAY_NAMES[lbl] for lbl in config.LABELS],
        "index": {lbl: i for i, lbl in enumerate(config.LABELS)},
        "no_finding_policy": "Not an output. 'No Finding predicted' = no score exceeds its threshold.",
    }
    model_config = {
        "created_at": utc_timestamp(),
        **model_info,
        "labels": list(config.LABELS),
        "input": {"image_size": cfg.image_size, "channels": 3, "value_range": "[0, 255] RGB (normalisation inside model)"},
        "train_config": cfg.to_dict(),
        "mode": cfg.mode,
        "mode_note": (
            "DEVELOPMENT SUBSET - results do not represent full-dataset performance."
            if cfg.mode == "development" else "Full available dataset."
        ),
        "split_method": data.split_method,
        "split_sizes": {k: {"images": v["images"], "patients": v["patients"]} for k, v in splits.items()},
        "excluded_images": data.excluded,
        "source_image_properties": data.image_properties,
        "label_weights": {"pos": label_weights[0], "neg": label_weights[1]} if cfg.loss == "weighted_bce" else None,
        "epochs_completed": len(history.get("loss", [])),
        "stage1_epochs_completed": stage1_epochs,
        "best_val_auc": max(history.get("val_auc", [float("nan")])),
        "training_seconds": round(training_seconds, 1),
        "checkpoints": {"best": best_path.name, "final": final_path.name, "last": last_path.name},
        "environment": environment_info(),
    }
    threshold_config = {
        "method": cfg.threshold_method,
        "selected_on": "validation split only",
        "validation_images": int(len(val_df)),
        "thresholds": {lbl: float(t) for lbl, t in zip(config.LABELS, thresholds)},
        "notes": notes,
        "score_interpretation": "Uncalibrated sigmoid model scores, not calibrated probabilities.",
    }

    save_json(model_config, model_dir / "model_config.json")
    save_json(labels_mapping, model_dir / "label_mapping.json")
    save_json(threshold_config, model_dir / "threshold_config.json")
    save_json({"history": history, "mode": cfg.mode}, run_results / "training_history.json")
    save_json({"split": "validation", **val_metrics}, run_results / "validation_metrics.json")
    save_json({"split_summary": splits, "split_method": data.split_method}, run_results / "split_summary.json")

    log.info("Training finished in %.1f min. Best val AUC: %s", training_seconds / 60, model_config["best_val_auc"])
    return {"model_dir": str(model_dir), "results_dir": str(run_results), **model_config}

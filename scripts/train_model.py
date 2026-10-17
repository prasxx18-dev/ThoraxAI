"""Train a ThoraxAI multi-label model.

Examples:
    # Quick, CPU-friendly development run on a 3,000-image patient-level subset
    python scripts/train_model.py --model densenet121 --mode development

    # Full dataset with fine-tuning (GPU strongly recommended)
    python scripts/train_model.py --model densenet121 --mode full --epochs 8 --fine-tune-epochs 5 --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.dataset import DatasetError  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("train_cli")


def build_parser() -> argparse.ArgumentParser:
    d = config.TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(config.ARCHITECTURES), default=d.architecture)
    p.add_argument("--mode", choices=["development", "full"], default=d.mode)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs, help="Stage 1 (head) epochs")
    p.add_argument("--fine-tune-epochs", type=int, default=d.fine_tune_epochs, help="Stage 2 epochs (0 = off)")
    p.add_argument("--fine-tune-layers", type=int, default=d.fine_tune_layers)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--fine-tune-learning-rate", type=float, default=d.fine_tune_learning_rate)
    p.add_argument("--loss", choices=["weighted_bce", "bce"], default=d.loss)
    p.add_argument("--dev-max-images", type=int, default=d.dev_max_images)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--threshold-method", choices=["f1", "youden"], default=d.threshold_method)
    p.add_argument("--max-missing-fraction", type=float, default=d.max_missing_fraction)
    p.add_argument("--patience", type=int, default=d.early_stopping_patience)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--no-pretrained", action="store_true", help="Random init (not recommended)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--skip-image-check", action="store_true", help="Skip integrity check of selected images")
    p.add_argument("--cache-dir", type=str, default=None, help="Optional on-disk tf.data cache directory")
    p.add_argument("--resume", action="store_true", help="Initialise from models/<model>/last_model.keras")
    p.add_argument("--yes", action="store_true", help="Do not ask for confirmation in full mode")
    p.add_argument("--dry-run", action="store_true",
                   help="Check data, splits, input pipeline and model build without training or saving anything")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.TrainConfig(
        architecture=args.model, mode=args.mode, image_size=args.image_size, batch_size=args.batch_size,
        epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs, fine_tune_layers=args.fine_tune_layers,
        learning_rate=args.learning_rate, fine_tune_learning_rate=args.fine_tune_learning_rate, loss=args.loss,
        dev_max_images=args.dev_max_images, val_fraction=args.val_fraction, threshold_method=args.threshold_method,
        max_missing_fraction=args.max_missing_fraction, early_stopping_patience=args.patience, seed=args.seed,
        pretrained=not args.no_pretrained, augment=not args.no_augment, verify_images=not args.skip_image_check,
        cache_dir=args.cache_dir, resume=args.resume,
    )
    print("Training configuration:")
    for k, v in cfg.to_dict().items():
        print(f"  {k:24s} {v}")

    if args.dry_run:
        from src.train import dry_run

        try:
            report = dry_run(cfg, args.data_root, args.metadata_csv, args.image_root)
        except DatasetError as exc:
            log.error("%s", exc)
            return 2
        print("\n=== DRY RUN: nothing was trained and no files were written ===")
        print(f"Split method: {report['split_method']}")
        for split, info in report["splits"].items():
            print(f"  {split:5s}: {info['images']:6d} images / {info['patients']:5d} patients")
        print(f"Excluded: {report['excluded']}")
        props = report["image_properties"]
        print(f"Source image sizes (sample of {props['sampled']}): {props['sizes']}")
        for name, c in report["pipeline_checks"].items():
            print(f"  {name} batch: images {c['batch_images']}, labels {c['batch_labels']}, "
                  f"pixel range {c['pixel_range'][0]:.0f}-{c['pixel_range'][1]:.0f}")
        m = report["model"]
        print(f"Model: {m['architecture']} | ImageNet weights loaded: {m['pretrained_weights_loaded']} | "
              f"output {m['output_shape']} | trainable {m['trainable_params']:,} / total {m['total_params']:,}")
        print("Positive-class loss weights:", report["positive_label_weights"])
        print("\nDry run OK. Remove --dry-run to start training.")
        return 0

    if cfg.mode == "full" and not args.yes:
        print(
            "\nFULL MODE: ~112k images (~78k training images with the official split)."
            "\nMeasured on a 12-thread laptop CPU: ~72 ms per image for a frozen-backbone DenseNet121 step,"
            "\ni.e. roughly 1.5-3 hours per epoch including image decoding; fine-tuning is slower."
            "\nA CUDA GPU (Linux/WSL2) is strongly recommended. Disk reads ~42 GB per epoch."
        )
        if input("Continue? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 1

    from src.train import run_training

    try:
        summary = run_training(cfg, args.data_root, args.metadata_csv, args.image_root)
    except DatasetError as exc:
        log.error("%s", exc)
        return 2
    print("\nArtifacts written to:", summary["model_dir"], "and", summary["results_dir"])
    print("Best validation macro AUC:", summary["best_val_auc"])
    print(f"Next step: python scripts/evaluate_model.py --model {cfg.architecture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

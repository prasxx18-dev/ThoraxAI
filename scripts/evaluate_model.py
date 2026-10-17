"""Evaluate a trained ThoraxAI model on the held-out test split recorded at training time.

Usage:
    python scripts/evaluate_model.py --model densenet121
    python scripts/evaluate_model.py --model mobilenetv2 --checkpoint final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.dataset import DatasetError  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("evaluate_cli")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(config.ARCHITECTURES), default="densenet121")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--image-root", type=Path, default=None)
    args = p.parse_args(argv)

    from src.evaluate import EvaluationError, evaluate_model
    from src.model import ModelNotFoundError

    try:
        result = evaluate_model(args.model, args.checkpoint, "test", args.batch_size,
                                args.data_root, args.metadata_csv, args.image_root)
    except (DatasetError, EvaluationError, ModelNotFoundError) as exc:
        log.error("%s", exc)
        return 2

    print(f"\n=== Test results: {args.model} ({result['mode']}) on {result['num_samples']} images ===")
    if result.get("mode_note"):
        print("NOTE:", result["mode_note"])
    print(f"{'Finding':20s} {'AUC':>6s} {'AP':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Supp':>6s}")
    fmt = lambda v: f"{v:6.3f}" if v is not None else "   n/a"  # noqa: E731
    for label, m in result["per_label"].items():
        print(f"{config.DISPLAY_NAMES[label]:20s} {fmt(m['roc_auc'])} {fmt(m['average_precision'])} "
              f"{fmt(m['precision'])} {fmt(m['recall'])} {fmt(m['f1'])} {m['support']:6d}")
    print(f"{'MACRO':20s} {fmt(result['macro']['roc_auc'])} {fmt(result['macro']['average_precision'])} "
          f"{fmt(result['macro']['precision'])} {fmt(result['macro']['recall'])} {fmt(result['macro']['f1'])}")
    print(f"Model size: {result['model_size_mb']} MB | latency: {result['inference']['single_image_latency_ms']} ms")
    print(f"Artifacts: {config.RESULTS_DIR / args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

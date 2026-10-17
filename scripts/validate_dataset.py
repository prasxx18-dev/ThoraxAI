"""Validate the locally extracted NIH ChestX-ray14 dataset and write a report.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data-root D:/datasets/nih --verify all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.dataset import DatasetError, discover_dataset, save_validation_report, validate_dataset  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("validate")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help=f"Dataset root (default: {config.DATA_DIR})")
    p.add_argument("--metadata-csv", type=Path, default=None, help="Explicit path to Data_Entry_2017*.csv")
    p.add_argument("--image-root", type=Path, default=None, help="Folder searched recursively for images")
    p.add_argument("--verify", choices=["none", "sample", "all"], default="sample",
                   help="Image integrity check: none, a random sample, or every image (slow)")
    p.add_argument("--sample-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = p.parse_args(argv)

    paths = discover_dataset(args.data_root, args.metadata_csv, args.image_root)
    log.info("Discovered: %s", paths.to_dict())
    try:
        report, _ = validate_dataset(paths, verify=args.verify, sample_size=args.sample_size, seed=args.seed)
    except DatasetError as exc:
        log.error("%s", exc)
        return 2

    files = save_validation_report(report, args.out_dir)
    from src.plots import plot_dataset_overview

    for i, fig_path in enumerate(plot_dataset_overview(report, args.out_dir / "dataset_figures")):
        files[f"figure_{i}"] = fig_path
    print("\n=== ThoraxAI dataset validation ===")
    for key in ("metadata_rows", "unique_images_in_metadata", "image_files_found", "missing_images",
                "duplicate_filenames_on_disk", "patients", "split_method"):
        print(f"{key:30s}: {report[key]}")
    ic = report["integrity_check"]
    print(f"{'integrity check':30s}: {ic['images_checked']} checked ({ic['mode']}), {ic['corrupted']} corrupted")
    props = report["image_properties"]
    print(f"{'image sizes (sample)':30s}: {props['sizes']} from {props['sampled']} images; modes {props['modes']}")
    default_size = config.TrainConfig().image_size
    if props["max_side"] and props["max_side"] < default_size:
        print(f"{'':30s}  WARNING: images are smaller than the default {default_size}px training input.")
    for split, info in report["splits"].items():
        print(f"{split + ' split':30s}: {info['images']} images / {info['patients']} patients")
    print(f"{'patient overlap':30s}: {report['patient_overlap']}")
    print("\nPositives per label (images present on disk):")
    for label, count in report["labels"]["positives"].items():
        print(f"  {config.DISPLAY_NAMES[label]:20s} {count:7d}  ({report['labels']['prevalence'][label]:.2%})")
    print("\nReports written:", *[str(v) for v in files.values()], sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

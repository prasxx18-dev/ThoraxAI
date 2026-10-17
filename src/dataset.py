"""Dataset discovery, metadata parsing, label encoding, splitting and validation
for the NIH ChestX-ray14 dataset (Kaggle: nih-chest-xrays/data).

Nothing in this module fabricates data: every count comes from the files found
on disk. If the dataset is missing, functions raise ``DatasetError`` with a
message explaining how to configure it.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config
from src.preprocessing import verify_image_file
from src.utils import get_logger, save_json, utc_timestamp

log = get_logger("dataset")

IMAGE_COLUMN = "Image Index"
LABEL_COLUMN = "Finding Labels"
PATIENT_COLUMN = "Patient ID"
OPTIONAL_COLUMNS = {
    "Follow-up #": "follow_up",
    "Patient Age": "age",
    "Patient Gender": "gender",
    "View Position": "view_position",
}
SPLITS = ("train", "val", "test")


class DatasetError(Exception):
    """Raised when the dataset cannot be found or is structurally invalid."""


class MetadataError(DatasetError):
    """Raised when the metadata CSV is missing required columns or is empty."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class DatasetPaths:
    root: Path
    metadata_csv: Optional[Path]
    image_root: Optional[Path]
    train_val_list: Optional[Path]
    test_list: Optional[Path]
    has_images: bool = False  # at least one supported image file exists under image_root

    @property
    def is_ready(self) -> bool:
        """True only if the metadata CSV exists and the image folder really contains images."""
        return self.metadata_csv is not None and self.image_root is not None and self.has_images

    def to_dict(self) -> dict:
        return {k: (v if v is None or isinstance(v, bool) else str(v)) for k, v in asdict(self).items()}


def _search(root: Path, patterns: Iterable[str], max_depth: int = 3) -> Optional[Path]:
    """Breadth-limited search for the first file matching any glob pattern."""
    if not root.is_dir():
        return None
    for depth in range(max_depth + 1):
        prefix = "*/" * depth
        for pattern in patterns:
            matches = sorted(root.glob(prefix + pattern))
            matches = [m for m in matches if m.is_file()]
            if matches:
                return matches[0]
    return None


def _contains_image(folder: Path, max_depth: int = 4) -> bool:
    """Return True as soon as one supported image file is found (bounded, early-exit walk)."""
    base_depth = len(folder.parts)
    exts = config.SUPPORTED_IMAGE_EXTENSIONS
    for dirpath, dirnames, filenames in os.walk(folder):
        if any(name.lower().endswith(exts) for name in filenames):
            return True
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames.clear()
    return False


def discover_dataset(
    root: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    image_root: Optional[Path] = None,
) -> DatasetPaths:
    """Locate the metadata CSV, image folder and official split lists.

    Supports the Kaggle layout (``images_001/images/*.png`` ... ``images_012``)
    as well as a flat ``images/`` folder. Explicit arguments (or the
    THORAXAI_* environment variables) take precedence over discovery.
    """
    root = Path(root) if root else config.DATA_DIR
    metadata_csv = Path(metadata_csv) if metadata_csv else config.METADATA_CSV
    image_root = Path(image_root) if image_root else config.IMAGE_ROOT

    if metadata_csv is None or not metadata_csv.is_file():
        metadata_csv = _search(root, ["Data_Entry_2017*.csv", "Data_Entry*.csv"])
    if image_root is None or not image_root.is_dir():
        image_root = root if root.is_dir() else None

    search_dir = metadata_csv.parent if metadata_csv else root
    train_val_list = _search(search_dir, ["train_val_list*.txt"], max_depth=1)
    test_list = _search(search_dir, ["test_list*.txt"], max_depth=1)
    if train_val_list is None and search_dir != root:
        train_val_list = _search(root, ["train_val_list*.txt"])
    if test_list is None and search_dir != root:
        test_list = _search(root, ["test_list*.txt"])

    has_images = image_root is not None and _contains_image(image_root)
    return DatasetPaths(root, metadata_csv, image_root, train_val_list, test_list, has_images)


def require_dataset(paths: DatasetPaths) -> None:
    """Raise a helpful DatasetError if the dataset is not configured."""
    if paths.metadata_csv is None:
        raise DatasetError(
            f"Metadata CSV (Data_Entry_2017*.csv) not found under '{paths.root}'. "
            "Download the NIH Chest X-ray dataset from https://www.kaggle.com/datasets/nih-chest-xrays/data, "
            "extract it into data/chest_xray/, or set THORAXAI_DATA_DIR / THORAXAI_METADATA_CSV."
        )
    if paths.image_root is None:
        raise DatasetError(
            f"Image folder not found (looked in '{paths.root}'). Set THORAXAI_IMAGE_ROOT or pass --image-root."
        )
    if not paths.has_images:
        raise DatasetError(
            f"No PNG/JPEG images found under '{paths.image_root}' (searched 4 folder levels). "
            "Check that the image archives were extracted, or set image_root in local_config.json."
        )


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
_CANONICAL = {label.lower(): label for label in config.LABELS}
_CANONICAL.update(config.LABEL_ALIASES)


def parse_finding_labels(value: object) -> tuple[list[str], list[str]]:
    """Split a pipe-separated label string into (known_labels, unknown_tokens).

    ``"No Finding"`` yields no labels. Empty/NaN values yield no labels.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [], []
    known: list[str] = []
    unknown: list[str] = []
    for token in str(value).split("|"):
        token = token.strip()
        if not token or token.lower() == config.NO_FINDING_TOKEN.lower():
            continue
        canonical = _CANONICAL.get(token.lower())
        if canonical is None:
            unknown.append(token)
        elif canonical not in known:
            known.append(canonical)
    return known, unknown


def encode_labels(label_lists: Iterable[Iterable[str]]) -> np.ndarray:
    """Multi-hot encode label lists into an (N, 14) float32 matrix."""
    index = {label: i for i, label in enumerate(config.LABELS)}
    rows = list(label_lists)
    matrix = np.zeros((len(rows), config.NUM_LABELS), dtype=np.float32)
    for r, labels in enumerate(rows):
        for label in labels:
            if label not in index:
                raise ValueError(f"Unknown label '{label}'")
            matrix[r, index[label]] = 1.0
    return matrix


def decode_labels(vector: Iterable[float], threshold: float = 0.5) -> list[str]:
    return [label for label, v in zip(config.LABELS, vector) if v >= threshold]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load and normalise the metadata CSV.

    Returns a DataFrame with columns ``image``, ``finding_labels``, ``labels``
    (list of canonical labels), ``patient_id`` and optional demographic
    columns. ``df.attrs`` records unknown label tokens and duplicate rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise MetadataError(f"Metadata CSV not found: {csv_path}")
    try:
        raw = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MetadataError(f"Could not parse metadata CSV {csv_path}: {exc}") from exc

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = {IMAGE_COLUMN, LABEL_COLUMN} - set(raw.columns)
    if missing:
        raise MetadataError(f"Metadata CSV is missing required column(s): {sorted(missing)}")
    if raw.empty:
        raise MetadataError("Metadata CSV contains no rows.")

    df = pd.DataFrame({"image": raw[IMAGE_COLUMN].astype(str).str.strip()})
    df["finding_labels"] = raw[LABEL_COLUMN].fillna("").astype(str)
    parsed = df["finding_labels"].map(parse_finding_labels)
    df["labels"] = parsed.map(lambda p: p[0])
    unknown = sorted({tok for _, toks in parsed for tok in toks})

    df["patient_id"] = raw[PATIENT_COLUMN].astype(str) if PATIENT_COLUMN in raw.columns else None
    for source, target in OPTIONAL_COLUMNS.items():
        if source in raw.columns:
            df[target] = raw[source]

    empty_names = int((df["image"] == "").sum())
    duplicated_rows = int(df["image"].duplicated().sum())
    if duplicated_rows:
        log.warning("%d duplicated image names in metadata; keeping first occurrence.", duplicated_rows)
        df = df.drop_duplicates("image", keep="first").reset_index(drop=True)
    if empty_names:
        df = df[df["image"] != ""].reset_index(drop=True)

    df.attrs.update(
        {
            "source_rows": int(len(raw)),
            "unknown_labels": unknown,
            "duplicate_metadata_rows": duplicated_rows,
            "empty_image_names": empty_names,
            "has_patient_id": PATIENT_COLUMN in raw.columns,
        }
    )
    if unknown:
        log.warning("Unknown label tokens ignored: %s", unknown)
    return df


def build_image_index(image_root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Map image filename -> path for every supported image under ``image_root``.

    Returns ``(index, duplicates)`` where ``duplicates`` lists filenames that
    occur in more than one directory (the first path found is used).
    """
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    exts = config.SUPPORTED_IMAGE_EXTENSIONS
    for dirpath, _dirnames, filenames in os.walk(image_root):
        for name in filenames:
            if not name.lower().endswith(exts):
                continue
            path = Path(dirpath) / name
            if name in index:
                duplicates.setdefault(name, [index[name]]).append(path)
            else:
                index[name] = path
    return index, duplicates


def attach_image_paths(df: pd.DataFrame, index: dict[str, Path]) -> pd.DataFrame:
    df = df.copy()
    df["path"] = df["image"].map(lambda n: str(index[n]) if n in index else None)
    df["exists"] = df["path"].notna()
    return df


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def read_split_list(path: Optional[Path]) -> set[str]:
    if path is None or not Path(path).is_file():
        return set()
    with Path(path).open("r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def _patient_groups(df: pd.DataFrame) -> pd.Series:
    """Grouping key: patient ID when available, otherwise the image itself."""
    if "patient_id" in df.columns and df["patient_id"].notna().all():
        return df["patient_id"].astype(str)
    log.warning("Patient IDs unavailable: falling back to image-level grouping (leakage cannot be ruled out).")
    return df["image"].astype(str)


def _choose_patients(patients: np.ndarray, fraction: float, rng: np.random.Generator) -> set[str]:
    shuffled = rng.permutation(np.sort(patients))
    n = int(round(len(shuffled) * fraction))
    if fraction > 0 and len(shuffled) > 1:
        n = max(1, min(n, len(shuffled) - 1))
    return set(shuffled[:n].tolist())


def assign_splits(
    df: pd.DataFrame,
    train_val_names: set[str] | None = None,
    test_names: set[str] | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Assign each image to train/val/test with all images of a patient kept together.

    If the official NIH ``train_val_list.txt`` / ``test_list.txt`` are supplied,
    the official (patient-level) test split is preserved and validation patients
    are drawn from the train/val pool. Otherwise a reproducible patient-grouped
    random split is created. Returns ``(df_with_split_column, method_description)``.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    groups = _patient_groups(df)

    if train_val_names and test_names:
        df["split"] = np.where(
            df["image"].isin(test_names), "test", np.where(df["image"].isin(train_val_names), "train", "unassigned")
        )
        pool = df["split"] == "train"
        val_patients = _choose_patients(groups[pool].unique(), val_fraction, rng)
        df.loc[pool & groups.isin(val_patients), "split"] = "val"
        method = (
            "official NIH train_val_list/test_list; validation = "
            f"{val_fraction:.0%} of train/val patients (seed={seed})"
        )
        unassigned = int((df["split"] == "unassigned").sum())
        if unassigned:
            log.warning("%d metadata rows are in neither official split list and are excluded.", unassigned)
    else:
        patients = groups.unique()
        test_patients = _choose_patients(patients, test_fraction, rng)
        remaining = np.array([p for p in np.sort(patients) if p not in test_patients])
        val_patients = _choose_patients(remaining, val_fraction / max(1e-9, 1 - test_fraction), rng)
        df["split"] = np.where(
            groups.isin(test_patients), "test", np.where(groups.isin(val_patients), "val", "train")
        )
        method = (
            f"patient-grouped random split (test={test_fraction:.0%}, val={val_fraction:.0%}, seed={seed}); "
            "official split lists not found"
        )
    return df, method


def development_subset(df: pd.DataFrame, max_images: int, seed: int = 42) -> pd.DataFrame:
    """Reproducible patient-level subset for development mode.

    The image budget is distributed across splits in proportion to their size.
    Whole patients are added until the budget is reached; then, where feasible,
    one extra patient is added for any label that has no positive example in a
    split. Patients never move between splits.
    """
    if max_images <= 0 or len(df) <= max_images:
        return df.copy()
    rng = np.random.default_rng(seed)
    groups = _patient_groups(df)
    selected_idx: list[int] = []
    for split in SPLITS:
        part = df[df["split"] == split]
        if part.empty:
            continue
        budget = max(1, int(round(max_images * len(part) / len(df))))
        part_groups = groups.loc[part.index]
        by_patient = {p: idx.tolist() for p, idx in part.groupby(part_groups).groups.items()}
        order = rng.permutation(np.array(sorted(by_patient), dtype=object))
        chosen: list[int] = []
        chosen_patients: set = set()
        for patient in order:
            if len(chosen) >= budget:
                break
            chosen.extend(by_patient[patient])
            chosen_patients.add(patient)
        # Label coverage: add one patient for each missing label where possible.
        for label in config.LABELS:
            has = any(label in labels for labels in df.loc[chosen, "labels"])
            if has:
                continue
            for patient in order:
                if patient in chosen_patients:
                    continue
                if any(label in labels for labels in df.loc[by_patient[patient], "labels"]):
                    chosen.extend(by_patient[patient])
                    chosen_patients.add(patient)
                    break
        selected_idx.extend(chosen)
    return df.loc[sorted(selected_idx)].copy()


def patient_overlap(df: pd.DataFrame) -> dict[str, int]:
    """Number of patients shared between each pair of splits (should be 0)."""
    if "patient_id" not in df.columns or df["patient_id"].isna().any() or "split" not in df.columns:
        return {}
    sets = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in SPLITS}
    return {
        "train-val": len(sets["train"] & sets["val"]),
        "train-test": len(sets["train"] & sets["test"]),
        "val-test": len(sets["val"] & sets["test"]),
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def label_statistics(df: pd.DataFrame) -> dict:
    y = encode_labels(df["labels"])
    positives = y.sum(axis=0).astype(int)
    n = len(df)
    co = (y.T @ y).astype(int)
    per_image = y.sum(axis=1)
    return {
        "num_images": n,
        "positives": {label: int(c) for label, c in zip(config.LABELS, positives)},
        "prevalence": {label: (float(c) / n if n else 0.0) for label, c in zip(config.LABELS, positives)},
        "no_finding_images": int((per_image == 0).sum()),
        "multi_label_images": int((per_image > 1).sum()),
        "labels_per_image_hist": {int(k): int(v) for k, v in zip(*np.unique(per_image, return_counts=True))},
        "cooccurrence": {
            "labels": list(config.LABELS),
            "matrix": co.tolist(),
        },
    }


def split_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        y = encode_labels(part["labels"]) if len(part) else np.zeros((0, config.NUM_LABELS))
        summary[split] = {
            "images": int(len(part)),
            "patients": int(part["patient_id"].nunique()) if "patient_id" in part and len(part) else 0,
            "positives": {label: int(c) for label, c in zip(config.LABELS, y.sum(axis=0))},
        }
    return summary


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #
def check_images(paths: list[str], workers: int = 8) -> list[tuple[str, str]]:
    """Verify image files; returns a list of (path, reason) for unreadable files."""

    def _check(p: str) -> Optional[tuple[str, str]]:
        ok, reason = verify_image_file(Path(p))
        return None if ok else (p, reason)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(_check, paths) if r is not None]


def survey_image_properties(paths: list[str], max_images: int = 500, seed: int = 42) -> dict:
    """Pixel dimensions and colour modes of a random sample (reads image headers only).

    Some Kaggle mirrors ship pre-resized copies (e.g. 224x224 instead of the
    original 1024x1024); training at a larger input size would only upsample.
    """
    from PIL import Image

    if not paths:
        return {"sampled": 0, "sizes": {}, "modes": {}, "max_side": None, "min_side": None}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(paths, dtype=object), size=min(max_images, len(paths)), replace=False)
    sizes: dict[str, int] = {}
    modes: dict[str, int] = {}
    sides: list[int] = []
    for p in chosen:
        try:
            with Image.open(p) as img:
                w, h = img.size
                sizes[f"{w}x{h}"] = sizes.get(f"{w}x{h}", 0) + 1
                modes[img.mode] = modes.get(img.mode, 0) + 1
                sides += [w, h]
        except Exception:  # noqa: BLE001 - unreadable files are counted by the integrity check
            continue
    return {
        "sampled": int(len(chosen)),
        "sizes": dict(sorted(sizes.items(), key=lambda kv: -kv[1])),
        "modes": modes,
        "max_side": max(sides) if sides else None,
        "min_side": min(sides) if sides else None,
    }


def validate_dataset(
    paths: DatasetPaths,
    verify: str = "sample",
    sample_size: int = 2000,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> tuple[dict, pd.DataFrame]:
    """Run all dataset checks and return ``(report, dataframe_with_splits)``.

    ``verify`` controls image integrity checking: ``"none"``, ``"sample"``
    (random ``sample_size`` images) or ``"all"``.
    """
    require_dataset(paths)
    df = load_metadata(paths.metadata_csv)
    index, duplicates = build_image_index(paths.image_root)
    df = attach_image_paths(df, index)

    present = df[df["exists"]]
    to_check = present["path"].tolist()
    if verify == "sample" and len(to_check) > sample_size:
        rng = np.random.default_rng(seed)
        to_check = rng.choice(np.array(to_check, dtype=object), size=sample_size, replace=False).tolist()
    elif verify == "none":
        to_check = []
    corrupted = check_images(to_check) if to_check else []
    corrupted_paths = {p for p, _ in corrupted}

    df["corrupted"] = df["path"].isin(corrupted_paths)
    df, split_method = assign_splits(
        df,
        read_split_list(paths.train_val_list),
        read_split_list(paths.test_list),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    metadata_names = set(df["image"])
    report = {
        "generated_at": utc_timestamp(),
        "paths": paths.to_dict(),
        "metadata_rows": df.attrs.get("source_rows", len(df)),
        "unique_images_in_metadata": int(len(df)),
        "duplicate_metadata_rows": df.attrs.get("duplicate_metadata_rows", 0),
        "unknown_label_tokens": df.attrs.get("unknown_labels", []),
        "image_files_found": len(index),
        "image_files_not_in_metadata": len(set(index) - metadata_names),
        "duplicate_filenames_on_disk": len(duplicates),
        "duplicate_filename_examples": sorted(duplicates)[:20],
        "missing_images": int((~df["exists"]).sum()),
        "missing_image_examples": df.loc[~df["exists"], "image"].head(20).tolist(),
        "integrity_check": {
            "mode": verify,
            "images_checked": len(to_check),
            "corrupted": len(corrupted),
            "corrupted_examples": [{"file": Path(p).name, "reason": r} for p, r in corrupted[:20]],
        },
        "readable_images_estimate": int(df["exists"].sum() - len(corrupted)),
        "image_properties": survey_image_properties(present["path"].tolist(), seed=seed),
        "patients": int(df["patient_id"].nunique()) if df.attrs.get("has_patient_id") else None,
        "split_method": split_method,
        "splits": split_summary(df),
        "unassigned_rows": int((df["split"] == "unassigned").sum()),
        "patient_overlap": patient_overlap(df),
        "labels": label_statistics(df[df["exists"]]),
        "label_policy": (
            "14 independent binary targets. 'No Finding' is encoded as all-zero, not as a separate class. "
            "Labels were text-mined from radiology reports by the dataset authors and are noisy; an absent "
            "label does not prove the finding is absent."
        ),
    }
    return report, df


def save_validation_report(report: dict, out_dir: Path) -> dict[str, Path]:
    """Write the report as JSON plus two CSVs (summary and per-label counts)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_json(report, out_dir / "dataset_validation.json")

    summary_rows = [
        ("metadata_rows", report["metadata_rows"]),
        ("unique_images_in_metadata", report["unique_images_in_metadata"]),
        ("image_files_found", report["image_files_found"]),
        ("missing_images", report["missing_images"]),
        ("duplicate_filenames_on_disk", report["duplicate_filenames_on_disk"]),
        ("images_integrity_checked", report["integrity_check"]["images_checked"]),
        ("corrupted_images", report["integrity_check"]["corrupted"]),
        ("image_sizes_in_sample", "; ".join(f"{k}: {v}" for k, v in report["image_properties"]["sizes"].items())),
        ("patients", report["patients"]),
        ("split_method", report["split_method"]),
    ]
    for split, info in report["splits"].items():
        summary_rows += [(f"{split}_images", info["images"]), (f"{split}_patients", info["patients"])]
    for pair, count in report["patient_overlap"].items():
        summary_rows.append((f"patient_overlap_{pair}", count))
    summary_csv = out_dir / "dataset_validation_summary.csv"
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(summary_csv, index=False)

    label_rows = []
    for label in config.LABELS:
        row = {
            "label": config.DISPLAY_NAMES[label],
            "positives": report["labels"]["positives"][label],
            "prevalence": report["labels"]["prevalence"][label],
        }
        for split, info in report["splits"].items():
            row[f"{split}_positives"] = info["positives"][label]
        label_rows.append(row)
    label_csv = out_dir / "dataset_label_counts.csv"
    pd.DataFrame(label_rows).to_csv(label_csv, index=False)
    return {"json": json_path, "summary_csv": summary_csv, "labels_csv": label_csv}

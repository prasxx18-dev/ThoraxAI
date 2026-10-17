"""Metadata loading, label mapping/encoding, splitting and dataset validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.dataset import (
    DatasetError,
    MetadataError,
    assign_splits,
    attach_image_paths,
    build_image_index,
    decode_labels,
    development_subset,
    discover_dataset,
    encode_labels,
    load_metadata,
    parse_finding_labels,
    patient_overlap,
    read_split_list,
    require_dataset,
    save_validation_report,
    validate_dataset,
)


def test_label_mapping_has_14_unique_labels():
    assert len(config.LABELS) == 14 == config.NUM_LABELS
    assert len(set(config.LABELS)) == 14
    assert "No Finding" not in config.LABELS
    assert config.DISPLAY_NAMES["Pleural_Thickening"] == "Pleural Thickening"


@pytest.mark.parametrize(
    "raw, known, unknown",
    [
        ("Cardiomegaly|Effusion", ["Cardiomegaly", "Effusion"], []),
        ("No Finding", [], []),
        ("", [], []),
        (float("nan"), [], []),
        ("pleural thickening|Hernia", ["Pleural_Thickening", "Hernia"], []),
        ("Mass|Mass|Alien", ["Mass"], ["Alien"]),
    ],
)
def test_parse_finding_labels(raw, known, unknown):
    assert parse_finding_labels(raw) == (known, unknown)


def test_multilabel_encoding_roundtrip():
    y = encode_labels([["Cardiomegaly", "Effusion"], [], ["Hernia"]])
    assert y.shape == (3, 14) and y.dtype == np.float32
    assert y[0].sum() == 2 and y[1].sum() == 0 and y[2, 13] == 1
    assert decode_labels(y[0]) == ["Cardiomegaly", "Effusion"]
    with pytest.raises(ValueError):
        encode_labels([["NotALabel"]])


def test_load_metadata(synthetic_dataset):
    df = load_metadata(synthetic_dataset / "Data_Entry_2017.csv")
    assert {"image", "labels", "patient_id", "finding_labels"} <= set(df.columns)
    assert len(df) == 60
    assert df.attrs["has_patient_id"]


def test_load_metadata_kaggle_header_format(tmp_path):
    """Mirror the NIH Data_Entry_2017.csv header (bracketed column names, trailing comma)."""
    csv = tmp_path / "Data_Entry_2017.csv"
    csv.write_text(
        "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,"
        "OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],\n"
        "00000001_000.png,Cardiomegaly,0,1,58,M,PA,2682,2749,0.143,0.143,\n"
        "00000001_001.png,Cardiomegaly|Emphysema,1,1,58,M,PA,2894,2729,0.143,0.143,\n"
        "00000002_000.png,No Finding,0,2,81,M,PA,2500,2048,0.171,0.171,\n"
    )
    df = load_metadata(csv)
    assert len(df) == 3
    assert df["labels"].tolist() == [["Cardiomegaly"], ["Cardiomegaly", "Emphysema"], []]
    assert df["patient_id"].tolist() == ["1", "1", "2"]
    assert df["view_position"].tolist() == ["PA", "PA", "PA"]
    assert df.attrs["unknown_labels"] == []


def test_load_metadata_missing_file(tmp_path):
    with pytest.raises(MetadataError):
        load_metadata(tmp_path / "nope.csv")


def test_load_metadata_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(csv, index=False)
    with pytest.raises(MetadataError, match="missing required column"):
        load_metadata(csv)


def test_load_metadata_empty(tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("Image Index,Finding Labels\n")
    with pytest.raises(MetadataError, match="no rows"):
        load_metadata(csv)


def test_duplicate_metadata_rows_are_reported(tmp_path):
    csv = tmp_path / "dup.csv"
    pd.DataFrame({"Image Index": ["a.png", "a.png"], "Finding Labels": ["Mass", "Mass"]}).to_csv(csv, index=False)
    df = load_metadata(csv)
    assert len(df) == 1 and df.attrs["duplicate_metadata_rows"] == 1


def test_discovery_and_missing_dataset(tmp_path, synthetic_dataset):
    paths = discover_dataset(synthetic_dataset)
    assert paths.is_ready and paths.metadata_csv.name == "Data_Entry_2017.csv"
    assert paths.test_list is not None and paths.train_val_list is not None
    empty = discover_dataset(tmp_path / "does_not_exist")
    assert not empty.is_ready
    with pytest.raises(DatasetError, match="kaggle"):
        require_dataset(empty)


def test_image_index_detects_duplicates_and_missing(synthetic_dataset):
    dup_dir = synthetic_dataset / "images_002" / "images"
    dup_dir.mkdir(parents=True)
    first = next((synthetic_dataset / "images_001" / "images").glob("*.png"))
    (dup_dir / first.name).write_bytes(first.read_bytes())
    first_missing = sorted((synthetic_dataset / "images_001" / "images").glob("*.png"))[-1]
    first_missing.unlink()
    index, duplicates = build_image_index(synthetic_dataset)
    assert first.name in duplicates
    df = attach_image_paths(load_metadata(synthetic_dataset / "Data_Entry_2017.csv"), index)
    assert (~df["exists"]).sum() == 1


def test_official_split_has_no_patient_overlap(synthetic_dataset):
    df = load_metadata(synthetic_dataset / "Data_Entry_2017.csv")
    df, method = assign_splits(df, read_split_list(synthetic_dataset / "train_val_list.txt"),
                               read_split_list(synthetic_dataset / "test_list.txt"), val_fraction=0.2)
    assert "official" in method
    assert set(df["split"]) == {"train", "val", "test"}
    assert patient_overlap(df) == {"train-val": 0, "train-test": 0, "val-test": 0}


def test_grouped_split_is_reproducible_and_leak_free(synthetic_dataset):
    df = load_metadata(synthetic_dataset / "Data_Entry_2017.csv")
    a, method = assign_splits(df, seed=7)
    b, _ = assign_splits(df, seed=7)
    assert "patient-grouped" in method
    assert (a["split"] == b["split"]).all()
    assert all(v == 0 for v in patient_overlap(a).values())


def test_development_subset_keeps_patients_together(synthetic_dataset):
    df = load_metadata(synthetic_dataset / "Data_Entry_2017.csv")
    df, _ = assign_splits(df, seed=1)
    sub = development_subset(df, max_images=20, seed=1)
    assert 0 < len(sub) < len(df)
    for pid, group in sub.groupby("patient_id"):
        assert len(group) == (df["patient_id"] == pid).sum()  # whole patients only
        assert group["split"].nunique() == 1


def test_validate_dataset_report(synthetic_dataset, tmp_path):
    bad = synthetic_dataset / "images_001" / "images" / "00000001_000.png"
    bad.write_bytes(b"not a png")
    report, df = validate_dataset(discover_dataset(synthetic_dataset), verify="all")
    assert report["metadata_rows"] == 60
    assert report["image_files_found"] == 60
    assert report["missing_images"] == 0
    assert report["integrity_check"]["corrupted"] == 1
    assert sum(s["images"] for s in report["splits"].values()) == 60
    assert all(v == 0 for v in report["patient_overlap"].values())
    assert sum(report["labels"]["positives"].values()) >= 14
    files = save_validation_report(report, tmp_path / "out")
    assert all(Path(p).is_file() for p in files.values())
    assert len(pd.read_csv(files["labels_csv"])) == 14

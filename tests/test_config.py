"""local_config.json handling, image-property survey and training dry run."""

import json

import pytest

import config
from src.dataset import survey_image_properties


def test_local_config_loading(tmp_path):
    f = tmp_path / "local_config.json"
    assert config._load_local_config(f) == {}
    f.write_text(json.dumps({"data_dir": "D:/nih"}))
    assert config._load_local_config(f) == {"data_dir": "D:/nih"}
    f.write_text("{not json")
    with pytest.warns(UserWarning):
        assert config._load_local_config(f) == {}
    f.write_text("[1, 2]")
    with pytest.warns(UserWarning):
        assert config._load_local_config(f) == {}


def test_setting_precedence(tmp_path, monkeypatch):
    default = tmp_path / "default"
    monkeypatch.setattr(config, "LOCAL_SETTINGS", {})
    monkeypatch.delenv("THORAXAI_DATA_DIR", raising=False)
    assert config._setting("THORAXAI_DATA_DIR", "data_dir", default) == default

    monkeypatch.setattr(config, "LOCAL_SETTINGS", {"data_dir": str(tmp_path / "from file")})
    assert config._setting("THORAXAI_DATA_DIR", "data_dir", default) == (tmp_path / "from file").resolve()

    monkeypatch.setattr(config, "LOCAL_SETTINGS", {"data_dir": "relative/data"})
    assert config._setting("THORAXAI_DATA_DIR", "data_dir", default) == (config.PROJECT_ROOT / "relative/data").resolve()

    monkeypatch.setenv("THORAXAI_DATA_DIR", str(tmp_path / "from env"))
    assert config._setting("THORAXAI_DATA_DIR", "data_dir", default) == (tmp_path / "from env").resolve()


def test_survey_image_properties(synthetic_dataset):
    paths = [str(p) for p in synthetic_dataset.rglob("*.png")]
    props = survey_image_properties(paths, max_images=10)
    assert props["sampled"] == 10
    assert props["sizes"] == {"72x72": 10} and props["modes"] == {"L": 10}
    assert props["max_side"] == props["min_side"] == 72
    assert survey_image_properties([])["sampled"] == 0


def test_training_dry_run_writes_nothing(synthetic_dataset, tmp_path, monkeypatch):
    from src.train import dry_run

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    cfg = config.TrainConfig(architecture="mobilenetv2", image_size=64, batch_size=4, pretrained=False,
                             val_fraction=0.25, max_missing_fraction=0.0)
    report = dry_run(cfg, data_root=synthetic_dataset)
    assert report["model"]["output_shape"] == [4, 14]
    assert report["pipeline_checks"]["train"]["batch_images"] == [4, 64, 64, 3]
    assert report["pipeline_checks"]["val"]["batch_labels"] == [4, 14]
    assert 0 <= report["pipeline_checks"]["val"]["pixel_range"][0] <= report["pipeline_checks"]["val"]["pixel_range"][1] <= 255
    assert "official" in report["split_method"]
    assert not (tmp_path / "models").exists() and not (tmp_path / "results").exists()

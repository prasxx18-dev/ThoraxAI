"""End-to-end SOFTWARE smoke test: train -> thresholds -> evaluate on synthetic noise.

This proves the pipeline runs and writes the expected artifacts. The numbers
it produces are meaningless (random noise images, random weights) and are
written only to a temporary directory, never to models/ or results/.
"""

import json

import pandas as pd

import config
from src.evaluate import evaluate_model
from src.train import run_training


def test_train_and_evaluate_end_to_end(synthetic_dataset, tmp_path, local_config):
    models_dir, results_dir = tmp_path / "models", tmp_path / "results"
    cfg = config.TrainConfig(architecture="mobilenetv2", mode="development", image_size=64, batch_size=8, epochs=1,
                             fine_tune_epochs=1, fine_tune_layers=5, pretrained=False, dev_max_images=50,
                             val_fraction=0.25, max_missing_fraction=0.0)
    summary = run_training(cfg, data_root=synthetic_dataset, models_dir=models_dir, results_dir=results_dir)
    mdir, rdir = models_dir / "mobilenetv2", results_dir / "mobilenetv2"
    for name in ("best_model.keras", "final_model.keras", "model_config.json", "label_mapping.json",
                 "threshold_config.json"):
        assert (mdir / name).is_file(), name
    for name in ("training_history.json", "validation_metrics.json", "splits.csv", "training_log.csv"):
        assert (rdir / name).is_file(), name
    assert summary["epochs_completed"] == 2
    assert summary["pretrained_weights_loaded"] is False
    thr = json.loads((mdir / "threshold_config.json").read_text())
    assert thr["selected_on"] == "validation split only" and len(thr["thresholds"]) == 14

    splits = pd.read_csv(rdir / "splits.csv")
    assert set(splits["split"]) == {"train", "val", "test"}

    result = evaluate_model("mobilenetv2", data_root=synthetic_dataset, models_dir=models_dir,
                            results_dir=results_dir, batch_size=8)
    assert result["num_samples"] == (splits["split"] == "test").sum()
    assert len(result["per_label"]) == 14
    for name in ("test_metrics.json", "per_label_metrics.csv", "test_predictions.csv",
                 "confusion_matrices/confusion_matrices_all.png", "training_curves.png"):
        assert (rdir / name).is_file(), name
    assert (results_dir / "model_comparison.csv").is_file()

    # The app pages must render the artifacts the pipeline actually produces.
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from src.dataset import discover_dataset, save_validation_report, validate_dataset

    report, _ = validate_dataset(discover_dataset(synthetic_dataset), verify="all")
    save_validation_report(report, results_dir)
    root = config.PROJECT_ROOT
    local_config(data_dir=synthetic_dataset, models_dir=models_dir, results_dir=results_dir)
    st.cache_data.clear()
    def rendered(at):
        cells = [str(v) for d in at.dataframe for v in d.value.to_numpy().ravel()]
        return " ".join(str(m.value) for m in at.markdown) + " " + " ".join(cells)

    try:
        cases = [("app.py", None), ("app_pages/analyze.py", None), ("app_pages/about.py", None),
                 ("app_pages/insights.py", "Model performance"), ("app_pages/insights.py", "Dataset"),
                 ("app_pages/insights.py", "System status")]
        for page, view in cases:
            at = AppTest.from_file(str(root / "app.py"), default_timeout=120)
            if view:
                at.session_state["insights_view"] = view
            at.run()
            if page != "app.py":
                at.switch_page(page).run()
            assert not at.exception, (page, view, [e.value for e in at.exception])
            text = rendered(at)
            if view == "Model performance":
                assert "Macro ROC-AUC" in text  # real metrics from test_metrics.json
            if view == "Dataset":
                assert "Patients" in text
            if view == "System status":
                assert "Dataset found." in text and "Trained: MobileNetV2" in text
            if page == "app_pages/analyze.py":
                assert any(b.label == "Analyze X-ray" for b in at.button)
    finally:
        st.cache_data.clear()

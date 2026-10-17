"""Streamlit application tests (streamlit.testing.v1.AppTest).

Every page must render without exceptions when the dataset and models are
missing, show an empty state instead of any predictions or metrics, and report
the dataset status from genuine filesystem checks.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py", "app_pages/analyze.py", "app_pages/insights.py", "app_pages/about.py"]
INSIGHT_VIEWS = ["Model performance", "Dataset", "System status"]


@pytest.fixture
def empty_environment(tmp_path, monkeypatch, local_config):
    """App configured (via a temporary local_config.json) with no dataset and no models."""
    monkeypatch.chdir(ROOT)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield tmp_path
    st.cache_data.clear()


def open_page(page: str, insights_view: str | None = None) -> AppTest:
    """Run the real entrypoint (app.py) and navigate to ``page``, as in the browser."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    if insights_view:
        at.session_state["insights_view"] = insights_view
    at.run()
    if page != "app.py":
        at.switch_page(page).run()
    return at


def page_text(at: AppTest) -> str:
    """All Markdown/HTML text rendered on the page."""
    return " ".join(str(m.value) for m in at.markdown) + " " + " ".join(str(c.value) for c in at.caption)


@pytest.mark.parametrize("page", PAGES)
def test_page_starts_without_dataset_or_model(empty_environment, page):
    at = open_page(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("view", INSIGHT_VIEWS)
def test_insight_views_start_without_dataset_or_model(empty_environment, view):
    at = open_page("app_pages/insights.py", insights_view=view)
    assert not at.exception, [e.value for e in at.exception]


def test_analysis_page_shows_empty_state_instead_of_predictions(empty_environment):
    at = open_page("app_pages/analyze.py")
    text = page_text(at)
    assert "Analysis is not available yet" in text
    assert "never displays simulated results" in text
    assert not any(b.label == "Analyze X-ray" for b in at.button)
    assert "crossed the display threshold" not in text


def test_performance_view_empty_state(empty_environment):
    at = open_page("app_pages/insights.py", insights_view="Model performance")
    text = page_text(at)
    assert "No evaluation results yet" in text
    assert "Macro ROC-AUC" not in text  # no metrics invented


def test_home_is_product_focused(empty_environment):
    at = open_page("app.py")
    text = page_text(at)
    assert "Intelligent insights" in text
    assert any(b.label == "Analyze an X-ray" for b in at.button)
    for phrase in ("mini project", "mini-project", "student project", "college", "synthetic"):
        assert phrase not in text.lower()
    assert "Research use only" in text  # usage notice is still present (footer)


def test_home_call_to_action_opens_workspace(empty_environment):
    at = open_page("app.py")
    next(b for b in at.button if b.label == "Analyze an X-ray").click().run()
    assert not at.exception
    assert "Analyze a chest X-ray" in page_text(at)


def test_about_keeps_intended_use_notice(empty_environment):
    assert "not intended for independent medical diagnosis" in page_text(open_page("app_pages/about.py"))


def test_analysis_page_with_trained_model(empty_environment):
    from src.model import build_model
    from src.utils import save_json

    model, _ = build_model("mobilenetv2", image_size=64, pretrained=False)
    d = config.MODELS_DIR / "mobilenetv2"
    d.mkdir(parents=True)
    model.save(d / "best_model.keras")
    save_json({"thresholds": {lbl: 0.5 for lbl in config.LABELS}}, d / "threshold_config.json")
    at = open_page("app_pages/analyze.py")
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == "mobilenetv2"
    analyze = next(b for b in at.button if b.label == "Analyze X-ray")
    assert analyze.disabled  # nothing uploaded yet
    text = page_text(at)
    from src.findings_info import RESULTS_NOTICE

    assert text.count(RESULTS_NOTICE) == 1  # exactly one subtle notice on the analysis page
    for repeated in ("not a confirmed finding", "has not been clinically validated", "Research use only"):
        assert repeated not in text, repeated


# ----------------------------------------------------------------- dataset status
def system_text(at: AppTest) -> str:
    cells = [str(v) for d in at.dataframe for v in d.value.to_numpy().ravel()]  # full values, not truncated repr
    return page_text(at) + " " + " ".join(cells)


def test_system_status_dataset_found_when_configured(empty_environment, local_config, synthetic_dataset):
    local_config(data_dir=synthetic_dataset)
    at = open_page("app_pages/insights.py", insights_view="System status")
    assert not at.exception
    text = system_text(at)
    assert "Metadata CSV and images found" in text and "Dataset found." in text
    assert str(synthetic_dataset) in text


def test_system_status_missing_without_images(empty_environment, local_config, tmp_path):
    """A folder with the metadata CSV but no images must not be reported as found."""
    csv_only = tmp_path / "csv_only"
    csv_only.mkdir()
    (csv_only / "Data_Entry_2017.csv").write_text("Image Index,Finding Labels\na.png,Mass\n")
    local_config(data_dir=csv_only)
    text = system_text(open_page("app_pages/insights.py", insights_view="System status"))
    assert "No PNG/JPEG images found" in text and "Dataset not available." in text


def test_system_status_missing_folder(empty_environment):
    text = system_text(open_page("app_pages/insights.py", insights_view="System status"))
    assert "Configured folder does not exist" in text


def test_running_app_picks_up_new_local_config(empty_environment, local_config, synthetic_dataset):
    """Regression: the server started before local_config.json pointed at the dataset.

    The same AppTest session (same process, cache NOT cleared) must report the
    dataset once the config file is changed - no restart required.
    """
    at = open_page("app_pages/insights.py", insights_view="System status")
    assert "Dataset not available." in system_text(at)
    local_config(data_dir=synthetic_dataset)  # user edits local_config.json while the app runs
    config.DATA_DIR = config.PROJECT_ROOT / "stale" / "path"  # simulate the value frozen at import time
    at.run()
    assert "Dataset found." in system_text(at)


def test_no_legacy_pages_directory():
    """A ``pages/`` folder beside app.py triggers Streamlit's legacy auto-navigation, which can hijack
    routes such as /about and replace the top navigation bar with a sidebar. Pages live in app_pages/."""
    assert not (ROOT / "pages").exists()
    assert all((ROOT / p).is_file() for p in PAGES if p != "app.py")

"""Finding explanations, results headline, percentage display and finding-card rendering."""

import re
from urllib.parse import urlparse

import pytest

import config
from src.findings_info import FINDINGS, RESULTS_NOTICE, all_references
from src.predict import PredictionError, to_percent
from src.summary import MULTI_LABEL_NOTE, NONE_MESSAGE, build_summary
from src.views.findings_view import PCT_LABEL, all_outputs_html, finding_card_html, glossary_html

TRUSTED_DOMAINS = {"medlineplus.gov", "www.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "my.clevelandclinic.org",
                   "radiopaedia.org"}
FORBIDDEN = ["you should", "you must", "prescrib", "dosage", "take medication", "will cure", "definitely",
             "certainly", "guaranteed", "you have ", "deadly", "fatal", "life-threatening"]
# Sections removed from the main results; they must not reappear in cards.
OLD_SECTIONS = ["Also called", "In simple terms", "What it can look like", "Common symptoms",
                "Why a clinician", "AI prediction vs", "Sources:"]


# ----------------------------------------------------------------- knowledge base
def test_every_label_has_a_short_explanation_in_model_order():
    assert list(FINDINGS) == list(config.LABELS)
    for label, info in FINDINGS.items():
        assert info.label == label and info.name == config.DISPLAY_NAMES[label]
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", info.meaning.strip()) if s]
        assert 1 <= len(sentences) <= 2, (label, sentences)
        assert 30 <= len(info.meaning) <= 230, (label, len(info.meaning))
        assert info.sources, label


def test_sources_are_https_and_trusted():
    for title, url in all_references():
        parsed = urlparse(url)
        assert parsed.scheme == "https" and parsed.netloc in TRUSTED_DOMAINS, url
        assert title.strip()


def test_no_advice_alarm_or_certainty_language():
    blob = " ".join(i.meaning for i in FINDINGS.values()).lower() + " " + RESULTS_NOTICE.lower()
    for phrase in FORBIDDEN:
        assert phrase not in blob, phrase


def test_requested_example_wording():
    assert FINDINGS["Infiltration"].meaning.startswith("Material such as fluid or inflammatory cells")
    assert "pneumonia" in FINDINGS["Consolidation"].meaning.lower()


# ----------------------------------------------------------------- headline
def _maps(scores: dict, default=0.1, thr=0.5):
    return ({lbl: scores.get(lbl, default) for lbl in config.LABELS}, {lbl: thr for lbl in config.LABELS})


def test_headline_lists_flagged_in_score_order():
    s, t = _maps({"Consolidation": 0.801, "Infiltration": 0.843})
    summ = build_summary(s, t)
    assert summ.flagged == ["Infiltration", "Consolidation"]
    assert summ.headline == "2 findings crossed the display threshold"
    assert summ.top_label == "Infiltration" and summ.top_score == pytest.approx(0.843)


def test_headline_none_flagged_is_neutral():
    s, t = _maps({})
    summ = build_summary(s, t)
    assert summ.status == "none_flagged" and summ.headline == NONE_MESSAGE
    assert "normal" not in summ.headline.lower()
    assert "independently" in MULTI_LABEL_NOTE


def test_headline_singular_and_validation():
    s, t = _maps({"Mass": 0.6})
    assert build_summary(s, t).headline == "1 finding crossed the display threshold"
    with pytest.raises(ValueError):
        build_summary({"Mass": 0.2}, {})


# ----------------------------------------------------------------- percentages
@pytest.mark.parametrize("score, pct", [(0.8431, 84.3), (0.801, 80.1), (0.0, 0.0), (1.0, 100.0),
                                        (0.00049, 0.0), (0.99951, 100.0), (1 + 1e-9, 100.0), (-1e-9, 0.0)])
def test_to_percent(score, pct):
    assert to_percent(score) == pct


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_to_percent_rejects_invalid(bad):
    with pytest.raises(PredictionError):
        to_percent(bad)


# ----------------------------------------------------------------- card rendering
def test_card_contains_only_name_percentage_and_short_explanation():
    html = finding_card_html("Infiltration", 0.8431)
    assert "Infiltration" in html and "84.3%" in html and PCT_LABEL in html
    assert "What it means" in html and FINDINGS["Infiltration"].meaning in html
    assert 'aria-valuenow="84.3"' in html
    assert "0.843" not in html and "threshold" not in html.lower()
    for section in OLD_SECTIONS:
        assert section not in html, section
    assert "probability" not in html.lower()  # never claims a calibrated probability


def test_all_outputs_list_has_14_percentages_and_no_decimals_or_thresholds():
    s, _ = _maps({"Effusion": 0.9123})
    html = all_outputs_html(s, crossed=["Effusion"])
    assert html.count('class="tx-orow"') == 14
    assert html.index("Effusion") < html.index("Atelectasis")  # sorted by output
    assert "91.2%" in html and "0.912" not in html and "threshold" not in html.lower()


def test_glossary_covers_all_findings():
    html = glossary_html()
    assert all(info.name in html for info in FINDINGS.values())
    assert "%" not in html  # general information, no scores


def test_finding_cards_render_in_streamlit():
    from streamlit.testing.v1 import AppTest

    def app():
        import config as cfg
        from src.views.findings_view import render_finding_cards

        scores = {lbl: 0.2 for lbl in cfg.LABELS}
        scores.update({"Infiltration": 0.843, "Consolidation": 0.801})
        render_finding_cards(["Infiltration", "Consolidation"], scores)

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(str(m.value) for m in at.markdown)
    assert html.count('class="tx-fcard"') == 2 and "84.3%" in html and "80.1%" in html

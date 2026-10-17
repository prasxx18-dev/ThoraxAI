"""About / model information: overview, architecture, data, multi-label scoring, Grad-CAM,
evaluation (read from real artifacts), findings glossary, limitations, privacy and sources."""

import streamlit as st

import config
from src.ui import (USAGE_NOTICE, feature_card, feature_grid, footer, load_model_config, notice, page_header,
                    page_setup, render, stat_tiles)
from src.utils import load_json_if_exists
from src.views.findings_view import glossary_html, sources_html

page_setup("About")
page_header("About", "AI-assisted chest X-ray analysis, built for transparency",
            "ThoraxAI explores how deep learning can surface multiple thoracic findings from a single chest X-ray, "
            "and makes every output inspectable with visual explanations and openly reported evaluation.")

st.subheader("How it works", anchor="how-it-works")
feature_grid([
    feature_card("upload", "Image preparation",
                 "The upload is validated, converted to a consistent grayscale format and resized to 224 × 224 "
                 "pixels, the model's input size."),
    feature_card("layers", "DenseNet121 model",
                 "A DenseNet121 convolutional network, initialised from ImageNet and trained on ChestX-ray14, "
                 "ends in 14 independent sigmoid outputs — one per finding."),
    feature_card("eye", "Grad-CAM explanation",
                 "Grad-CAM traces a finding's output back to the image regions that influenced it most and "
                 "overlays them as a heatmap."),
])

st.subheader("Multi-label classification and percentages", anchor="multi-label")
st.markdown(
    "- Each of the 14 findings is scored **independently** (sigmoid outputs, not softmax), so several findings can "
    "appear for the same image and the percentages do not add up to 100%.\n"
    "- The percentage shown on a result card is the model's output for that finding multiplied by 100, labelled "
    "**model-estimated likelihood**. The model has **not been calibrated**, so these values express the model's "
    "confidence, not the true chance that a finding is present.\n"
    "- A finding card is shown when its output reaches the finding's **display threshold**. Validated thresholds "
    "were chosen per finding on held-out validation data to balance missed findings against false alarms; they are "
    "not clinical decision points. All 14 outputs remain available under *View all 14 model outputs*.")

st.subheader("Model evaluation", anchor="evaluation")
test = load_json_if_exists(config.RESULTS_DIR / "densenet121" / "test_metrics.json")
mcfg = load_model_config("densenet121")
if test:
    macro = test["macro"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    stat_tiles([
        ("Macro ROC-AUC", fmt(macro["roc_auc"]), "0.5 = chance level"),
        ("Macro PR-AUC", fmt(macro["average_precision"]), "average precision"),
        ("Macro F1", fmt(macro["f1"]), "at validated thresholds"),
        ("Held-out test images", f"{test['num_samples']:,}", "patient-disjoint"),
    ])
    sizes = mcfg.get("split_sizes") or {}
    if test.get("mode") == "development":
        detail = " · ".join(f"{k}: {v['images']:,} images" for k, v in sizes.items())
        st.write(f"These results come from a **limited development subset** of the dataset ({detail}). They are only "
                 "modestly above chance and are not representative of what this architecture can reach with "
                 "full-dataset training. Values are read from the evaluation output of the trained model.")
    st.page_link("app_pages/insights.py", label="See curves, confusion matrices and training history in Insights",
                 icon=":material/insights:")
else:
    st.write("No evaluation results are available yet. Metrics appear here after the evaluation script has been run.")

st.subheader("Dataset", anchor="data-source")
st.write("Training and evaluation use the NIH Clinical Center **ChestX-ray14** dataset (Wang et al., “ChestX-ray8: "
         "Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of "
         "Common Thorax Diseases”, IEEE CVPR 2017), used under its terms of use. It contains frontal chest X-rays "
         "with 14 finding labels text-mined from radiology reports. Patients never appear in more than one of the "
         "training, validation and test splits.")

st.subheader("The 14 findings", anchor="findings")
render(glossary_html())

st.subheader("Intended use and limitations", anchor="intended-use")
notice(USAGE_NOTICE, "warning")
st.markdown(
    "- ThoraxAI is a research and educational tool. It is not a medical device and has not been clinically "
    "validated or approved by any regulator. It cannot confirm a diagnosis.\n"
    "- Model-estimated likelihoods are uncalibrated outputs, not probabilities of disease.\n"
    "- Training labels were extracted automatically from radiology reports and contain errors. All images come from "
    "a single hospital system, so results may not generalise to other scanners or populations.\n"
    "- Images are analyzed at 224 × 224 pixels, which loses fine detail.\n"
    "- Grad-CAM heatmaps show where the model looked, not where disease is, and can highlight artefacts such as "
    "tubes, wires or text markers.")

st.subheader("Privacy", anchor="privacy")
st.markdown(
    "- Uploaded images are processed in memory on the computer running ThoraxAI and are not sent to any third-party "
    "service.\n"
    "- Images and results are not stored after your session ends unless you download a heatmap.\n"
    "- No account or personal information is required. Avoid uploading images that contain names or other patient "
    "identifiers.")

with st.expander("Medical information sources", icon=":material/menu_book:"):
    st.caption("The short finding explanations are general information based on these sources (accessed "
               "17 October 2026). They are not medical advice.")
    render(sources_html())

footer()

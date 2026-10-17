"""Analysis workspace: upload, real model inference, percentage finding cards and Grad-CAM.

All results come from the trained model via src.predict / src.gradcam - nothing is simulated.
Percentages are the model's sigmoid outputs x 100 (see src/predict.py); they are labelled
"model-estimated likelihood" because the model has not been calibrated.
"""

import hashlib
import io

import streamlit as st

import config
from src.findings_info import RESULTS_NOTICE
from src.preprocessing import ImageValidationError, load_image, prepare_batch
from src.summary import MULTI_LABEL_NOTE, NONE_MESSAGE, build_summary
from src.ui import (empty_state, esc, footer, model_dir, notice, page_header, page_setup, render, section_title,
                    system_status)
from src.utils import get_logger, load_json_if_exists
from src.views.findings_view import all_outputs_html, render_finding_cards

log = get_logger("analyze_page")

page_setup("Analyze X-ray")
page_header("Workspace", "Analyze a chest X-ray",
            "Upload a frontal chest X-ray to see model-estimated likelihoods for 14 findings and a visual "
            "explanation of what influenced them.")

ERROR_HELP = {
    "unsupported_format": "Please upload a PNG or JPEG image.",
    "corrupted": "This file could not be read as an image. Try exporting it again as PNG or JPEG.",
    "empty": "The selected file is empty.",
    "too_small": f"The image is too small. Please use an image at least {config.MIN_IMAGE_SIDE} px on each side.",
    "too_large": f"The file is too large. Please use an image under {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
}

ss = st.session_state
ss.setdefault("upload_nonce", 0)


def _purge_results(keep_file_key=None) -> None:
    """Drop cached results/heatmaps that don't belong to the current image."""
    for key in [k for k in ss.keys() if k.startswith(("result_", "cam_"))]:
        if keep_file_key is None or keep_file_key not in key:
            del ss[key]


def remove_image() -> None:
    ss.upload_nonce += 1  # a new uploader key clears the selected file
    _purge_results()


@st.cache_resource(show_spinner=False)
def get_predictor(arch: str, mtime: float):
    """One loaded model per architecture; ``mtime`` reloads it after retraining."""
    from src.predict import Predictor

    return Predictor.from_model_dir(model_dir(arch))


status = system_status()
trained = status["trained_models"]
if not trained:
    empty_state(
        "activity", "Analysis is not available yet",
        "No trained model was found, so images cannot be analyzed. ThoraxAI never displays simulated results. "
        "Setup steps are listed under Insights → System status.",
        commands=["python scripts/train_model.py --model densenet121 --mode development",
                  "python scripts/evaluate_model.py --model densenet121"],
    )
    st.page_link("app_pages/insights.py", label="Open Insights", icon=":material/insights:")
    footer()
    st.stop()

# ------------------------------------------------------------------ upload + preview
left, right = st.columns([1, 1.2], gap="large")
with left:
    uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"],
                                key=f"uploader_{ss.upload_nonce}",
                                help="Images are processed locally in this session and are not sent to "
                                     "third-party services.")
    st.caption("Drag and drop or browse · PNG or JPEG · up to 20 MB · frontal (PA or AP) view. "
               "To replace the image, drop a new file here.")

    with st.expander("Display settings", icon=":material/tune:"):
        arch = st.selectbox("Model", trained, disabled=len(trained) == 1,
                            format_func=lambda a: config.ARCHITECTURES[a]["display_name"])
        mode = st.radio("Which findings to show",
                        ["Validated thresholds (recommended)", "One threshold for all findings",
                         "Custom threshold per finding"],
                        help="A finding card is shown when its model-estimated likelihood reaches the display "
                             "threshold. Validated thresholds were chosen on held-out validation data.")
        thr_file = load_json_if_exists(model_dir(arch) / "threshold_config.json") or {}
        validated = {lbl: float(thr_file.get("thresholds", {}).get(lbl, 0.5)) for lbl in config.LABELS}
        custom_thr = None
        if mode == "One threshold for all findings":
            custom_thr = [st.slider("Display threshold (%)", 5, 95, 50, 1) / 100] * config.NUM_LABELS
        elif mode == "Custom threshold per finding":
            custom_thr = [st.slider(f"{config.DISPLAY_NAMES[lbl]} (%)", 1, 99, int(round(validated[lbl] * 100)), 1,
                                    key=f"thr_{lbl}") / 100 for lbl in config.LABELS]

image = data = None
if uploaded is not None:
    data = uploaded.getvalue()
    try:
        image = load_image(data, filename=uploaded.name)
    except ImageValidationError as exc:
        with left:
            notice(ERROR_HELP.get(exc.category, "This file could not be read as an image."), "error",
                   "This file can't be analyzed.")

file_key = hashlib.sha256(data).hexdigest()[:16] if data else None
if ss.get("current_file_key") != file_key:  # new, replaced or removed image -> never show stale results
    _purge_results(keep_file_key=file_key)
    ss.current_file_key = file_key

with right:
    if image is None:
        empty_state("image", "No image selected", "Your chest X-ray preview will appear here.")
    else:
        st.image(image, width="stretch")
        w, h = image.size
        st.caption(f"{esc(uploaded.name)} · {w} × {h} px")
        if w / h > 1.6 or h / w > 1.6:
            notice("This image has an unusual shape and may not be a standard frontal chest X-ray.", "warning")

try:
    with st.spinner("Loading the analysis model…"):
        predictor = get_predictor(arch, (model_dir(arch) / "best_model.keras").stat().st_mtime)
except Exception:  # noqa: BLE001 - details go to the server log, not to the visitor
    log.exception("Model could not be loaded")
    notice("The analysis model could not be loaded. Please try again later or contact the administrator.",
           "error", "Analysis unavailable.")
    footer()
    st.stop()

result_key = f"result_{arch}_{file_key}"
with left:
    run = st.button("Analyze X-ray", type="primary", icon=":material/radiology:", width="stretch",
                    disabled=image is None)
    if uploaded is not None:
        st.button("Remove image", icon=":material/close:", width="stretch", on_click=remove_image)

if run and image is not None:
    with st.spinner("Analyzing the X-ray…"):
        try:
            ss[result_key] = predictor.predict(image)
        except Exception:  # noqa: BLE001
            log.exception("Inference failed")
            notice("The image could not be analyzed. Please try a different PNG or JPEG chest X-ray.", "error",
                   "Analysis failed.")

raw = ss.get(result_key) if image is not None else None
if raw is None:
    render(f'<p class="tx-bottom-note">{esc(RESULTS_NOTICE)}</p>')
    footer(notice_line=False)
    st.stop()

# ------------------------------------------------------------------ results
thr_list = custom_thr if custom_thr is not None else predictor.thresholds
scores = {lbl: raw.scores[lbl] for lbl in config.LABELS}
summary = build_summary(scores, dict(zip(config.LABELS, thr_list)))

section_title("Results")
render(f'<div class="tx-result-head"><h3>{esc(summary.headline)}</h3>'
       f'<span class="meta">{esc(config.ARCHITECTURES[arch]["display_name"])} · analyzed in {raw.inference_ms:,.0f} ms</span>'
       f'</div><p class="tx-quiet">{esc(MULTI_LABEL_NOTE)}</p>')
if summary.flagged:
    render_finding_cards(summary.flagged, scores)
else:
    notice(NONE_MESSAGE, "info")
with st.expander("View all 14 model outputs", icon=":material/list:"):
    render(all_outputs_html(scores, summary.flagged))

# ------------------------------------------------------------------ Grad-CAM
section_title("Visual explanation",
              "The heatmap highlights image regions that most influenced the model's output for the selected "
              "finding. It is not a confirmed location of disease.")
ranked = sorted(config.LABELS, key=scores.get, reverse=True)
names = [config.DISPLAY_NAMES[lbl] for lbl in ranked]
c1, c2 = st.columns([2, 1])
target = c1.selectbox("Finding to visualize", names, index=0)
alpha = c2.slider("Heatmap opacity", 0.1, 0.8, 0.45, 0.05)
label = ranked[names.index(target)]
cam_key = f"cam_{result_key}_{label}"
if cam_key not in ss:
    from src.gradcam import compute_gradcam

    with st.spinner(f"Generating the heatmap for {target}…"):
        try:
            ss[cam_key] = compute_gradcam(predictor.model, prepare_batch(image, predictor.image_size),
                                          config.LABELS.index(label))
        except Exception:  # noqa: BLE001 - keep the page usable; results above are unaffected
            log.exception("Grad-CAM failed")
            notice("The heatmap could not be generated for this finding. The results above are unaffected.",
                   "warning", "Heatmap unavailable.")

if cam_key in ss:
    from src.gradcam import overlay_heatmap

    heatmap, _ = ss[cam_key]
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    o1, o2 = st.columns(2, gap="medium")
    o1.image(image, caption="Original X-ray", width="stretch")
    o2.image(overlay, caption=f"Grad-CAM heatmap · {target}", width="stretch")
    if heatmap.max() == 0:
        notice("No region contributed noticeably to this finding's output, so the heatmap is blank.", "info")
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    d1, _ = st.columns([1, 2])
    d1.download_button("Download heatmap", buf.getvalue(), file_name=f"thoraxai_gradcam_{label}.png",
                       mime="image/png", icon=":material/download:", width="stretch")

st.button("Analyze another X-ray", icon=":material/restart_alt:", on_click=remove_image, key="reset_bottom")
render(f'<p class="tx-bottom-note">{esc(RESULTS_NOTICE)}</p>')
footer(notice_line=False)

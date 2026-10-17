"""System status view: genuine readiness checks, dataset location and setup steps."""

from __future__ import annotations

import os

import streamlit as st

import config
from src.ui import dataset_status_reason, esc, icon, notice, render


def _row(ok: bool, title: str, detail: str) -> str:
    color = "#067647" if ok else "#94A3B8"
    mark = icon("check" if ok else "info", 20)
    return (f'<div style="display:flex;gap:.75rem;align-items:flex-start;padding:.75rem 0;'
            f'border-bottom:1px solid #F1F5F9"><span style="color:{color}">{mark}</span>'
            f'<div><div style="font-weight:600;color:#0B1F3A">{title}</div>'
            f'<div style="font-size:.88rem;color:#64748B">{detail}</div></div></div>')


def render_system(status: dict) -> None:
    ds = status["dataset_paths"]
    names = ", ".join(config.ARCHITECTURES[m]["display_name"] for m in status["trained_models"]) or "none"
    rows = [
        _row(status["dataset_ready"], "Dataset", esc(dataset_status_reason(status))),
        _row(status["validation_report"], "Dataset validation",
             "Validation report available" if status["validation_report"] else "Not yet validated"),
        _row(bool(status["trained_models"]), "Analysis model", f"Trained: {esc(names)}"),
        _row(bool(status["evaluated_models"]), "Held-out evaluation",
             "Test metrics available" if status["evaluated_models"] else "Not yet evaluated"),
    ]
    render(f'<div class="tx-card" style="padding:.4rem 1.2rem">{"".join(rows)}</div>')

    source = ("environment variable THORAXAI_DATA_DIR" if os.environ.get("THORAXAI_DATA_DIR")
              else "local_config.json" if config.LOCAL_SETTINGS.get("data_dir") else "built-in default")
    details = {
        "Configured folder": ds["root"],
        "Configured by": source,
        "Metadata CSV": ds["metadata_csv"] or "not found",
        "Image folder": ds["image_root"] or "not found",
        "Images present": "yes" if ds.get("has_images") else "no",
        "Official split lists": "found" if ds["train_val_list"] and ds["test_list"] else "not found (patient-grouped split used)",
    }
    # Local filesystem paths are administrator information, kept out of the default view.
    with st.expander("Dataset location (administrator details)", icon=":material/folder:"):
        st.dataframe({"Item": list(details), "Value": [str(v) for v in details.values()]}, hide_index=True,
                     width="stretch")
    if status["dataset_ready"]:
        notice("Status is computed from the files on disk each time this page loads.", "success",
               "Dataset found.")
    else:
        notice(f"{esc(dataset_status_reason(status))}. Set <code>data_dir</code> in <code>local_config.json</code> "
               "(project root) to the extracted dataset folder, then refresh this page.", "warning",
               "Dataset not available.")

    with st.expander("Setup and maintenance commands (administrators)", icon=":material/terminal:"):
        st.markdown("Dataset location, in `local_config.json` at the project root:")
        st.code('{ "data_dir": "C:/path/to/NIH-ChestXray14" }', language="json")
        st.markdown("Validate, train, evaluate (Windows, from the project folder):")
        st.code(".venv\\Scripts\\python scripts\\validate_dataset.py\n"
                ".venv\\Scripts\\python scripts\\train_model.py --model densenet121 --mode development\n"
                ".venv\\Scripts\\python scripts\\evaluate_model.py --model densenet121", language="bash")
        st.markdown("Dataset source: [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)."
                    " See `README.md` for full instructions.")

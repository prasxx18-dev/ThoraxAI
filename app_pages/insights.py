"""Insights: model performance, dataset and system status (advanced information)."""

import streamlit as st

from src.ui import footer, page_header, page_setup, system_status
from src.views.dataset_view import render_dataset
from src.views.performance import render_performance
from src.views.system_view import render_system

page_setup("Insights")
page_header("Insights", "Model and data transparency",
            "How the analysis model performs on held-out data, what it was trained on, and the current system "
            "status.")

VIEWS = {"Model performance": "performance", "Dataset": "dataset", "System status": "system"}
choice = st.segmented_control("View", list(VIEWS), default="Model performance", key="insights_view",
                              label_visibility="collapsed") or "Model performance"
st.write("")

status = system_status()
if VIEWS[choice] == "performance":
    render_performance()
elif VIEWS[choice] == "dataset":
    render_dataset(status)
else:
    render_system(status)

footer()

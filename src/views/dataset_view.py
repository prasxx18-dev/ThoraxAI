"""Dataset view - verified counts from the local dataset and its validation report."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.ui import empty_state, esc, notice, stat_tiles
from src.utils import load_json_if_exists


@st.cache_data(show_spinner="Indexing images…")
def _sample_table(metadata_csv: str, image_root: str) -> pd.DataFrame:
    """Cached per dataset location, so a changed path is never served from a stale index."""
    from src.dataset import attach_image_paths, build_image_index, load_metadata

    df = load_metadata(Path(metadata_csv))
    index, _ = build_image_index(Path(image_root))
    df = attach_image_paths(df, index)
    return df.loc[df["exists"], ["path", "labels"]].reset_index(drop=True)


def render_dataset(status: dict) -> None:
    report = load_json_if_exists(config.RESULTS_DIR / "dataset_validation.json")
    if report is None:
        if status["dataset_ready"]:
            notice("The dataset was found but has not been validated yet. Validation counts every file, checks a "
                   "random sample of images for corruption and builds the patient-level splits.", "info")
            if st.button("Validate dataset now (sample integrity check)", type="primary"):
                from src.dataset import discover_dataset, save_validation_report, validate_dataset

                with st.spinner("Validating dataset — this can take a few minutes…"):
                    rep, _ = validate_dataset(discover_dataset(), verify="sample", sample_size=1000)
                    save_validation_report(rep, config.RESULTS_DIR)
                st.cache_data.clear()
                st.rerun()
        else:
            empty_state("database", "Dataset not configured",
                        "The dataset location has not been configured on this machine. See System status for "
                        "the folder that was checked.",
                        commands=["python scripts/validate_dataset.py --data-root <path-to-extracted-dataset>"])
        return

    st.caption(f"NIH ChestX-ray14 · report generated {report['generated_at']}")
    ic = report["integrity_check"]
    stat_tiles([
        ("Images", f"{report['image_files_found']:,}", f"{report['metadata_rows']:,} metadata rows"),
        ("Patients", f"{report['patients']:,}" if report.get("patients") else "n/a", ""),
        ("Missing images", f"{report['missing_images']:,}", ""),
        ("Corrupted", f"{ic['corrupted']:,}", f"of {ic['images_checked']:,} checked ({ic['mode']})"),
        ("Multi-label images", f"{report['labels']['multi_label_images']:,}", "two or more findings"),
    ])

    tabs = st.tabs(["Prevalence", "Co-occurrence", "Splits", "Sample images", "Data quality & bias"])
    labels = report["labels"]
    with tabs[0]:
        prev = pd.DataFrame({
            "Finding": [config.DISPLAY_NAMES[lbl] for lbl in config.LABELS],
            "Images": [labels["positives"][lbl] for lbl in config.LABELS],
            "Prevalence": [labels["prevalence"][lbl] for lbl in config.LABELS],
        }).sort_values("Images", ascending=False)
        st.altair_chart(alt.Chart(prev).mark_bar(color="#0F766E", cornerRadiusEnd=3).encode(
            x=alt.X("Images:Q"), y=alt.Y("Finding:N", sort="-x", title=None),
            tooltip=["Finding", "Images", alt.Tooltip("Prevalence:Q", format=".2%")]).properties(height=420),
            width="stretch")
        st.caption(f"Images with no listed finding: {labels['no_finding_images']:,} of {labels['num_images']:,}. "
                   "Strong imbalance motivates the weighted loss and PR-AUC reporting.")
        hist = pd.DataFrame({"Findings per image": list(labels["labels_per_image_hist"].keys()),
                             "Images": list(labels["labels_per_image_hist"].values())})
        st.altair_chart(alt.Chart(hist).mark_bar(color="#1E3A5F").encode(
            x="Findings per image:O", y="Images:Q").properties(height=220, title="Findings per image"),
            width="stretch")

    with tabs[1]:
        m = np.array(labels["cooccurrence"]["matrix"])
        names = [config.DISPLAY_NAMES[lbl] for lbl in labels["cooccurrence"]["labels"]]
        normalise = st.toggle("Normalise by row (P(column | row))", value=False)
        vals = m / np.maximum(np.diag(m)[:, None], 1) if normalise else m
        co = pd.DataFrame([{"Row": names[i], "Column": names[j], "Value": float(vals[i, j])}
                           for i in range(len(names)) for j in range(len(names)) if i != j])
        st.altair_chart(alt.Chart(co).mark_rect().encode(
            x=alt.X("Column:N", sort=names), y=alt.Y("Row:N", sort=names),
            color=alt.Color("Value:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=["Row", "Column", alt.Tooltip("Value:Q", format=".3f" if normalise else ".0f")],
        ).properties(height=480), width="stretch")
        st.caption("Diagonal omitted. Frequent pairs illustrate why a multi-label formulation is needed.")

    with tabs[2]:
        rows = [{"Split": s, "Images": v["images"], "Patients": v["patients"]} for s, v in report["splits"].items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        overlap = report.get("patient_overlap", {})
        if overlap:
            if any(overlap.values()):
                notice(f"Patient overlap detected between splits: {esc(overlap)}", "error")
            else:
                notice("No patient appears in more than one split (no patient-level leakage).", "success")
        st.caption(f"Split method: {report['split_method']}")
        pos = pd.DataFrame({s: v["positives"] for s, v in report["splits"].items()})
        pos.index = [config.DISPLAY_NAMES[i] for i in pos.index]
        with st.expander("Positive images per finding and split"):
            st.dataframe(pos, width="stretch")
        if report.get("unassigned_rows"):
            st.caption(f"{report['unassigned_rows']} metadata rows were in neither official split list.")

    with tabs[3]:
        if not status["dataset_ready"]:
            st.info("Dataset images are not available on this machine, so samples cannot be shown.")
        else:
            table = _sample_table(status["dataset_paths"]["metadata_csv"], status["dataset_paths"]["image_root"])
            c1, c2 = st.columns([3, 1])
            choice = c1.selectbox("Show images labelled with",
                                  ["No Finding"] + [config.DISPLAY_NAMES[x] for x in config.LABELS])
            seed = c2.number_input("Sample", 0, 10_000, 0, 1, help="Change to draw a different sample")
            if choice == "No Finding":
                subset = table[table["labels"].map(len) == 0]
            else:
                key = config.LABELS[[config.DISPLAY_NAMES[x] for x in config.LABELS].index(choice)]
                subset = table[table["labels"].map(lambda ls: key in ls)]
            if subset.empty:
                st.info("No locally available images carry this label.")
            else:
                pick = subset.sample(min(6, len(subset)), random_state=int(seed))
                cols = st.columns(3, gap="medium")
                for i, (_, row) in enumerate(pick.iterrows()):
                    caption = ", ".join(config.DISPLAY_NAMES[x] for x in row["labels"]) or "No Finding"
                    cols[i % 3].image(row["path"], caption=caption, width="stretch")
                st.caption("Captions show dataset labels only. Filenames and patient identifiers are hidden.")

    with tabs[4]:
        st.markdown("**Missing and corrupted files**")
        st.markdown(f"- Missing images (listed in metadata, not on disk): {report['missing_images']:,}\n"
                    f"- Image files on disk not listed in metadata: {report['image_files_not_in_metadata']:,}\n"
                    f"- Corrupted among {ic['images_checked']:,} checked: {ic['corrupted']}")
        if ic.get("corrupted_examples"):
            st.dataframe(pd.DataFrame(ic["corrupted_examples"]), width="stretch", hide_index=True)
        if report.get("unknown_label_tokens"):
            notice(f"Unknown label tokens ignored: {esc(report['unknown_label_tokens'])}", "warning")
        st.markdown("**Label uncertainty and bias**")
        st.markdown(
            "- Labels were text-mined from radiology reports by NLP; the dataset authors report imperfect accuracy, "
            "and independent reviews found substantial label noise for some findings.\n"
            "- A missing label does not prove a finding is absent ('No Finding' means none was mentioned).\n"
            "- All images come from one US hospital system; scanner type, patient population and follow-up imaging "
            "patterns may not generalise. Patients with many follow-up studies contribute many images.\n"
            "- Some findings (e.g. pneumothorax) can co-occur with treatment devices such as chest drains, which a "
            "model may learn as a shortcut.\n"
            f"- {report['label_policy']}")

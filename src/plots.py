"""Matplotlib figures for evaluation artifacts (headless 'Agg' backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

ACCENT = "#1f6f8b"
SECOND = "#e07a5f"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _draw_confusion(ax, label: str, m: dict) -> None:
    c = m["confusion"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > mat.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["Pred −", "Pred +"])
    ax.set_yticks([0, 1], ["True −", "True +"])
    ax.set_title(f"{config.DISPLAY_NAMES.get(label, label)}\n(thr={m['threshold']:.2f})", fontsize=9)


def plot_confusion_matrices(per_label: dict, out_dir: Path) -> list[Path]:
    """One small 2x2 matrix per finding plus a combined grid."""
    out_dir = Path(out_dir)
    paths = []
    fig_all, axes = plt.subplots(3, 5, figsize=(16, 10))
    for ax in axes.ravel()[len(per_label):]:
        ax.axis("off")
    for ax, (label, m) in zip(axes.ravel(), per_label.items()):
        _draw_confusion(ax, label, m)
        fig, single = plt.subplots(figsize=(3.4, 3.4))
        _draw_confusion(single, label, m)
        paths.append(_save(fig, out_dir / f"cm_{label}.png"))
    paths.insert(0, _save(fig_all, out_dir / "confusion_matrices_all.png"))
    return paths


def plot_curves(curves: dict, per_label: dict, roc_dir: Path, pr_dir: Path) -> list[Path]:
    """Combined ROC and PR plots (all findings on one axes each)."""
    paths = []
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, c in curves.items():
        auc = per_label[label]["roc_auc"]
        ax.plot(c["fpr"], c["tpr"], lw=1.2, label=f"{config.DISPLAY_NAMES[label]} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set(xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)",
           title="ROC curves (test split)")
    ax.legend(fontsize=7, loc="lower right")
    paths.append(_save(fig, Path(roc_dir) / "roc_all.png"))

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, c in curves.items():
        ap = per_label[label]["average_precision"]
        ax.plot(c["recall"], c["precision"], lw=1.2, label=f"{config.DISPLAY_NAMES[label]} ({ap:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test split)")
    ax.legend(fontsize=7, loc="upper right")
    paths.append(_save(fig, Path(pr_dir) / "pr_all.png"))
    return paths


def plot_per_label_bars(per_label: dict, path: Path) -> Path:
    labels = [config.DISPLAY_NAMES[lbl] for lbl in per_label]
    x = np.arange(len(labels))
    keys = [("roc_auc", "ROC-AUC", ACCENT), ("f1", "F1", SECOND), ("recall", "Recall", "#81b29a")]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.27
    for k, (key, name, color) in enumerate(keys):
        values = [per_label[lbl][key] if per_label[lbl][key] is not None else 0 for lbl in per_label]
        ax.bar(x + (k - 1) * width, values, width, label=name, color=color)
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Per-finding performance (test split)")
    ax.legend()
    return _save(fig, path)


def plot_history(history: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "auc", "Macro ROC-AUC (multi-label)")):
        if key in history:
            ax.plot(epochs, history[key], "o-", color=ACCENT, label="train")
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], "s-", color=SECOND, label="validation")
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    return _save(fig, path)


def plot_dataset_overview(report: dict, out_dir: Path) -> list[Path]:
    """Label prevalence bar chart and co-occurrence heatmap from a validation report."""
    import pandas as pd
    import seaborn as sns

    out_dir = Path(out_dir)
    labels = report["labels"]
    names = [config.DISPLAY_NAMES[lbl] for lbl in config.LABELS]
    prev = pd.DataFrame({"Finding": names, "Images": [labels["positives"][lbl] for lbl in config.LABELS]})
    prev = prev.sort_values("Images", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=prev, y="Finding", x="Images", color=ACCENT, ax=ax)
    ax.set_title(f"Images per finding (n={labels['num_images']:,} images on disk)")
    paths = [_save(fig, out_dir / "label_prevalence.png")]

    matrix = np.array(labels["cooccurrence"]["matrix"], dtype=float)
    np.fill_diagonal(matrix, np.nan)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(pd.DataFrame(matrix, index=names, columns=names), cmap="mako_r", square=True, ax=ax,
                cbar_kws={"label": "Images with both findings"})
    ax.set_title("Label co-occurrence (diagonal omitted)")
    paths.append(_save(fig, out_dir / "label_cooccurrence.png"))
    return paths

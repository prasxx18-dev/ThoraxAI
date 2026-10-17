"""Finding cards: name, model-estimated percentage, progress bar and a short explanation."""

from __future__ import annotations

from typing import Iterable, Mapping

import config
from src.findings_info import FINDINGS, all_references, get_info
from src.predict import to_percent
from src.ui import esc, render

PCT_LABEL = "Model-estimated likelihood"


def _bar(pct: float, muted: bool = False) -> str:
    return (f'<div class="tx-pbar" role="progressbar" aria-valuenow="{pct:.1f}" aria-valuemin="0" '
            f'aria-valuemax="100"><span class="{"muted" if muted else ""}" style="width:{pct:.1f}%"></span></div>')


def finding_card_html(label: str, score: float) -> str:
    """One result card. ``score`` is the model's [0, 1] output for this label."""
    pct = to_percent(score)
    info = get_info(label)
    return (
        '<article class="tx-fcard">'
        f'<div class="tx-fcard-head"><h4>{esc(info.name)}</h4>'
        f'<div class="tx-fcard-pct"><span class="v">{pct:.1f}%</span><span class="l">{PCT_LABEL}</span></div></div>'
        f"{_bar(pct)}"
        f'<div class="tx-fcard-h">What it means</div><p>{esc(info.meaning)}</p>'
        "</article>"
    )


def render_finding_cards(labels: Iterable[str], scores: Mapping[str, float]) -> None:
    render('<div class="tx-fgrid">' + "".join(finding_card_html(lbl, scores[lbl]) for lbl in labels) + "</div>")


def all_outputs_html(scores: Mapping[str, float], crossed: Iterable[str]) -> str:
    """Compact list of all 14 outputs as percentages, highest first (no thresholds, no raw decimals)."""
    crossed = set(crossed)
    rows = []
    for label in sorted(scores, key=scores.get, reverse=True):
        pct = to_percent(scores[label])
        rows.append(f'<div class="tx-orow"><span class="n">{esc(config.DISPLAY_NAMES[label])}</span>'
                    f"{_bar(pct, muted=label not in crossed)}<span class=\"p\">{pct:.1f}%</span></div>")
    return '<div class="tx-olist">' + "".join(rows) + "</div>"


def glossary_html() -> str:
    """All 14 short explanations (About page) - general information, no scores."""
    cards = "".join(
        f'<article class="tx-fcard"><div class="tx-fcard-head"><h4>{esc(i.name)}</h4></div>'
        f'<p>{esc(i.meaning)}</p></article>' for i in FINDINGS.values())
    return f'<div class="tx-fgrid">{cards}</div>'


def sources_html() -> str:
    items = "".join(f'<li><a href="{esc(u)}" target="_blank" rel="noopener noreferrer">{esc(t)}</a></li>'
                    for t, u in all_references())
    return f'<ul class="tx-sources">{items}</ul>'

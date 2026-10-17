"""ThoraxAI design system: tokens, global CSS and presentation components.

Only presentation lives here. Data loading, inference, Grad-CAM and metrics live
in the framework-independent modules (dataset, predict, gradcam, evaluate) so
they can be tested without Streamlit.

All HTML produced here is passed through ``_html`` which strips indentation and
blank lines - Streamlit's Markdown renderer would otherwise treat indented HTML
as a code block.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st

import config
from src.utils import load_json_if_exists

ASSETS = config.PROJECT_ROOT / "assets"

USAGE_NOTICE = (
    "Research use only. ThoraxAI outputs are experimental model scores, not diagnoses, and are not intended for "
    "independent medical diagnosis or treatment decisions."
)

# --------------------------------------------------------------------------- #
# Icons (24px line icons, drawn for ThoraxAI; stroke = currentColor)
# --------------------------------------------------------------------------- #
_ICON_PATHS = {
    "layers": '<path d="M12 3 3 7.5 12 12l9-4.5L12 3Z"/><path d="m3 12 9 4.5 9-4.5"/><path d="m3 16.5 9 4.5 9-4.5"/>',
    "eye": '<path d="M2.5 12S6 5 12 5s9.5 7 9.5 7-3.5 7-9.5 7-9.5-7-9.5-7Z"/><circle cx="12" cy="12" r="3"/>',
    "chart": '<path d="M4 4v16h16"/><path d="m8 14 3.5-3.5 3 3L20 8"/>',
    "upload": '<path d="M12 15V4"/><path d="m7 9 5-5 5 5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "scan": '<path d="M4 8V6a2 2 0 0 1 2-2h2"/><path d="M16 4h2a2 2 0 0 1 2 2v2"/><path d="M20 16v2a2 2 0 0 1-2 '
            '2h-2"/><path d="M8 20H6a2 2 0 0 1-2-2v-2"/><path d="M12 8v8"/><path d="M9 10.5c-1.5.5-2 2-2 3.5"/>'
            '<path d="M15 10.5c1.5.5 2 2 2 3.5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
    "alert": '<path d="M12 4 2.8 19.5h18.4L12 4Z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.5 2.5L16 9.5"/>',
    "x": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6"/><path d="m15 9-6 6"/>',
    "lock": '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
    "image": '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="10" r="1.6"/>'
             '<path d="m20.5 16-4.5-4.5L6 19.5"/>',
    "database": '<ellipse cx="12" cy="6" rx="7.5" ry="2.8"/><path d="M4.5 6v12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 '
                '7.5-2.8V6"/><path d="M4.5 12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M12 2.8v2.4M12 18.8v2.4M2.8 12h2.4M18.8 12h2.4M5.5 5.5l1.7 '
                '1.7M16.8 16.8l1.7 1.7M5.5 18.5l1.7-1.7M16.8 7.2l1.7-1.7"/>',
    "arrow": '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    "activity": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
}


def icon(name: str, size: int = 20, cls: str = "") -> str:
    """Inline SVG line icon."""
    return (f'<svg class="tx-icon {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_ICON_PATHS[name]}</svg>')


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root {
  --tx-navy: #0B1F3A; --tx-navy-2: #1E3A5F; --tx-slate: #475569; --tx-muted: #64748B; --tx-faint: #94A3B8;
  --tx-border: #E2E8F0; --tx-surface: #F8FAFC; --tx-surface-2: #F1F5F9; --tx-teal: #0F766E; --tx-teal-2: #0D9488;
  --tx-teal-50: #F0FDFA; --tx-teal-100: #CCFBF1; --tx-amber: #B45309; --tx-amber-50: #FFFBEB; --tx-amber-100: #FDE68A;
  --tx-red: #B42318; --tx-red-50: #FEF3F2; --tx-green: #067647; --tx-green-50: #ECFDF3;
  --tx-radius: 14px; --tx-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.05);
}
/* Layout ----------------------------------------------------------------- */
.block-container { max-width: 1160px; padding-top: 5.2rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: rgba(255,255,255,.94); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--tx-border); }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
h1, h2, h3, h4 { color: var(--tx-navy); letter-spacing: -0.02em; }
p, li { color: var(--tx-slate); }
/* Buttons ---------------------------------------------------------------- */
[data-testid^="stBaseButton"] { font-weight: 600; border-radius: 10px; min-height: 2.6rem; }
[data-testid^="stBaseButton"] p, [data-testid="stPageLink"] p, [data-testid="stDownloadButton"] p { color: inherit; }
[data-testid="stBaseButton-primary"] { box-shadow: 0 1px 2px rgba(15,118,110,.25); }
/* Upload dropzone --------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed #CBD5E1; background: var(--tx-surface);
  border-radius: var(--tx-radius); padding: 1.6rem 1.2rem; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--tx-teal-2); background: var(--tx-teal-50); }
/* Tabs / segmented ------------------------------------------------------- */
[data-testid="stHeaderActionElements"] { display: none; }
[data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--tx-border); }
[data-baseweb="tab"] { font-weight: 500; }
[data-testid="stExpander"] details { border-radius: 12px; border-color: var(--tx-border); }
[data-testid="stImageCaption"] { color: var(--tx-muted); }
/* Shared components ------------------------------------------------------ */
.tx-icon { flex: none; }
.tx-eyebrow { color: var(--tx-teal); font-weight: 600; font-size: .78rem; letter-spacing: .08em;
  text-transform: uppercase; margin: 0 0 .5rem; }
.tx-page-header { margin: 0 0 1.4rem; }
.tx-page-header h1 { font-size: clamp(1.7rem, 3vw, 2.25rem); font-weight: 700; margin: 0; padding: 0; line-height: 1.15; }
.tx-page-header p { margin: .5rem 0 0; font-size: 1.04rem; max-width: 46rem; }
.tx-section-title { margin: 2.2rem 0 1rem; }
.tx-section-title h2 { font-size: 1.35rem; font-weight: 650; margin: 0; padding: 0; }
.tx-section-title p { margin: .3rem 0 0; color: var(--tx-muted); }
.tx-card { background: #fff; border: 1px solid var(--tx-border); border-radius: var(--tx-radius);
  box-shadow: var(--tx-shadow); padding: 1.35rem 1.4rem; height: 100%; }
.tx-card h3 { font-size: 1.05rem; font-weight: 650; margin: .9rem 0 .35rem; padding: 0; }
.tx-card p { margin: 0; font-size: .95rem; line-height: 1.55; }
.tx-icon-badge { width: 40px; height: 40px; border-radius: 11px; display: inline-flex; align-items: center;
  justify-content: center; background: var(--tx-teal-50); color: var(--tx-teal); border: 1px solid var(--tx-teal-100); }
.tx-notice { display: flex; gap: .7rem; align-items: flex-start; border-radius: 12px; padding: .8rem 1rem;
  font-size: .92rem; line-height: 1.5; margin: .2rem 0 1rem; border: 1px solid; }
.tx-notice p { margin: 0; color: inherit; }
.tx-notice strong { color: inherit; }
.tx-notice.info { background: var(--tx-surface); border-color: var(--tx-border); color: var(--tx-slate); }
.tx-notice.info .tx-icon { color: var(--tx-teal); }
.tx-notice.warning { background: var(--tx-amber-50); border-color: var(--tx-amber-100); color: #7A2E0E; }
.tx-notice.success { background: var(--tx-green-50); border-color: #ABEFC6; color: #085D3A; }
.tx-notice.error { background: var(--tx-red-50); border-color: #FECDCA; color: #912018; }
.tx-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .9rem; margin: .3rem 0 1rem; }
.tx-stat { background: #fff; border: 1px solid var(--tx-border); border-radius: 12px; padding: .95rem 1.05rem; }
.tx-stat .label { color: var(--tx-muted); font-size: .8rem; font-weight: 500; }
.tx-stat .value { color: var(--tx-navy); font-size: 1.45rem; font-weight: 650; margin-top: .2rem; letter-spacing: -.01em; }
.tx-stat .sub { color: var(--tx-faint); font-size: .78rem; margin-top: .15rem; }
.tx-chip { display: inline-flex; align-items: center; gap: .35rem; padding: .22rem .65rem; border-radius: 999px;
  margin: 0 .35rem .4rem 0; font-size: .82rem; font-weight: 500; border: 1px solid var(--tx-border);
  background: var(--tx-surface); color: var(--tx-slate); }
.tx-chip.accent { background: var(--tx-teal-50); border-color: var(--tx-teal-100); color: var(--tx-teal); }
.tx-empty { text-align: center; border: 1px dashed #CBD5E1; border-radius: var(--tx-radius); background: var(--tx-surface);
  padding: 2.2rem 1.5rem; margin: .5rem 0 1rem; }
.tx-empty .tx-icon-badge { margin-bottom: .8rem; }
.tx-empty h3 { font-size: 1.08rem; margin: 0 0 .35rem; padding: 0; }
.tx-empty p { margin: 0 auto; max-width: 34rem; font-size: .95rem; }
/* Score table ------------------------------------------------------------ */
.tx-scores { width: 100%; border-collapse: collapse; font-size: .93rem; border: 0 !important; margin: 0; }
.tx-scores th, .tx-scores td, .tx-scores tr { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important;
  background: transparent !important; }
.tx-scores th { text-align: left; color: var(--tx-muted); font-weight: 500; font-size: .78rem; padding: .5rem .6rem;
  border-bottom: 1px solid var(--tx-border); }
.tx-scores td { padding: .55rem .6rem; border-bottom: 1px solid var(--tx-surface-2); color: var(--tx-navy); vertical-align: middle; }
.tx-scores tr:last-child td { border-bottom: none; }
.tx-scores .name { font-weight: 500; white-space: nowrap; }
.tx-scores .num { font-variant-numeric: tabular-nums; text-align: right; width: 4.2rem; color: var(--tx-slate); }
.tx-bar { position: relative; height: 8px; border-radius: 999px; background: var(--tx-surface-2); min-width: 120px; }
.tx-bar .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: #CBD5E1; }
.tx-bar .fill.hi { background: var(--tx-teal-2); }
.tx-bar .thr { position: absolute; top: -4px; width: 2px; height: 16px; background: var(--tx-navy-2); border-radius: 1px; }
.tx-pill { display: inline-block; font-size: .74rem; font-weight: 600; padding: .12rem .5rem; border-radius: 999px; white-space: nowrap; }
.tx-pill.hi { background: var(--tx-teal-50); color: var(--tx-teal); border: 1px solid var(--tx-teal-100); }
.tx-pill.lo { background: var(--tx-surface); color: var(--tx-faint); border: 1px solid var(--tx-border); }
/* Finding cards ---------------------------------------------------------- */
.tx-fgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: .85rem; margin: .4rem 0 1rem; }
.tx-fcard { background: #fff; border: 1px solid var(--tx-border); border-radius: var(--tx-radius);
  box-shadow: var(--tx-shadow); padding: 1rem 1.1rem 1.05rem; }
.tx-fcard-head { display: flex; justify-content: space-between; align-items: flex-start; gap: .75rem; }
.tx-fcard h4 { font-size: 1.02rem; font-weight: 650; margin: 0; padding: 0; color: var(--tx-navy); }
.tx-fcard-pct { text-align: right; line-height: 1.1; flex: none; }
.tx-fcard-pct .v { display: block; font-size: 1.35rem; font-weight: 700; color: var(--tx-teal);
  font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.tx-fcard-pct .l { display: block; font-size: .7rem; color: var(--tx-muted); margin-top: .15rem; }
.tx-fcard-h { font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
  color: var(--tx-muted); margin: .8rem 0 .25rem; }
.tx-fcard p { margin: 0; font-size: .92rem; line-height: 1.55; color: var(--tx-slate); }
.tx-fcard .tx-fcard-head + p { margin-top: .5rem; }
.tx-pbar { height: 6px; border-radius: 999px; background: var(--tx-surface-2); overflow: hidden; margin-top: .7rem; }
.tx-pbar span { display: block; height: 100%; border-radius: 999px; background: var(--tx-teal-2); }
.tx-pbar span.muted { background: #CBD5E1; }
.tx-olist { display: grid; gap: .55rem; padding: .3rem .1rem; }
.tx-orow { display: grid; grid-template-columns: minmax(130px, 190px) 1fr 3.6rem; gap: .8rem; align-items: center; }
.tx-orow .n { font-size: .9rem; color: var(--tx-navy); }
.tx-orow .p { font-size: .88rem; text-align: right; color: var(--tx-slate); font-variant-numeric: tabular-nums; }
.tx-orow .tx-pbar { margin-top: 0; }
.tx-result-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: .4rem 1rem;
  margin: 0 0 .2rem; }
.tx-result-head h3 { font-size: 1.15rem; margin: 0; padding: 0; }
.tx-result-head .meta { color: var(--tx-muted); font-size: .85rem; }
.tx-quiet { color: var(--tx-muted); font-size: .88rem; margin: 0 0 .6rem; }
.tx-bottom-note { color: var(--tx-muted); font-size: .84rem; text-align: center; border-top: 1px solid var(--tx-border);
  padding: 1rem .5rem 0; margin: 2rem 0 0; }
.tx-sources { margin: .2rem 0 0; padding-left: 1.1rem; }
.tx-sources li { font-size: .86rem; margin: .15rem 0; }
.tx-sources a { color: var(--tx-teal) !important; }
@media (max-width: 640px) { .tx-orow { grid-template-columns: 1fr 3.4rem; } .tx-orow .tx-pbar { display: none; } }
/* Landing ---------------------------------------------------------------- */
.tx-hero-title { font-size: clamp(2.3rem, 4.6vw, 3.5rem); line-height: 1.06; font-weight: 700; color: var(--tx-navy);
  letter-spacing: -0.035em; margin: .2rem 0 1rem; }
.tx-hero-title span { color: var(--tx-teal); }
.tx-hero-lead { font-size: 1.12rem; line-height: 1.6; color: var(--tx-slate); max-width: 33rem; margin: 0 0 1.6rem; }
.tx-hero-meta { display: flex; flex-wrap: wrap; gap: 1.1rem; margin-top: 1.2rem; color: var(--tx-muted); font-size: .86rem; }
.tx-hero-meta span { display: inline-flex; align-items: center; gap: .4rem; }
.tx-hero-meta .tx-icon { color: var(--tx-teal); }
.tx-preview { width: 100%; height: auto; display: block; border-radius: 18px; box-shadow: 0 18px 48px rgba(11,31,58,.12); }
.tx-preview-caption { text-align: center; color: var(--tx-faint); font-size: .78rem; margin-top: .6rem; }
.tx-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: .2rem 0 1rem; }
.tx-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
.tx-step { display: flex; gap: .8rem; align-items: flex-start; }
.tx-step .n { flex: none; width: 30px; height: 30px; border-radius: 50%; background: var(--tx-navy); color: #fff;
  display: inline-flex; align-items: center; justify-content: center; font-weight: 600; font-size: .85rem; }
.tx-step h4 { margin: .15rem 0 .2rem; font-size: 1rem; padding: 0; }
.tx-step p { margin: 0; font-size: .92rem; }
/* Footer ----------------------------------------------------------------- */
.tx-footer { margin-top: 3.5rem; padding: 1.6rem 0 .5rem; border-top: 1px solid var(--tx-border);
  display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; justify-content: space-between; }
.tx-footer .brand { font-weight: 700; color: var(--tx-navy); letter-spacing: -.01em; }
.tx-footer .brand span { color: var(--tx-teal); }
.tx-footer nav { display: flex; gap: 1.3rem; flex-wrap: wrap; }
.tx-footer a { color: var(--tx-slate) !important; text-decoration: none; font-size: .88rem; }
.tx-footer a:hover { color: var(--tx-teal) !important; }
.tx-footer .fine { width: 100%; color: var(--tx-faint); font-size: .8rem; margin: 0; }
@media (max-width: 640px) {
  .block-container { padding-top: 4.4rem; }
  .tx-card { padding: 1.1rem; }
  .tx-scores .bar-col { display: none; }
}
</style>
"""


def _html(markup: str) -> str:
    """Strip indentation/blank lines so Markdown never turns HTML into a code block."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def render(markup: str) -> None:
    st.markdown(_html(markup), unsafe_allow_html=True)


def esc(text: object) -> str:
    return html.escape(str(text))


# --------------------------------------------------------------------------- #
# Page scaffolding
# --------------------------------------------------------------------------- #
def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · ThoraxAI", page_icon=str(ASSETS / "icon.png"), layout="wide",
                       initial_sidebar_state="collapsed")
    st.logo(str(ASSETS / "logo.svg"), size="large")
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    render(f'<div class="tx-page-header"><div class="tx-eyebrow">{eyebrow}</div><h1>{title}</h1>{sub}</div>')


def section_title(title: str, subtitle: str = "", anchor: Optional[str] = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    anchor_attr = f' id="{anchor}"' if anchor else ""
    render(f'<div class="tx-section-title"{anchor_attr}><h2>{title}</h2>{sub}</div>')


_NOTICE_ICONS = {"info": "info", "warning": "alert", "success": "check", "error": "x"}


def notice(body: str, kind: str = "info", title: str = "") -> None:
    """Inline banner. ``body`` may contain simple HTML; callers escape user data."""
    head = f"<strong>{title}</strong> " if title else ""
    render(f'<div class="tx-notice {kind}">{icon(_NOTICE_ICONS[kind], 18)}<p>{head}{body}</p></div>')


def usage_notice() -> None:
    notice(USAGE_NOTICE, "info")


def stat_tiles(items: Iterable[tuple[str, str, str]]) -> None:
    """Responsive grid of (label, value, sub) tiles."""
    tiles = "".join(
        f'<div class="tx-stat"><div class="label">{esc(lbl)}</div><div class="value">{esc(val)}</div>'
        + (f'<div class="sub">{esc(sub)}</div>' if sub else "") + "</div>"
        for lbl, val, sub in items
    )
    render(f'<div class="tx-stats">{tiles}</div>')


def feature_card(icon_name: str, title: str, body: str) -> str:
    return (f'<div class="tx-card"><span class="tx-icon-badge">{icon(icon_name, 21)}</span>'
            f"<h3>{title}</h3><p>{body}</p></div>")


def feature_grid(cards: Iterable[str]) -> None:
    """Equal-height responsive grid of ``feature_card`` HTML strings."""
    render(f'<div class="tx-grid">{"".join(cards)}</div>')


def chips(labels: Iterable[str], accent: bool = False) -> str:
    cls = "tx-chip accent" if accent else "tx-chip"
    return "".join(f'<span class="{cls}">{esc(config.DISPLAY_NAMES.get(lbl, lbl))}</span>' for lbl in labels)


def empty_state(icon_name: str, title: str, body: str, commands: Optional[list[str]] = None) -> None:
    render(f'<div class="tx-empty"><span class="tx-icon-badge">{icon(icon_name, 21)}</span>'
           f"<h3>{title}</h3><p>{body}</p></div>")
    if commands:
        with st.expander("Setup commands (for administrators)"):
            st.code("\n".join(commands), language="bash")


def score_table(rows: list[tuple[str, float, float]]) -> str:
    """HTML table of (label, score, threshold) rows, already sorted by the caller.

    Bars show the raw model score on a 0-1 scale; the dark tick marks the
    finding's review threshold. Teal = at or above threshold. No percentages
    are shown, because scores are not calibrated probabilities.
    """
    body = []
    for label, score, thr in rows:
        hi = score >= thr
        width = max(0.0, min(1.0, score)) * 100
        body.append(
            f'<tr><td class="name">{esc(config.DISPLAY_NAMES.get(label, label))}</td>'
            f'<td class="bar-col"><div class="tx-bar"><div class="fill{" hi" if hi else ""}" style="width:{width:.1f}%">'
            f'</div><div class="thr" style="left:calc({thr * 100:.1f}% - 1px)"></div></div></td>'
            f'<td class="num">{score:.3f}</td>'
            f'<td><span class="tx-pill {"hi" if hi else "lo"}">{"Above threshold" if hi else "Below"}</span></td></tr>'
        )
    return ('<table class="tx-scores"><thead><tr><th>Finding</th><th class="bar-col">Model score '
            '(tick = review threshold)</th><th class="num">Score</th><th>Status</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def footer(notice_line: bool = True) -> None:
    """Site footer. Pages that already show their own notice pass ``notice_line=False`` to avoid repetition."""
    fine = f'<p class="fine">{USAGE_NOTICE}</p>' if notice_line else ""
    render(
        '<footer class="tx-footer"><div class="brand">Thorax<span>AI</span></div>'
        '<nav><a href="/about" target="_self">About</a><a href="/about#privacy" target="_self">Privacy</a>'
        '<a href="/about#intended-use" target="_self">Intended use</a>'
        f'<a href="/insights" target="_self">Insights</a></nav>{fine}</footer>'
    )


# --------------------------------------------------------------------------- #
# Status (unchanged behaviour: re-reads local_config.json on every call)
# --------------------------------------------------------------------------- #
def system_status() -> dict:
    """Readiness checks used by the pages.

    ``config.refresh_paths()`` re-reads ``local_config.json`` and the
    environment on every call, so a running server notices a newly configured
    dataset location without a restart. The filesystem checks are cached for
    15 s, keyed on the resolved paths, so a path change takes effect at once.
    """
    paths = config.refresh_paths()
    return _system_status(*(str(paths[k]) for k in ("DATA_DIR", "METADATA_CSV", "IMAGE_ROOT", "MODELS_DIR",
                                                     "RESULTS_DIR")))


@st.cache_data(ttl=15, show_spinner=False)
def _system_status(data_dir: str, metadata_csv: str, image_root: str, models_dir: str, results_dir: str) -> dict:
    from src.dataset import discover_dataset
    from src.predict import available_models

    paths = discover_dataset(Path(data_dir), None if metadata_csv == "None" else Path(metadata_csv),
                             None if image_root == "None" else Path(image_root))
    results = Path(results_dir)
    return {
        "dataset_paths": paths.to_dict(),
        "dataset_ready": paths.is_ready,
        "validation_report": (results / "dataset_validation.json").is_file(),
        "trained_models": available_models(Path(models_dir)),
        "evaluated_models": [a for a in config.ARCHITECTURES if (results / a / "test_metrics.json").is_file()],
    }


def dataset_status_reason(status: dict) -> str:
    """Human-readable reason derived from the real filesystem checks."""
    ds = status["dataset_paths"]
    if status["dataset_ready"]:
        return "Metadata CSV and images found"
    if not Path(ds["root"]).is_dir():
        return "Configured folder does not exist"
    if not ds["metadata_csv"]:
        return "Data_Entry_2017*.csv not found"
    return "No PNG/JPEG images found"


def model_dir(arch: str) -> Path:
    return config.MODELS_DIR / arch


def load_model_config(arch: str) -> dict:
    return load_json_if_exists(model_dir(arch) / "model_config.json") or {}

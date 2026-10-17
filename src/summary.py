"""Results headline built only from the real model outputs (pure function, unit-testable).

The wording never upgrades a score to a diagnosis and never calls an image normal:
when nothing crosses a threshold it says exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import config

MULTI_LABEL_NOTE = ("Each finding is assessed independently, so more than one finding can appear for the same "
                    "image.")
NONE_MESSAGE = "No findings crossed the current display threshold."


@dataclass(frozen=True)
class ResultSummary:
    status: str  # "flagged" or "none_flagged"
    headline: str
    flagged: list[str]  # labels at/above threshold, highest score first
    top_label: str
    top_score: float


def build_summary(scores: Mapping[str, float], thresholds: Mapping[str, float]) -> ResultSummary:
    """Summarise which findings crossed their display threshold. ``scores``/``thresholds``: label -> value."""
    labels = [lbl for lbl in config.LABELS if lbl in scores]
    if not labels or len(labels) != len(scores) or any(lbl not in thresholds for lbl in labels):
        raise ValueError("scores and thresholds must cover the same known labels")
    ranked = sorted(labels, key=lambda lbl: scores[lbl], reverse=True)
    flagged = [lbl for lbl in ranked if scores[lbl] >= thresholds[lbl]]
    if flagged:
        n = len(flagged)
        headline = f"{n} finding{'s' if n != 1 else ''} crossed the display threshold"
        status = "flagged"
    else:
        headline, status = NONE_MESSAGE, "none_flagged"
    return ResultSummary(status, headline, flagged, ranked[0], float(scores[ranked[0]]))

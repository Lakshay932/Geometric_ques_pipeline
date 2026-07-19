"""Difficulty tagging for fold_punch records (FR-11).

Derived from fold count, punch count, whether a diagonal fold is involved,
and the average subtlety of the distractor rules used — not hand-labeled.
"""
from __future__ import annotations

from .geometry import Axis

# 0 = obvious tell (wrong count), 1 = very subtle (looks nearly correct).
RULE_SUBTLETY: dict[str, float] = {
    "missing_hole": 0.2,
    "extra_hole": 0.2,
    "wrong_symmetry_axis": 0.6,
    "mirrored_wrong": 0.5,
    "shifted_hole": 0.8,
}
_DEFAULT_SUBTLETY = 0.5


def compute_difficulty(
    fold_count: int,
    punch_count: int,
    axis_sequence: list[Axis],
    distractor_rules: list[str],
) -> int:
    subtlety_scores = [RULE_SUBTLETY.get(rule, _DEFAULT_SUBTLETY) for rule in distractor_rules]
    avg_subtlety = sum(subtlety_scores) / len(subtlety_scores) if subtlety_scores else _DEFAULT_SUBTLETY

    raw = (
        fold_count
        + (punch_count - 1) * 0.3
        + (0.5 if "diagonal" in axis_sequence else 0.0)
        + avg_subtlety * 1.5
    )
    return int(min(5, max(1, round(raw))))

from __future__ import annotations

from typing import Literal

from .geometry import MirrorAxis

AXIS_SUBTLETY: dict[str, float] = {
    "vertical": 0.2,
    "horizontal": 0.2,
    "diagonal_main": 0.7,
    "diagonal_anti": 0.7,
}
RULE_SUBTLETY: dict[str, float] = {
    "identical_copy": 0.1,
    "rotated_180": 0.5,
    "wrong_axis": 0.6,
    "shifted_reflection": 0.8,
}
_DEFAULT_SUBTLETY = 0.5

RULE_SUBTLETY.update(
    {
        "rotated_90": 0.3,
        "scaled_reflection": 0.6,
        "glide_reflection": 0.85,
        "partial_reflection": 0.9,
    }
)

RULE_TIER: dict[str, int] = {
    "identical_copy": 1,
    "rotated_180": 2,
    "wrong_axis": 2,
    "shifted_reflection": 3,
    "rotated_90": 1,
    "scaled_reflection": 2,
    "glide_reflection": 3,
    "partial_reflection": 3,
}
_DEFAULT_TIER = 2
HIGH_DIFFICULTY_THRESHOLD = 4
MIN_TIER3_DISTRACTORS_AT_HIGH_DIFFICULTY = 2

AXIS_WEIGHT = 2.0
SUBTLETY_WEIGHT = 2.0
_BASE = 0.3

PROCEDURAL_FIGURE_BONUS = 1.2

AXIS_OFFSET_WEIGHT = 4.0

def compute_difficulty(
    axis: MirrorAxis,
    distractor_rules: list[str],
    figure_source: Literal["hand_designed", "procedural"] = "hand_designed",
    axis_offset: float = 0.0,
) -> int:
    axis_subtlety = AXIS_SUBTLETY.get(axis, _DEFAULT_SUBTLETY)
    rule_scores = [RULE_SUBTLETY.get(rule, _DEFAULT_SUBTLETY) for rule in distractor_rules]
    avg_rule_subtlety = sum(rule_scores) / len(rule_scores) if rule_scores else _DEFAULT_SUBTLETY

    raw = (
        _BASE
        + axis_subtlety * AXIS_WEIGHT
        + avg_rule_subtlety * SUBTLETY_WEIGHT
        + (PROCEDURAL_FIGURE_BONUS if figure_source == "procedural" else 0.0)
        + abs(axis_offset) * AXIS_OFFSET_WEIGHT
    )
    return int(min(5, max(1, round(raw))))

def rule_tier(rule: str) -> int:
    return RULE_TIER.get(rule, _DEFAULT_TIER)

def meets_tier3_requirement(difficulty: int, distractor_rules: list[str]) -> bool:
    if difficulty < HIGH_DIFFICULTY_THRESHOLD:
        return True
    tier3_count = sum(1 for rule in distractor_rules if rule_tier(rule) == 3)
    return tier3_count >= MIN_TIER3_DISTRACTORS_AT_HIGH_DIFFICULTY

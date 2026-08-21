from __future__ import annotations

from typing import Literal

from .geometry import RotationDirection

STEP_SUBTLETY: dict[int, float] = {
    30: 0.8,
    45: 0.6,
    60: 0.4,
    90: 0.2,
    120: 0.1,
}
_DEFAULT_STEP_SUBTLETY = 0.5

RULE_SUBTLETY: dict[str, float] = {
    "stale_repeat": 0.1,
    "wrong_direction": 0.5,
    "reflected_instead": 0.5,
    "wrong_step_size": 0.6,
    "skipped_a_step": 0.9,
}
_DEFAULT_RULE_SUBTLETY = 0.5

RULE_TIER: dict[str, int] = {
    "stale_repeat": 1,
    "wrong_direction": 2,
    "reflected_instead": 2,
    "wrong_step_size": 2,
    "skipped_a_step": 3,
}
_DEFAULT_TIER = 2
HIGH_DIFFICULTY_THRESHOLD = 4
MIN_TIER3_DISTRACTORS_AT_HIGH_DIFFICULTY = 1

_BASE = 0.3
STEP_WEIGHT = 2.0
SUBTLETY_WEIGHT = 2.0

SEQUENCE_LENGTH_4_BONUS = 0.6

PROCEDURAL_FIGURE_BONUS = 0.8

def compute_difficulty(
    step_degrees: int,
    sequence_length: int,
    direction: RotationDirection,
    distractor_rules: list[str],
    figure_source: Literal["hand_designed", "procedural"] = "hand_designed",
) -> int:
    del direction
    step_subtlety = STEP_SUBTLETY.get(step_degrees, _DEFAULT_STEP_SUBTLETY)
    rule_scores = [RULE_SUBTLETY.get(rule, _DEFAULT_RULE_SUBTLETY) for rule in distractor_rules]
    avg_rule_subtlety = sum(rule_scores) / len(rule_scores) if rule_scores else _DEFAULT_RULE_SUBTLETY

    raw = (
        _BASE
        + step_subtlety * STEP_WEIGHT
        + avg_rule_subtlety * SUBTLETY_WEIGHT
        + (SEQUENCE_LENGTH_4_BONUS if sequence_length >= 4 else 0.0)
        + (PROCEDURAL_FIGURE_BONUS if figure_source == "procedural" else 0.0)
    )
    return int(min(5, max(1, round(raw))))

def rule_tier(rule: str) -> int:
    return RULE_TIER.get(rule, _DEFAULT_TIER)

def meets_tier3_requirement(difficulty: int, distractor_rules: list[str]) -> bool:
    if difficulty < HIGH_DIFFICULTY_THRESHOLD:
        return True
    tier3_count = sum(1 for rule in distractor_rules if rule_tier(rule) == 3)
    return tier3_count >= MIN_TIER3_DISTRACTORS_AT_HIGH_DIFFICULTY

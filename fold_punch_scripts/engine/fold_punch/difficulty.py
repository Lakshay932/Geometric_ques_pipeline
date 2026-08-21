from __future__ import annotations

from .geometry import Axis

RULE_SUBTLETY: dict[str, float] = {
    "missing_hole": 0.2,
    "extra_hole": 0.2,
    "wrong_symmetry_axis": 0.6,
    "mirrored_wrong": 0.5,
    "shifted_hole": 0.8,
}
_DEFAULT_SUBTLETY = 0.5

RULE_SUBTLETY.update(
    {
        "single_hole_wrong_side": 0.9,
        "rotated_90": 0.3,
        "wrong_diagonal_variant": 0.7,
        "phantom_diagonal_fold": 0.3,
        "last_fold_axis_swapped": 0.7,
    }
)

RULE_TIER: dict[str, int] = {
    "missing_hole": 1,
    "extra_hole": 1,
    "wrong_symmetry_axis": 2,
    "mirrored_wrong": 2,
    "shifted_hole": 3,
    "single_hole_wrong_side": 3,
    "rotated_90": 1,
    "wrong_diagonal_variant": 2,
    "phantom_diagonal_fold": 1,
    "last_fold_axis_swapped": 2,
}
_DEFAULT_TIER = 2
HIGH_DIFFICULTY_THRESHOLD = 4
MIN_TIER3_DISTRACTORS_AT_HIGH_DIFFICULTY = 2

PUNCH_COUNT_WEIGHT = 0.3
DIAGONAL_FOLD_BONUS = 0.5
SUBTLETY_WEIGHT = 1.5

NON_CIRCULAR_SHAPE_BONUS = 0.3

EDGE_PUNCH_BONUS = 0.4

def compute_difficulty(
    fold_count: int,
    punch_count: int,
    axis_sequence: list[Axis],
    distractor_rules: list[str],
    punch_kinds: list[str] | None = None,
    has_edge_punch: bool = False,
) -> int:
    subtlety_scores = [RULE_SUBTLETY.get(rule, _DEFAULT_SUBTLETY) for rule in distractor_rules]
    avg_subtlety = sum(subtlety_scores) / len(subtlety_scores) if subtlety_scores else _DEFAULT_SUBTLETY
    has_non_circular_shape = bool(punch_kinds) and any(kind != "circle" for kind in punch_kinds)

    raw = (
        fold_count
        + (punch_count - 1) * PUNCH_COUNT_WEIGHT
        + (DIAGONAL_FOLD_BONUS if "diagonal" in axis_sequence else 0.0)
        + avg_subtlety * SUBTLETY_WEIGHT
        + (NON_CIRCULAR_SHAPE_BONUS if has_non_circular_shape else 0.0)
        + (EDGE_PUNCH_BONUS if has_edge_punch else 0.0)
    )
    return int(min(5, max(1, round(raw))))

def rule_tier(rule: str) -> int:
    return RULE_TIER.get(rule, _DEFAULT_TIER)

def meets_tier3_requirement(difficulty: int, distractor_rules: list[str]) -> bool:
    if difficulty < HIGH_DIFFICULTY_THRESHOLD:
        return True
    tier3_count = sum(1 for rule in distractor_rules if rule_tier(rule) == 3)
    return tier3_count >= MIN_TIER3_DISTRACTORS_AT_HIGH_DIFFICULTY

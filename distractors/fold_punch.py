"""Distractor generator for fold_punch (FR-5): exactly 3 wrong options, each
breaking one specific, tagged rule.

Rule pool (subtlety scores in engine/fold_punch/difficulty.py):
- missing_hole: drop one correct hole.
- extra_hole: add an extra hole at the point-symmetric position through center.
- wrong_symmetry_axis: unfold while skipping the mirror step of one fold —
  the classic "forgot to unfold one layer" mistake.
- mirrored_wrong: reflect the whole correct pattern across the vertical center.
- shifted_hole: shift every hole by the same small offset (systematic
  mis-punch), still inside the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.fold_punch.geometry import FoldStep, Point, unfold_points

RULE_POOL = [
    "missing_hole",
    "extra_hole",
    "wrong_symmetry_axis",
    "mirrored_wrong",
    "shifted_hole",
]


@dataclass
class Distractor:
    points: list[Point]
    rule: str


def _round_all(points: list[Point], ndigits: int = 6) -> frozenset[Point]:
    return frozenset((round(x, ndigits), round(y, ndigits)) for x, y in points)


def _missing_hole(answer_points: list[Point], rng: np.random.Generator) -> list[Point] | None:
    if len(answer_points) < 2:
        return None
    idx = int(rng.integers(0, len(answer_points)))
    return [p for i, p in enumerate(answer_points) if i != idx]


def _extra_hole(answer_points: list[Point], rng: np.random.Generator) -> list[Point] | None:
    base = answer_points[int(rng.integers(0, len(answer_points)))]
    extra = (1.0 - base[0], 1.0 - base[1])
    existing = {(round(x, 6), round(y, 6)) for x, y in answer_points}
    if (round(extra[0], 6), round(extra[1], 6)) in existing:
        extra = (min(0.95, max(0.05, base[0] + 0.15)), base[1])
    return list(answer_points) + [extra]


def _wrong_symmetry_axis(
    punch_points: list[Point], steps: list[FoldStep], rng: np.random.Generator
) -> list[Point] | None:
    if not steps:
        return None
    skip_idx = int(rng.integers(0, len(steps)))
    reduced_steps = [s for i, s in enumerate(steps) if i != skip_idx]
    return unfold_points(punch_points, reduced_steps)


def _mirrored_wrong(answer_points: list[Point]) -> list[Point]:
    return [(1.0 - x, y) for x, y in answer_points]


def _shifted_hole(
    answer_points: list[Point], rng: np.random.Generator, shift_mag: float = 0.08
) -> list[Point]:
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = shift_mag * np.cos(angle), shift_mag * np.sin(angle)
    return [
        (min(0.98, max(0.02, x + dx)), min(0.98, max(0.02, y + dy))) for x, y in answer_points
    ]


def _apply_rule(
    rule: str,
    answer_points: list[Point],
    punch_points: list[Point],
    steps: list[FoldStep],
    rng: np.random.Generator,
) -> list[Point] | None:
    if rule == "missing_hole":
        return _missing_hole(answer_points, rng)
    if rule == "extra_hole":
        return _extra_hole(answer_points, rng)
    if rule == "wrong_symmetry_axis":
        return _wrong_symmetry_axis(punch_points, steps, rng)
    if rule == "mirrored_wrong":
        return _mirrored_wrong(answer_points)
    if rule == "shifted_hole":
        return _shifted_hole(answer_points, rng)
    raise ValueError(f"Unknown distractor rule: {rule}")


def generate_distractors(
    answer_points: list[Point],
    punch_points: list[Point],
    steps: list[FoldStep],
    rng: np.random.Generator,
    count: int = 3,
    max_attempts_per_rule: int = 5,
) -> list[Distractor]:
    seen = {_round_all(answer_points)}
    results: list[Distractor] = []

    pool = list(RULE_POOL)
    rng.shuffle(pool)

    for rule in pool:
        if len(results) >= count:
            break
        for _ in range(max_attempts_per_rule):
            candidate = _apply_rule(rule, answer_points, punch_points, steps, rng)
            if candidate is None:
                break
            candidate_key = _round_all(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            results.append(Distractor(points=candidate, rule=rule))
            break

    if len(results) < count:
        raise RuntimeError(
            f"Could only generate {len(results)}/{count} distinct distractors "
            f"for this record"
        )
    return results[:count]

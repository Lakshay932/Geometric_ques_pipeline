from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.affinity import rotate
from shapely.geometry import Polygon

from engine.rotation_series.geometry import ROTATION_STEPS, RotationDirection, rotate_by_steps

RULE_POOL = ["wrong_direction", "wrong_step_size", "reflected_instead", "stale_repeat", "skipped_a_step"]

_ROUND_NDIGITS = 6

@dataclass
class RotationSeriesDistractor:

    shape: Polygon
    rule: str

def _signature(shape: Polygon, ndigits: int = _ROUND_NDIGITS) -> tuple:
    return tuple(sorted((round(x, ndigits), round(y, ndigits)) for x, y in shape.exterior.coords[:-1]))

def _wrong_direction(
    original: Polygon, step_degrees: int, direction: RotationDirection, sequence_length: int
) -> Polygon:
    opposite: RotationDirection = "ccw" if direction == "cw" else "cw"
    return rotate_by_steps(original, step_degrees, opposite, sequence_length + 1)

def _wrong_step_size(
    original: Polygon, step_degrees: int, direction: RotationDirection, sequence_length: int, rng: np.random.Generator
) -> Polygon:
    other_steps = [s for s in ROTATION_STEPS if s != step_degrees]
    wrong_step = other_steps[int(rng.integers(0, len(other_steps)))]
    return rotate_by_steps(original, wrong_step, direction, sequence_length + 1)

def _reflected_instead(last_panel: Polygon) -> Polygon:
    cx = last_panel.centroid.x
    return Polygon([(2 * cx - x, y) for x, y in last_panel.exterior.coords])

def _stale_repeat(last_panel: Polygon) -> Polygon:
    return Polygon(last_panel.exterior.coords)

def _skipped_a_step(
    original: Polygon, step_degrees: int, direction: RotationDirection, sequence_length: int
) -> Polygon:
    return rotate_by_steps(original, step_degrees, direction, sequence_length + 2)

def _apply_rule(
    rule: str,
    original: Polygon,
    panels: list[Polygon],
    step_degrees: int,
    direction: RotationDirection,
    sequence_length: int,
    rng: np.random.Generator,
) -> Polygon | None:
    if rule == "wrong_direction":
        return _wrong_direction(original, step_degrees, direction, sequence_length)
    if rule == "wrong_step_size":
        return _wrong_step_size(original, step_degrees, direction, sequence_length, rng)
    if rule == "reflected_instead":
        return _reflected_instead(panels[-1])
    if rule == "stale_repeat":
        return _stale_repeat(panels[-1])
    if rule == "skipped_a_step":
        return _skipped_a_step(original, step_degrees, direction, sequence_length)
    raise ValueError(f"Unknown distractor rule: {rule}")

def generate_distractors(
    original: Polygon,
    panels: list[Polygon],
    answer: Polygon,
    step_degrees: int,
    direction: RotationDirection,
    sequence_length: int,
    rng: np.random.Generator,
    count: int = 3,
    max_attempts_per_rule: int = 5,
    rule_pool: list[str] = RULE_POOL,
) -> list[RotationSeriesDistractor]:
    seen = {_signature(answer)}
    results: list[RotationSeriesDistractor] = []

    for _ in range(count):
        if len(results) >= count:
            break
        pool = list(rule_pool)
        rng.shuffle(pool)
        for rule in pool:
            if len(results) >= count:
                break
            for _ in range(max_attempts_per_rule):
                candidate = _apply_rule(rule, original, panels, step_degrees, direction, sequence_length, rng)
                if candidate is None:
                    break
                key = _signature(candidate)
                if key in seen:
                    continue
                seen.add(key)
                results.append(RotationSeriesDistractor(shape=candidate, rule=rule))
                break

    if len(results) < count:
        raise RuntimeError(f"Could only generate {len(results)}/{count} distinct rotation_series distractors")
    return results[:count]

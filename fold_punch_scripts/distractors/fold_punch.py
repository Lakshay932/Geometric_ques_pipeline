from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import Polygon

from engine.fold_punch.geometry import (
    FoldStep,
    Point,
    bbox_center_line,
    bbox_diagonal_line,
    reflect_point,
    reflect_polygon,
    unfold_points,
    unfold_shapes,
)

RULE_POOL = [
    "missing_hole",
    "extra_hole",
    "wrong_symmetry_axis",
    "mirrored_wrong",
    "shifted_hole",
]

NEW_RULE_POOL_V2 = [
    "single_hole_wrong_side",
    "rotated_90",
    "wrong_diagonal_variant",
    "phantom_diagonal_fold",
    "last_fold_axis_swapped",
]
RULE_POOL_V2 = RULE_POOL + NEW_RULE_POOL_V2

_EXTRA_HOLE_FALLBACK_OFFSET = 0.15
_EXTRA_HOLE_EDGE_MARGIN = 0.05
_SHIFTED_HOLE_MAGNITUDE = 0.08
_SHIFTED_HOLE_EDGE_MARGIN = 0.02
_ROUND_NDIGITS = 6

@dataclass
class Distractor:

    points: list[Point]
    rule: str

    shapes: list[Polygon] | None = None

def _round_all(points: list[Point], ndigits: int = _ROUND_NDIGITS) -> frozenset[Point]:
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
        extra = (
            min(1 - _EXTRA_HOLE_EDGE_MARGIN, max(_EXTRA_HOLE_EDGE_MARGIN, base[0] + _EXTRA_HOLE_FALLBACK_OFFSET)),
            base[1],
        )
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
    answer_points: list[Point], rng: np.random.Generator, shift_mag: float = _SHIFTED_HOLE_MAGNITUDE
) -> list[Point]:
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = shift_mag * np.cos(angle), shift_mag * np.sin(angle)
    return [
        (
            min(1 - _SHIFTED_HOLE_EDGE_MARGIN, max(_SHIFTED_HOLE_EDGE_MARGIN, x + dx)),
            min(1 - _SHIFTED_HOLE_EDGE_MARGIN, max(_SHIFTED_HOLE_EDGE_MARGIN, y + dy)),
        )
        for x, y in answer_points
    ]

def _single_hole_wrong_side(
    answer_points: list[Point], steps: list[FoldStep], rng: np.random.Generator
) -> list[Point] | None:
    if not steps or not answer_points:
        return None
    idx = int(rng.integers(0, len(answer_points)))
    line = steps[int(rng.integers(0, len(steps)))].line
    wrong_point = reflect_point(answer_points[idx], line)
    return [wrong_point if i == idx else p for i, p in enumerate(answer_points)]

def _rotated_90(answer_points: list[Point]) -> list[Point]:
    return [(1.0 - y, x) for x, y in answer_points]

def _wrong_diagonal_variant(
    punch_points: list[Point], steps: list[FoldStep]
) -> list[Point] | None:
    diagonal_idx = next((i for i, s in enumerate(steps) if s.axis == "diagonal"), None)
    if diagonal_idx is None:
        return None
    step = steps[diagonal_idx]
    other_variant = "anti" if step.diagonal_variant == "main" else "main"
    alt_line = bbox_diagonal_line(step.polygon_before, other_variant)
    modified_steps = list(steps)
    modified_steps[diagonal_idx] = replace(step, line=alt_line, diagonal_variant=other_variant)
    return unfold_points(punch_points, modified_steps)

def _phantom_diagonal_fold(answer_points: list[Point]) -> list[Point] | None:
    original = {tuple(p) for p in answer_points}
    swapped = {(y, x) for x, y in answer_points}
    union = original | swapped
    if union == original:
        return None
    return sorted(union)

def _last_fold_axis_swapped(
    punch_points: list[Point], steps: list[FoldStep]
) -> list[Point] | None:
    if not steps or steps[-1].axis == "diagonal":
        return None
    last = steps[-1]
    alt_axis = "horizontal" if last.axis == "vertical" else "vertical"
    alt_line = bbox_center_line(last.polygon_before, alt_axis)
    modified_steps = list(steps[:-1]) + [replace(last, line=alt_line, axis=alt_axis)]
    return unfold_points(punch_points, modified_steps)

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
    if rule == "single_hole_wrong_side":
        return _single_hole_wrong_side(answer_points, steps, rng)
    if rule == "rotated_90":
        return _rotated_90(answer_points)
    if rule == "wrong_diagonal_variant":
        return _wrong_diagonal_variant(punch_points, steps)
    if rule == "phantom_diagonal_fold":
        return _phantom_diagonal_fold(answer_points)
    if rule == "last_fold_axis_swapped":
        return _last_fold_axis_swapped(punch_points, steps)
    raise ValueError(f"Unknown distractor rule: {rule}")

def generate_distractors(
    answer_points: list[Point],
    punch_points: list[Point],
    steps: list[FoldStep],
    rng: np.random.Generator,
    count: int = 3,
    max_attempts_per_rule: int = 5,
    rule_pool: list[str] = RULE_POOL,
) -> list[Distractor]:
    seen = {_round_all(answer_points)}
    results: list[Distractor] = []

    pool = list(rule_pool)
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

def _reflect_shape_through_center(shape: Polygon) -> Polygon:
    return Polygon([(1.0 - x, 1.0 - y) for x, y in shape.exterior.coords])

def _translate_shape(shape: Polygon, dx: float, dy: float) -> Polygon:
    return Polygon([(x + dx, y + dy) for x, y in shape.exterior.coords])

def _shape_signature(shape: Polygon, ndigits: int = _ROUND_NDIGITS) -> tuple:
    return tuple(sorted((round(x, ndigits), round(y, ndigits)) for x, y in shape.exterior.coords[:-1]))

def _round_all_shapes(shapes: list[Polygon]) -> frozenset:
    return frozenset(_shape_signature(s) for s in shapes)

def _missing_hole_shapes(answer_shapes: list[Polygon], rng: np.random.Generator) -> list[Polygon] | None:
    if len(answer_shapes) < 2:
        return None
    idx = int(rng.integers(0, len(answer_shapes)))
    return [s for i, s in enumerate(answer_shapes) if i != idx]

def _extra_hole_shapes(answer_shapes: list[Polygon], rng: np.random.Generator) -> list[Polygon] | None:
    base = answer_shapes[int(rng.integers(0, len(answer_shapes)))]
    extra = _reflect_shape_through_center(base)
    if any(extra.equals_exact(s, tolerance=10**-_ROUND_NDIGITS) for s in answer_shapes):
        cx, cy = base.centroid.x, base.centroid.y
        fallback_cx = min(1 - _EXTRA_HOLE_EDGE_MARGIN, max(_EXTRA_HOLE_EDGE_MARGIN, cx + _EXTRA_HOLE_FALLBACK_OFFSET))
        extra = _translate_shape(base, fallback_cx - cx, 0.0)
    return list(answer_shapes) + [extra]

def _wrong_symmetry_axis_shapes(
    punch_shapes: list[Polygon], steps: list[FoldStep], rng: np.random.Generator
) -> list[Polygon] | None:
    if not steps:
        return None
    skip_idx = int(rng.integers(0, len(steps)))
    reduced_steps = [s for i, s in enumerate(steps) if i != skip_idx]
    return unfold_shapes(punch_shapes, reduced_steps)

def _mirrored_wrong_shapes(answer_shapes: list[Polygon]) -> list[Polygon]:
    vertical_center_line = ((0.5, 0.0), (0.5, 1.0))
    return [reflect_polygon(s, vertical_center_line) for s in answer_shapes]

def _shifted_hole_shapes(
    answer_shapes: list[Polygon], rng: np.random.Generator, shift_mag: float = _SHIFTED_HOLE_MAGNITUDE
) -> list[Polygon]:
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = shift_mag * np.cos(angle), shift_mag * np.sin(angle)
    shifted = []
    for shape in answer_shapes:
        cx, cy = shape.centroid.x, shape.centroid.y
        new_cx = min(1 - _SHIFTED_HOLE_EDGE_MARGIN, max(_SHIFTED_HOLE_EDGE_MARGIN, cx + dx))
        new_cy = min(1 - _SHIFTED_HOLE_EDGE_MARGIN, max(_SHIFTED_HOLE_EDGE_MARGIN, cy + dy))
        shifted.append(_translate_shape(shape, new_cx - cx, new_cy - cy))
    return shifted

def _apply_shape_rule(
    rule: str,
    answer_shapes: list[Polygon],
    punch_shapes: list[Polygon],
    steps: list[FoldStep],
    rng: np.random.Generator,
) -> list[Polygon] | None:
    if rule == "missing_hole":
        return _missing_hole_shapes(answer_shapes, rng)
    if rule == "extra_hole":
        return _extra_hole_shapes(answer_shapes, rng)
    if rule == "wrong_symmetry_axis":
        return _wrong_symmetry_axis_shapes(punch_shapes, steps, rng)
    if rule == "mirrored_wrong":
        return _mirrored_wrong_shapes(answer_shapes)
    if rule == "shifted_hole":
        return _shifted_hole_shapes(answer_shapes, rng)
    raise ValueError(f"Unknown distractor rule: {rule}")

def generate_shape_distractors(
    answer_shapes: list[Polygon],
    punch_shapes: list[Polygon],
    steps: list[FoldStep],
    rng: np.random.Generator,
    count: int = 3,
    max_attempts_per_rule: int = 5,
) -> list[Distractor]:
    seen = {_round_all_shapes(answer_shapes)}
    results: list[Distractor] = []

    pool = list(RULE_POOL)
    rng.shuffle(pool)

    for rule in pool:
        if len(results) >= count:
            break
        for _ in range(max_attempts_per_rule):
            candidate = _apply_shape_rule(rule, answer_shapes, punch_shapes, steps, rng)
            if candidate is None:
                break
            candidate_key = _round_all_shapes(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            points = [(s.centroid.x, s.centroid.y) for s in candidate]
            results.append(Distractor(points=points, rule=rule, shapes=candidate))
            break

    if len(results) < count:
        raise RuntimeError(
            f"Could only generate {len(results)}/{count} distinct shape distractors "
            f"for this record"
        )
    return results[:count]

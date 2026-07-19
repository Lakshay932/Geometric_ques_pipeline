
"""Deterministic fold/punch geometry engine (TRD Section 2.3 — correctness by construction).

The folded region is tracked as a Shapely polygon expressed directly in the
original unit-square coordinate frame (no separate "folded space" needed,
since every fold is a reflection and the visible region after folding is
always literally a subregion of the original square).

A fold is only valid if the chosen line is an actual symmetry line of the
*current* polygon (reflecting the whole polygon across it reproduces the
same polygon) — this is checked directly rather than hand-coded per shape,
so it automatically:
- allows unlimited vertical/horizontal folds to chain (a rectangle's bbox
  center lines are always symmetry lines of that rectangle),
- allows a diagonal fold only when the current shape is actually a square,
- and after a diagonal fold, correctly restricts further folds to the one
  remaining symmetry line of the resulting right-isosceles triangle (its
  other diagonal), rejecting vertical/horizontal/repeat-diagonal folds.

Unfolding a punched hole is reflecting it across each fold line in reverse
order, doubling the point set at each step — the same principle as
physically unfolding creased paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from shapely.ops import split as shapely_split

Point = tuple[float, float]
Line = tuple[Point, Point]
Axis = Literal["vertical", "horizontal", "diagonal"]

_SYMMETRY_TOL = 1e-6
_EXTENSION_MARGIN = 10.0


class InvalidFoldError(ValueError):
    """The requested fold axis is not a valid symmetry line of the current shape."""


def unit_square() -> Polygon:
    return Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def reflect_point(point: Point, line: Line) -> Point:
    p = np.array(point, dtype=float)
    a = np.array(line[0], dtype=float)
    b = np.array(line[1], dtype=float)
    direction = b - a
    direction = direction / np.linalg.norm(direction)
    projection = a + np.dot(p - a, direction) * direction
    reflected = 2 * projection - p
    return (float(reflected[0]), float(reflected[1]))


def bbox_center_line(polygon: Polygon, axis: Literal["vertical", "horizontal"]) -> Line:
    minx, miny, maxx, maxy = polygon.bounds
    if axis == "vertical":
        cx = (minx + maxx) / 2
        return ((cx, miny - _EXTENSION_MARGIN), (cx, maxy + _EXTENSION_MARGIN))
    cy = (miny + maxy) / 2
    return ((minx - _EXTENSION_MARGIN, cy), (maxx + _EXTENSION_MARGIN, cy))


def bbox_diagonal_line(polygon: Polygon, variant: Literal["main", "anti"]) -> Line:
    minx, miny, maxx, maxy = polygon.bounds
    if variant == "main":
        p1, p2 = (minx, miny), (maxx, maxy)
    else:
        p1, p2 = (minx, maxy), (maxx, miny)
    direction = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    direction = direction / np.linalg.norm(direction)
    ext1 = tuple(np.array(p1) - direction * _EXTENSION_MARGIN)
    ext2 = tuple(np.array(p2) + direction * _EXTENSION_MARGIN)
    return (ext1, ext2)


def is_symmetry_line(polygon: Polygon, line: Line, tol: float = _SYMMETRY_TOL) -> bool:
    reflected_coords = [reflect_point(pt, line) for pt in polygon.exterior.coords]
    reflected_polygon = Polygon(reflected_coords)
    if not reflected_polygon.is_valid or reflected_polygon.area < tol:
        return False
    diff_area = polygon.symmetric_difference(reflected_polygon).area
    return diff_area < tol


def candidate_lines_for_axis(
    polygon: Polygon,
    axis: Axis,
    diagonal_variant: Literal["main", "anti"] | None = None,
) -> list[tuple[Line, Literal["main", "anti"] | None]]:
    """Returns (line, variant) pairs — variant is None for vertical/horizontal,
    "main"/"anti" for diagonal, so callers can record which one was actually
    chosen (not just what was requested)."""
    if axis in ("vertical", "horizontal"):
        line = bbox_center_line(polygon, axis)
        return [(line, None)] if is_symmetry_line(polygon, line) else []

    variants: list[Literal["main", "anti"]] = (
        [diagonal_variant] if diagonal_variant is not None else ["main", "anti"]
    )
    candidates = []
    for variant in variants:
        line = bbox_diagonal_line(polygon, variant)
        if is_symmetry_line(polygon, line):
            candidates.append((line, variant))
    return candidates


def _split_polygon_by_line(polygon: Polygon, line: Line) -> list[Polygon]:
    from shapely.geometry import LineString

    cut_line = LineString(line)
    result = shapely_split(polygon, cut_line)
    parts = [geom for geom in result.geoms if geom.area > _SYMMETRY_TOL]
    if len(parts) != 2:
        raise InvalidFoldError(
            f"Fold line did not cleanly bisect the current shape into 2 parts (got {len(parts)})"
        )
    return parts


def _sort_key_for_axis(axis: Axis):
    if axis == "vertical":
        return lambda poly: poly.centroid.x
    if axis == "horizontal":
        return lambda poly: poly.centroid.y
    return lambda poly: poly.centroid.x + poly.centroid.y


@dataclass
class FoldStep:
    axis: Axis
    line: Line
    kept_side: int
    diagonal_variant: Literal["main", "anti"] | None
    polygon_before: Polygon
    polygon_after: Polygon


def apply_fold(
    polygon: Polygon,
    axis: Axis,
    rng: np.random.Generator,
    *,
    keep_side: int | None = None,
    keep_point: Point | None = None,
    diagonal_variant: Literal["main", "anti"] | None = None,
) -> FoldStep:
    candidates = candidate_lines_for_axis(polygon, axis, diagonal_variant=diagonal_variant)
    if not candidates:
        raise InvalidFoldError(f"No valid {axis} fold for the current shape")

    line, chosen_variant = (
        candidates[0] if len(candidates) == 1 else candidates[rng.integers(0, len(candidates))]
    )
    parts = _split_polygon_by_line(polygon, line)
    parts = sorted(parts, key=_sort_key_for_axis(axis))

    if keep_point is not None:
        matches = [i for i, part in enumerate(parts) if part.buffer(1e-9).contains(ShapelyPoint(keep_point))]
        if not matches:
            raise InvalidFoldError(f"keep_point {keep_point} is not inside either split part")
        side = matches[0]
    elif keep_side is not None:
        side = keep_side
    else:
        side = int(rng.integers(0, 2))

    return FoldStep(
        axis=axis,
        line=line,
        kept_side=side,
        diagonal_variant=chosen_variant,
        polygon_before=polygon,
        polygon_after=parts[side],
    )


@dataclass
class FoldSequenceResult:
    steps: list[FoldStep] = field(default_factory=list)
    final_polygon: Polygon = field(default_factory=unit_square)


def fold_paper(
    steps: list[tuple[Axis, int | None]],
    rng: np.random.Generator,
    *,
    diagonal_variant: Literal["main", "anti"] | None = None,
    keep_point: Point | None = None,
) -> FoldSequenceResult:
    polygon = unit_square()
    applied_steps: list[FoldStep] = []
    for axis, keep_side in steps:
        step_kwargs: dict = {"keep_side": keep_side}
        if axis == "diagonal" and keep_side is None:
            step_kwargs = {"keep_point": keep_point, "diagonal_variant": diagonal_variant}
        step = apply_fold(polygon, axis, rng, **step_kwargs)
        applied_steps.append(step)
        polygon = step.polygon_after
    return FoldSequenceResult(steps=applied_steps, final_polygon=polygon)


def _round_point(point: Point, ndigits: int = 6) -> Point:
    return (round(point[0], ndigits), round(point[1], ndigits))


def unfold_points(points: list[Point], steps: list[FoldStep]) -> list[Point]:
    current = {_round_point(p) for p in points}
    for step in reversed(steps):
        reflected = {_round_point(reflect_point(p, step.line)) for p in current}
        current = current | reflected
    return sorted(current)


def sample_punch_points(
    polygon: Polygon,
    count: int,
    rng: np.random.Generator,
    margin: float = 0.04,
    min_separation: float = 0.06,
    max_attempts: int = 2000,
) -> list[Point]:
    interior = polygon.buffer(-margin)
    if interior.is_empty:
        raise ValueError("Polygon too small for the requested punch margin")
    minx, miny, maxx, maxy = interior.bounds

    points: list[Point] = []
    for _ in range(max_attempts):
        if len(points) >= count:
            break
        candidate = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if not interior.contains(ShapelyPoint(candidate)):
            continue
        if any(
            ((candidate[0] - p[0]) ** 2 + (candidate[1] - p[1]) ** 2) ** 0.5 < min_separation
            for p in points
        ):
            continue
        points.append(candidate)

    if len(points) < count:
        raise ValueError(
            f"Could not sample {count} well-separated punch points in {max_attempts} attempts"
        )
    return points


@dataclass
class FoldPunchGeometry:
    axis_sequence: list[Axis]
    steps: list[FoldStep]
    final_polygon: Polygon
    punch_points: list[Point]
    answer_points: list[Point]


def generate(
    axis_sequence: list[Axis],
    punch_count: int,
    rng: np.random.Generator,
) -> FoldPunchGeometry:
    steps_spec: list[tuple[Axis, int | None]] = [(axis, None) for axis in axis_sequence]
    result = fold_paper(steps_spec, rng)
    punches = sample_punch_points(result.final_polygon, punch_count, rng)
    answer = unfold_points(punches, result.steps)
    return FoldPunchGeometry(
        axis_sequence=axis_sequence,
        steps=result.steps,
        final_polygon=result.final_polygon,
        punch_points=punches,
        answer_points=answer,
    )


def serialize_fold_steps(steps: list[FoldStep]) -> list[dict]:
    return [
        {"axis": s.axis, "kept_side": s.kept_side, "diagonal_variant": s.diagonal_variant}
        for s in steps
    ]


def reconstruct_geometry(
    axis_sequence: list[Axis],
    fold_steps: list[dict],
    punch_points: list[Point],
) -> FoldPunchGeometry:
    """Rebuilds the exact same geometry from persisted params (FR-12) —
    every choice (which side was kept, which diagonal, where each punch
    landed) is explicit here, so no randomness is involved.
    """
    dummy_rng = np.random.default_rng(0)  # unused: every choice below is explicit
    polygon = unit_square()
    steps: list[FoldStep] = []
    for step_spec in fold_steps:
        step = apply_fold(
            polygon,
            step_spec["axis"],
            dummy_rng,
            keep_side=step_spec["kept_side"],
            diagonal_variant=step_spec.get("diagonal_variant"),
        )
        steps.append(step)
        polygon = step.polygon_after

    points = [tuple(p) for p in punch_points]
    answer = unfold_points(points, steps)
    return FoldPunchGeometry(
        axis_sequence=axis_sequence,
        steps=steps,
        final_polygon=polygon,
        punch_points=points,
        answer_points=answer,
    )

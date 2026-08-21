from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from shapely.ops import split as shapely_split
from shapely.ops import unary_union

Point = tuple[float, float]
Line = tuple[Point, Point]
Axis = Literal["vertical", "horizontal", "diagonal"]

_SYMMETRY_TOL = 1e-6

_EXTENSION_MARGIN = 10.0

_CONTAINS_TOL = 1e-9

_ROUND_NDIGITS = 6

class InvalidFoldError(ValueError):
    pass

def unit_square() -> Polygon:
    return Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

StartShape = Literal["square", "circle", "rectangle_2_1"]

_START_CIRCLE_CENTER: Point = (0.5, 0.5)
_START_CIRCLE_RADIUS = 0.5
_START_CIRCLE_QUAD_SEGS = 32

_START_RECTANGLE_2_1_SIZE: tuple[float, float] = (1.0, 0.5)

def make_start_polygon(shape: StartShape) -> Polygon:
    if shape == "square":
        return unit_square()
    if shape == "circle":
        return ShapelyPoint(_START_CIRCLE_CENTER).buffer(
            _START_CIRCLE_RADIUS, quad_segs=_START_CIRCLE_QUAD_SEGS
        )
    if shape == "rectangle_2_1":
        w, h = _START_RECTANGLE_2_1_SIZE
        return Polygon([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])
    raise ValueError(f"Unknown start shape: {shape!r}")

def reflect_point(point: Point, line: Line) -> Point:
    p = np.array(point, dtype=float)
    a = np.array(line[0], dtype=float)
    b = np.array(line[1], dtype=float)
    direction = b - a
    direction = direction / np.linalg.norm(direction)
    projection = a + np.dot(p - a, direction) * direction
    reflected = 2 * projection - p
    return (float(reflected[0]), float(reflected[1]))

def reflect_polygon(polygon: Polygon, line: Line) -> Polygon:
    return Polygon([reflect_point(pt, line) for pt in polygon.exterior.coords])

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
        corner_a, corner_b = (minx, miny), (maxx, maxy)
    else:
        corner_a, corner_b = (minx, maxy), (maxx, miny)
    direction = np.array(corner_b, dtype=float) - np.array(corner_a, dtype=float)
    direction = direction / np.linalg.norm(direction)
    ext1 = tuple(np.array(corner_a) - direction * _EXTENSION_MARGIN)
    ext2 = tuple(np.array(corner_b) + direction * _EXTENSION_MARGIN)
    return (ext1, ext2)

def is_symmetry_line(polygon: Polygon, line: Line, tol: float = _SYMMETRY_TOL) -> bool:
    reflected_polygon = reflect_polygon(polygon, line)
    if not reflected_polygon.is_valid or reflected_polygon.area < tol:
        return False
    diff_area = polygon.symmetric_difference(reflected_polygon).area
    return diff_area < tol

def candidate_lines_for_axis(
    polygon: Polygon,
    axis: Axis,
    diagonal_variant: Literal["main", "anti"] | None = None,
) -> list[tuple[Line, Literal["main", "anti"] | None]]:
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

def _choose_kept_side(
    parts: list[Polygon],
    rng: np.random.Generator,
    keep_side: int | None,
    keep_point: Point | None,
) -> int:
    if keep_point is not None:
        matches = [
            i
            for i, part in enumerate(parts)
            if part.buffer(_CONTAINS_TOL).contains(ShapelyPoint(keep_point))
        ]
        if not matches:
            raise InvalidFoldError(f"keep_point {keep_point} is not inside either split part")
        return matches[0]
    if keep_side is not None:
        return keep_side
    return int(rng.integers(0, 2))

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
    side = _choose_kept_side(parts, rng, keep_side, keep_point)

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
    start_polygon: Polygon | None = None,
) -> FoldSequenceResult:
    polygon = start_polygon if start_polygon is not None else unit_square()
    applied_steps: list[FoldStep] = []
    for axis, keep_side in steps:
        step_kwargs: dict = {"keep_side": keep_side}
        if axis == "diagonal" and keep_side is None:
            step_kwargs = {"keep_point": keep_point, "diagonal_variant": diagonal_variant}
        step = apply_fold(polygon, axis, rng, **step_kwargs)
        applied_steps.append(step)
        polygon = step.polygon_after
    return FoldSequenceResult(steps=applied_steps, final_polygon=polygon)

def _round_point(point: Point, ndigits: int = _ROUND_NDIGITS) -> Point:
    return (round(point[0], ndigits), round(point[1], ndigits))

def unfold_points(points: list[Point], steps: list[FoldStep]) -> list[Point]:
    current = {_round_point(p) for p in points}
    for step in reversed(steps):
        reflected = {_round_point(reflect_point(p, step.line)) for p in current}
        current = current | reflected
    return sorted(current)

def unfold_shapes(shapes: list[Polygon], steps: list[FoldStep]) -> list[Polygon]:
    current = list(shapes)
    for step in reversed(steps):
        current = current + [reflect_polygon(shape, step.line) for shape in current]
    return current

def unfold_shapes_merged(shapes: list[Polygon], steps: list[FoldStep]) -> list[Polygon]:
    doubled = unfold_shapes(shapes, steps)

    cleaned = [
        piece.buffer(0)
        for piece in doubled
        if not piece.is_empty and piece.area > _SYMMETRY_TOL
    ]
    cleaned = [piece for piece in cleaned if not piece.is_empty]
    if not cleaned:
        return []
    merged = unary_union(cleaned)
    if merged.is_empty:
        return []
    if isinstance(merged, MultiPolygon):
        return list(merged.geoms)
    return [merged]

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

def sample_boundary_punch_point(
    polygon: Polygon,
    rng: np.random.Generator,
    corner_probability: float = 0.5,
) -> Point:
    exterior_coords = list(polygon.exterior.coords[:-1])
    if rng.random() < corner_probability:
        idx = int(rng.integers(0, len(exterior_coords)))
        return exterior_coords[idx]

    idx = int(rng.integers(0, len(exterior_coords)))
    p1 = exterior_coords[idx]
    p2 = exterior_coords[(idx + 1) % len(exterior_coords)]
    t = float(rng.random())
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

PunchKind = Literal["circle", "square", "triangle", "star", "slit"]

@dataclass
class PunchSpec:

    center: Point
    kind: PunchKind
    size: float
    rotation: float = 0.0

@dataclass
class FoldPunchGeometry:

    axis_sequence: list[Axis]
    steps: list[FoldStep]
    final_polygon: Polygon
    punch_points: list[Point]
    answer_points: list[Point]

    punches: list[PunchSpec] = field(default_factory=list)
    answer_shapes: list[Polygon] = field(default_factory=list)

    start_shape: StartShape = "square"

def generate(
    axis_sequence: list[Axis],
    punch_count: int,
    rng: np.random.Generator,
    start_shape: StartShape = "square",
) -> FoldPunchGeometry:
    steps_spec: list[tuple[Axis, int | None]] = [(axis, None) for axis in axis_sequence]
    result = fold_paper(steps_spec, rng, start_polygon=make_start_polygon(start_shape))
    punches = sample_punch_points(result.final_polygon, punch_count, rng)
    answer = unfold_points(punches, result.steps)
    return FoldPunchGeometry(
        axis_sequence=axis_sequence,
        steps=result.steps,
        final_polygon=result.final_polygon,
        punch_points=punches,
        answer_points=answer,
        start_shape=start_shape,
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
    start_shape: StartShape = "square",
) -> FoldPunchGeometry:
    dummy_rng = np.random.default_rng(0)
    polygon = make_start_polygon(start_shape)
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
        start_shape=start_shape,
    )

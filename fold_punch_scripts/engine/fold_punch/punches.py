from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from .geometry import (
    Axis,
    FoldPunchGeometry,
    Point,
    PunchKind,
    PunchSpec,
    fold_paper,
    generate,
    reconstruct_geometry,
    sample_boundary_punch_point,
    sample_punch_points,
    unfold_shapes,
    unfold_shapes_merged,
)

PUNCH_KINDS: tuple[PunchKind, ...] = ("circle", "square", "triangle", "star", "slit")

DEFAULT_PUNCH_SIZE = 0.035

_CIRCLE_RESOLUTION = 16
_STAR_INNER_RATIO = 0.45
_SLIT_ASPECT_RATIO = 4.0

def _rotate_and_translate(corners: list[Point], center: Point, rotation: float) -> Polygon:
    cx, cy = center
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    return Polygon(
        [(cx + x * cos_r - y * sin_r, cy + x * sin_r + y * cos_r) for x, y in corners]
    )

def _regular_polygon(center: Point, radius: float, n_sides: int, rotation: float) -> Polygon:
    cx, cy = center
    return Polygon(
        [
            (
                cx + radius * math.cos(math.pi / 2 + rotation + i * (2 * math.pi / n_sides)),
                cy + radius * math.sin(math.pi / 2 + rotation + i * (2 * math.pi / n_sides)),
            )
            for i in range(n_sides)
        ]
    )

def _square_polygon(center: Point, size: float, rotation: float) -> Polygon:
    corners = [(-size, -size), (size, -size), (size, size), (-size, size)]
    return _rotate_and_translate(corners, center, rotation)

def _star_polygon(center: Point, outer_radius: float, rotation: float) -> Polygon:
    cx, cy = center
    inner_radius = outer_radius * _STAR_INNER_RATIO
    points = []
    for i in range(10):
        angle = math.pi / 2 + rotation + i * (math.pi / 5)
        r = outer_radius if i % 2 == 0 else inner_radius
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return Polygon(points)

def _slit_polygon(center: Point, size: float, rotation: float) -> Polygon:
    half_len = size
    half_w = size / _SLIT_ASPECT_RATIO
    corners = [(-half_len, -half_w), (half_len, -half_w), (half_len, half_w), (-half_len, half_w)]
    return _rotate_and_translate(corners, center, rotation)

def make_punch_polygon(spec: PunchSpec) -> Polygon:
    if spec.kind == "circle":
        return ShapelyPoint(spec.center).buffer(spec.size, quad_segs=_CIRCLE_RESOLUTION)
    if spec.kind == "square":
        return _square_polygon(spec.center, spec.size, spec.rotation)
    if spec.kind == "triangle":
        return _regular_polygon(spec.center, spec.size, 3, spec.rotation)
    if spec.kind == "star":
        return _star_polygon(spec.center, spec.size, spec.rotation)
    if spec.kind == "slit":
        return _slit_polygon(spec.center, spec.size, spec.rotation)
    raise ValueError(f"Unknown punch kind: {spec.kind!r}")

def generate_with_shapes(
    axis_sequence: list[Axis],
    punch_kinds: list[PunchKind],
    rng: np.random.Generator,
    punch_size: float = DEFAULT_PUNCH_SIZE,
    punch_rotations: list[float] | None = None,
) -> FoldPunchGeometry:
    base = generate(axis_sequence, len(punch_kinds), rng)
    rotations = punch_rotations if punch_rotations is not None else [0.0] * len(punch_kinds)
    punches = [
        PunchSpec(center=center, kind=kind, size=punch_size, rotation=rotation)
        for center, kind, rotation in zip(base.punch_points, punch_kinds, rotations)
    ]
    shapes = [make_punch_polygon(spec) for spec in punches]
    answer_shapes = unfold_shapes(shapes, base.steps)
    return replace(base, punches=punches, answer_shapes=answer_shapes)

def reconstruct_geometry_with_shapes(
    axis_sequence: list[Axis],
    fold_steps: list[dict],
    punch_points: list[Point],
    punch_shapes: list[dict],
) -> FoldPunchGeometry:
    base = reconstruct_geometry(axis_sequence, fold_steps, punch_points)
    punches = [
        PunchSpec(
            center=center,
            kind=spec["kind"],
            size=spec["size"],
            rotation=spec.get("rotation", 0.0),
        )
        for center, spec in zip(base.punch_points, punch_shapes)
    ]
    shapes = [make_punch_polygon(spec) for spec in punches]
    answer_shapes = unfold_shapes(shapes, base.steps)
    return replace(base, punches=punches, answer_shapes=answer_shapes)

def serialize_punch_shapes(punches: list[PunchSpec]) -> list[dict]:
    return [{"kind": p.kind, "size": p.size, "rotation": p.rotation} for p in punches]

def make_clipped_punch_polygon(spec: PunchSpec, folded_region: Polygon) -> Polygon:
    return make_punch_polygon(spec).intersection(folded_region)

def generate_with_edge_shapes(
    axis_sequence: list[Axis],
    punch_kinds: list[PunchKind],
    rng: np.random.Generator,
    punch_size: float = DEFAULT_PUNCH_SIZE,
    punch_rotations: list[float] | None = None,
    boundary_probability: float = 0.5,
    corner_probability: float = 0.5,
) -> FoldPunchGeometry:
    steps_spec: list[tuple[Axis, int | None]] = [(axis, None) for axis in axis_sequence]
    fold_result = fold_paper(steps_spec, rng)
    final_polygon = fold_result.final_polygon

    punch_count = len(punch_kinds)
    is_boundary = [rng.random() < boundary_probability for _ in range(punch_count)]
    interior_count = punch_count - sum(is_boundary)
    interior_points = iter(
        sample_punch_points(final_polygon, interior_count, rng) if interior_count else []
    )

    centers: list[Point] = []
    for boundary in is_boundary:
        if boundary:
            centers.append(sample_boundary_punch_point(final_polygon, rng, corner_probability))
        else:
            centers.append(next(interior_points))

    rotations = punch_rotations if punch_rotations is not None else [0.0] * punch_count
    punches = [
        PunchSpec(center=center, kind=kind, size=punch_size, rotation=rotation)
        for center, kind, rotation in zip(centers, punch_kinds, rotations)
    ]
    clipped_shapes = [make_clipped_punch_polygon(spec, final_polygon) for spec in punches]
    answer_shapes = unfold_shapes_merged(clipped_shapes, fold_result.steps)
    answer_points = [(shape.centroid.x, shape.centroid.y) for shape in answer_shapes]

    return FoldPunchGeometry(
        axis_sequence=axis_sequence,
        steps=fold_result.steps,
        final_polygon=final_polygon,
        punch_points=centers,
        answer_points=answer_points,
        punches=punches,
        answer_shapes=answer_shapes,
    )

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from engine.fold_punch.geometry import reflect_polygon
from .geometry import MIRROR_AXES, mirror_line_for_axis

_CENTER = (0.275, 0.275)
_MIN_RADIUS = 0.05
_MAX_RADIUS = 0.11
_ANGLE_JITTER_FRAC = 0.35
MIN_VERTICES = 5
MAX_VERTICES = 8

MIN_AREA = 0.0015
MIN_EDGE_LENGTH = 0.02

_SYMMETRY_TOLERANCE = 0.01

MAX_RETRIES = 30

def _draw_candidate(rng: np.random.Generator, num_vertices: int) -> Polygon:
    base_step = 2 * math.pi / num_vertices
    points = []
    for i in range(num_vertices):
        jitter = rng.uniform(-_ANGLE_JITTER_FRAC, _ANGLE_JITTER_FRAC) * base_step
        angle = i * base_step + jitter
        radius = rng.uniform(_MIN_RADIUS, _MAX_RADIUS)
        points.append((_CENTER[0] + radius * math.cos(angle), _CENTER[1] + radius * math.sin(angle)))
    return Polygon(points)

def _min_edge_length(polygon: Polygon) -> float:
    coords = list(polygon.exterior.coords)
    return min(
        math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
        for i in range(len(coords) - 1)
    )

def _is_too_symmetric(polygon: Polygon, tolerance: float = _SYMMETRY_TOLERANCE) -> bool:
    original_pts = list(polygon.exterior.coords)[:-1]
    for axis in MIRROR_AXES:
        reflected = reflect_polygon(polygon, mirror_line_for_axis(axis))
        reflected_pts = list(reflected.exterior.coords)[:-1]
        max_min_dist = max(
            min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in reflected_pts) for p in original_pts
        )
        if max_min_dist < tolerance:
            return True
    return False

def generate_random_figure(
    rng: np.random.Generator,
    num_vertices_range: tuple[int, int] = (MIN_VERTICES, MAX_VERTICES),
    max_retries: int = MAX_RETRIES,
) -> Polygon:
    low, high = num_vertices_range
    for _ in range(max_retries):
        num_vertices = int(rng.integers(low, high + 1))
        candidate = _draw_candidate(rng, num_vertices)
        if not candidate.is_valid:
            continue
        if candidate.area < MIN_AREA:
            continue
        if _min_edge_length(candidate) < MIN_EDGE_LENGTH:
            continue
        if _is_too_symmetric(candidate):
            continue
        return candidate
    raise RuntimeError(f"Failed to generate a valid random figure after {max_retries} retries")

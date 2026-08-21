from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shapely.affinity import rotate
from shapely.geometry import Polygon

from engine.mirror_water.geometry import FIGURE_IDS, FIGURES, make_figure_polygon

RotationDirection = Literal["cw", "ccw"]
ROTATION_DIRECTIONS: tuple[RotationDirection, ...] = ("cw", "ccw")

ROTATION_STEPS: tuple[int, ...] = (30, 45, 60, 90, 120)
SEQUENCE_LENGTHS: tuple[int, ...] = (3, 4)

def _signed_angle(step_degrees: float, direction: RotationDirection, n: int) -> float:
    magnitude = step_degrees * n
    return magnitude if direction == "ccw" else -magnitude

def rotate_by_steps(original: Polygon, step_degrees: float, direction: RotationDirection, n: int) -> Polygon:
    return rotate(original, _signed_angle(step_degrees, direction, n), origin=original.centroid)

@dataclass
class RotationSeriesGeometry:

    figure_id: str
    original: Polygon
    step_degrees: int
    direction: RotationDirection
    sequence_length: int
    panels: list[Polygon]
    answer: Polygon

def _build(
    figure_id: str, original: Polygon, step_degrees: int, direction: RotationDirection, sequence_length: int
) -> RotationSeriesGeometry:
    panels = [rotate_by_steps(original, step_degrees, direction, n) for n in range(sequence_length)]
    answer = rotate_by_steps(original, step_degrees, direction, sequence_length)
    return RotationSeriesGeometry(
        figure_id=figure_id,
        original=original,
        step_degrees=step_degrees,
        direction=direction,
        sequence_length=sequence_length,
        panels=panels,
        answer=answer,
    )

def generate(
    figure_id: str, step_degrees: int, direction: RotationDirection, sequence_length: int
) -> RotationSeriesGeometry:
    return _build(figure_id, make_figure_polygon(figure_id), step_degrees, direction, sequence_length)

def generate_from_polygon(
    original: Polygon,
    step_degrees: int,
    direction: RotationDirection,
    sequence_length: int,
    figure_id: str = "procedural",
) -> RotationSeriesGeometry:
    return _build(figure_id, original, step_degrees, direction, sequence_length)

def reconstruct_geometry(
    figure_id: str, step_degrees: int, direction: RotationDirection, sequence_length: int
) -> RotationSeriesGeometry:
    return generate(figure_id, step_degrees, direction, sequence_length)

def reconstruct_geometry_from_points(
    original_points: list[tuple[float, float]],
    step_degrees: int,
    direction: RotationDirection,
    sequence_length: int,
) -> RotationSeriesGeometry:
    return generate_from_polygon(Polygon(original_points), step_degrees, direction, sequence_length)

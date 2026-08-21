from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shapely.geometry import Polygon

from engine.fold_punch.geometry import Line, bbox_diagonal_line, reflect_polygon, unit_square

MirrorAxis = Literal["vertical", "horizontal", "diagonal_main", "diagonal_anti"]
MIRROR_AXES: tuple[MirrorAxis, ...] = ("vertical", "horizontal", "diagonal_main", "diagonal_anti")

AXIS_OFFSET_RANGE: tuple[float, float] = (-0.15, 0.15)

_EXTENSION_MARGIN = 10.0

FIGURES: dict[str, list[tuple[float, float]]] = {
    "hook": [(0.15, 0.15), (0.30, 0.15), (0.30, 0.25), (0.40, 0.25), (0.40, 0.40), (0.15, 0.40)],
    "flag": [(0.15, 0.15), (0.40, 0.15), (0.15, 0.40)],
    "arrow": [
        (0.15, 0.20), (0.32, 0.20), (0.32, 0.15), (0.40, 0.275),
        (0.32, 0.40), (0.32, 0.35), (0.15, 0.35),
    ],
    "step": [(0.15, 0.15), (0.25, 0.15), (0.25, 0.25), (0.35, 0.25), (0.35, 0.40), (0.15, 0.40)],
    "notch": [(0.15, 0.15), (0.40, 0.15), (0.40, 0.40), (0.30, 0.40), (0.30, 0.30), (0.15, 0.30)],
}
FIGURE_IDS: tuple[str, ...] = tuple(FIGURES.keys())

def make_figure_polygon(figure_id: str) -> Polygon:
    if figure_id not in FIGURES:
        raise ValueError(f"Unknown figure: {figure_id!r}")
    return Polygon(FIGURES[figure_id])

def mirror_line_for_axis(axis: MirrorAxis, offset: float = 0.0) -> Line:
    canvas = unit_square()
    minx, miny, maxx, maxy = canvas.bounds
    if axis == "vertical":
        cx = (minx + maxx) / 2 + offset
        return ((cx, miny - _EXTENSION_MARGIN), (cx, maxy + _EXTENSION_MARGIN))
    if axis == "horizontal":
        cy = (miny + maxy) / 2 + offset
        return ((minx - _EXTENSION_MARGIN, cy), (maxx + _EXTENSION_MARGIN, cy))
    if axis == "diagonal_main":
        return bbox_diagonal_line(canvas, "main")
    if axis == "diagonal_anti":
        return bbox_diagonal_line(canvas, "anti")
    raise ValueError(f"Unknown mirror axis: {axis!r}")

@dataclass
class MirrorWaterGeometry:

    figure_id: str
    axis: MirrorAxis
    original: Polygon
    answer: Polygon

    axis_offset: float = 0.0

def _reflect(figure_id: str, original: Polygon, axis: MirrorAxis, offset: float) -> MirrorWaterGeometry:
    line = mirror_line_for_axis(axis, offset)
    answer = reflect_polygon(original, line)
    return MirrorWaterGeometry(figure_id=figure_id, axis=axis, original=original, answer=answer, axis_offset=offset)

def generate(figure_id: str, axis: MirrorAxis, offset: float = 0.0) -> MirrorWaterGeometry:
    return _reflect(figure_id, make_figure_polygon(figure_id), axis, offset)

def generate_from_polygon(
    original: Polygon, axis: MirrorAxis, offset: float = 0.0, figure_id: str = "procedural"
) -> MirrorWaterGeometry:
    return _reflect(figure_id, original, axis, offset)

def reconstruct_geometry(figure_id: str, axis: MirrorAxis, offset: float = 0.0) -> MirrorWaterGeometry:
    return generate(figure_id, axis, offset)

def reconstruct_geometry_from_points(
    original_points: list[tuple[float, float]], axis: MirrorAxis, offset: float = 0.0
) -> MirrorWaterGeometry:
    return generate_from_polygon(Polygon(original_points), axis, offset)

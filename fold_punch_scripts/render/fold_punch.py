from __future__ import annotations

import math

from PIL import Image, ImageDraw
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box

from engine.fold_punch.geometry import FoldStep, Line, Point, unit_square

PANEL_SIZE = 200
PANEL_PADDING = 12
PADDING_FRAC = 0.12
BACKGROUND = (245, 245, 245)
PAPER_FILL = (255, 255, 255)
PAPER_OUTLINE = (20, 20, 20)
PAPER_OUTLINE_WIDTH = 2
FOLD_LINE_COLOR = (200, 40, 40)
FOLD_LINE_WIDTH = 2
FOLD_LINE_DASH_LEN = 6
FOLD_LINE_DASH_GAP = 5
ARROW_COLOR = (200, 40, 40)
ARROW_WIDTH = 3
ARROW_HEAD_LEN = 10
ARROW_HEAD_ANGLE = 0.4
PUNCH_COLOR = (20, 20, 20)
PUNCH_RADIUS = 6

_MIN_DISCARDED_AREA = 1e-9

def _make_mapper(bounds: tuple[float, float, float, float], canvas_size: int):
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    w = w or 1e-6
    h = h or 1e-6
    pad = PADDING_FRAC * max(w, h)
    minx, maxx = minx - pad, maxx + pad
    miny, maxy = miny - pad, maxy + pad
    span = max(maxx - minx, maxy - miny)
    scale = canvas_size / span

    def to_px(point: Point) -> tuple[float, float]:
        x, y = point
        px = (x - minx) * scale
        py = canvas_size - (y - miny) * scale
        return (px, py)

    padded_bounds = (minx, miny, maxx, maxy)
    return to_px, padded_bounds

def _clip_line_to_bounds(line: Line, bounds: tuple[float, float, float, float]):
    minx, miny, maxx, maxy = bounds
    rect = shapely_box(minx, miny, maxx, maxy)
    clipped = LineString(line).intersection(rect)
    if clipped.is_empty:
        return None
    coords = list(clipped.coords)
    if len(coords) < 2:
        return None
    return (coords[0], coords[-1])

def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1,
    p2,
    color,
    width=FOLD_LINE_WIDTH,
    dash=FOLD_LINE_DASH_LEN,
    gap=FOLD_LINE_DASH_GAP,
):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg_len = dash if drawing else gap
        end_pos = min(pos + seg_len, length)
        if drawing:
            draw.line(
                [
                    (x1 + dx * pos, y1 + dy * pos),
                    (x1 + dx * end_pos, y1 + dy * end_pos),
                ],
                fill=color,
                width=width,
            )
        pos = end_pos
        drawing = not drawing

def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    p1,
    p2,
    color,
    width=ARROW_WIDTH,
    head_len=ARROW_HEAD_LEN,
    head_angle=ARROW_HEAD_ANGLE,
):
    x1, y1 = p1
    x2, y2 = p2
    draw.line([p1, p2], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    for sign in (-1, 1):
        hx = x2 - head_len * math.cos(angle + sign * head_angle)
        hy = y2 - head_len * math.sin(angle + sign * head_angle)
        draw.line([p2, (hx, hy)], fill=color, width=width)

def _draw_polygon(draw: ImageDraw.ImageDraw, polygon, to_px):
    pixel_coords = [to_px(pt) for pt in polygon.exterior.coords]
    draw.polygon(pixel_coords, fill=PAPER_FILL, outline=PAPER_OUTLINE, width=PAPER_OUTLINE_WIDTH)

def _draw_punches(draw: ImageDraw.ImageDraw, points: list[Point], to_px, radius=PUNCH_RADIUS):
    for point in points:
        px, py = to_px(point)
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill=PUNCH_COLOR,
        )

def _draw_shapes(draw: ImageDraw.ImageDraw, shapes: list[ShapelyPolygon], to_px):
    for shape in shapes:
        pixel_coords = [to_px(pt) for pt in shape.exterior.coords]
        draw.polygon(pixel_coords, fill=PUNCH_COLOR)

def _render_fold_step_panel(step: FoldStep) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    to_px, padded_bounds = _make_mapper(step.polygon_before.bounds, PANEL_SIZE)
    _draw_polygon(draw, step.polygon_before, to_px)

    clipped = _clip_line_to_bounds(step.line, padded_bounds)
    if clipped is not None:
        _draw_dashed_line(draw, to_px(clipped[0]), to_px(clipped[1]), FOLD_LINE_COLOR)

    discarded = step.polygon_before.difference(step.polygon_after)
    if not discarded.is_empty and discarded.area > _MIN_DISCARDED_AREA:
        kept_centroid = step.polygon_after.centroid
        discarded_centroid = discarded.centroid
        _draw_arrow(
            draw,
            to_px((discarded_centroid.x, discarded_centroid.y)),
            to_px((kept_centroid.x, kept_centroid.y)),
            ARROW_COLOR,
        )
    return canvas

def _render_final_panel(
    final_polygon, punch_points: list[Point], punch_shapes: list[ShapelyPolygon] | None = None
) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    to_px, _ = _make_mapper(final_polygon.bounds, PANEL_SIZE)
    _draw_polygon(draw, final_polygon, to_px)
    if punch_shapes is not None:
        _draw_shapes(draw, punch_shapes, to_px)
    else:
        _draw_punches(draw, punch_points, to_px)
    return canvas

def render_question_image(
    steps: list[FoldStep],
    final_polygon,
    punch_points: list[Point],
    punch_shapes: list[ShapelyPolygon] | None = None,
) -> Image.Image:
    panels = [_render_fold_step_panel(step) for step in steps]
    panels.append(_render_final_panel(final_polygon, punch_points, punch_shapes))

    n = len(panels)
    width = n * PANEL_SIZE + (n + 1) * PANEL_PADDING
    height = PANEL_SIZE + 2 * PANEL_PADDING
    strip = Image.new("RGB", (width, height), BACKGROUND)
    for i, panel in enumerate(panels):
        x = PANEL_PADDING + i * (PANEL_SIZE + PANEL_PADDING)
        strip.paste(panel, (x, PANEL_PADDING))
    return strip

def render_option_image(
    points: list[Point],
    shapes: list[ShapelyPolygon] | None = None,
    outline_polygon: ShapelyPolygon | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    outline = outline_polygon if outline_polygon is not None else unit_square()
    to_px, _ = _make_mapper(outline.bounds, PANEL_SIZE)

    _draw_polygon(draw, outline, to_px)
    if shapes is not None:
        _draw_shapes(draw, shapes, to_px)
    else:
        _draw_punches(draw, points, to_px)
    return canvas

def save_png(image: Image.Image, path: str) -> None:
    image.save(path, format="PNG", optimize=True)

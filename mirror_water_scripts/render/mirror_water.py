from __future__ import annotations

from PIL import Image, ImageDraw
from shapely.geometry import Polygon as ShapelyPolygon

from engine.fold_punch.geometry import Line, Point

PANEL_SIZE = 200
PADDING_FRAC = 0.12
BACKGROUND = (245, 245, 245)
CANVAS_FILL = (255, 255, 255)
CANVAS_OUTLINE = (20, 20, 20)
CANVAS_OUTLINE_WIDTH = 2
MIRROR_LINE_COLOR = (200, 40, 40)
MIRROR_LINE_WIDTH = 2
MIRROR_LINE_DASH_LEN = 6
MIRROR_LINE_DASH_GAP = 5
FIGURE_FILL = (30, 60, 160)
FIGURE_OUTLINE = (20, 20, 20)

def _make_mapper(canvas_size: int):
    pad = PADDING_FRAC
    minx, maxx = -pad, 1 + pad
    miny, maxy = -pad, 1 + pad
    span = maxx - minx
    scale = canvas_size / span

    def to_px(point: Point) -> tuple[float, float]:
        x, y = point
        return ((x - minx) * scale, canvas_size - (y - miny) * scale)

    return to_px

def _draw_dashed_line(draw: ImageDraw.ImageDraw, p1, p2, color, width=MIRROR_LINE_WIDTH):
    import math

    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos, drawing = 0.0, True
    while pos < length:
        seg_len = MIRROR_LINE_DASH_LEN if drawing else MIRROR_LINE_DASH_GAP
        end_pos = min(pos + seg_len, length)
        if drawing:
            draw.line(
                [(x1 + dx * pos, y1 + dy * pos), (x1 + dx * end_pos, y1 + dy * end_pos)],
                fill=color,
                width=width,
            )
        pos = end_pos
        drawing = not drawing

def _clip_line_to_unit_square(line: Line) -> tuple[Point, Point]:
    from shapely.geometry import LineString
    from shapely.geometry import box as shapely_box

    pad = PADDING_FRAC
    rect = shapely_box(-pad, -pad, 1 + pad, 1 + pad)
    clipped = LineString(line).intersection(rect)
    coords = list(clipped.coords)
    return (coords[0], coords[-1])

def _render_panel(figure: ShapelyPolygon, mirror_line: Line) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    to_px = _make_mapper(PANEL_SIZE)

    corners = [to_px((0, 0)), to_px((1, 0)), to_px((1, 1)), to_px((0, 1))]
    draw.polygon(corners, fill=CANVAS_FILL, outline=CANVAS_OUTLINE, width=CANVAS_OUTLINE_WIDTH)

    p1, p2 = _clip_line_to_unit_square(mirror_line)
    _draw_dashed_line(draw, to_px(p1), to_px(p2), MIRROR_LINE_COLOR)

    pixel_coords = [to_px(pt) for pt in figure.exterior.coords]
    draw.polygon(pixel_coords, fill=FIGURE_FILL, outline=FIGURE_OUTLINE)
    return canvas

def render_question_image(original: ShapelyPolygon, mirror_line: Line) -> Image.Image:
    return _render_panel(original, mirror_line)

def render_option_image(candidate: ShapelyPolygon, mirror_line: Line) -> Image.Image:
    return _render_panel(candidate, mirror_line)

def save_png(image: Image.Image, path: str) -> None:
    image.save(path, format="PNG", optimize=True)

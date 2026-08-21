from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon as ShapelyPolygon

PANEL_SIZE = 200
PANEL_PADDING = 12
PADDING_FRAC = 0.12
BACKGROUND = (245, 245, 245)
CANVAS_FILL = (255, 255, 255)
CANVAS_OUTLINE = (20, 20, 20)
CANVAS_OUTLINE_WIDTH = 2
FIGURE_FILL = (30, 60, 160)
FIGURE_OUTLINE = (20, 20, 20)
MYSTERY_COLOR = (120, 120, 120)
MYSTERY_FONT_SIZE = 90

def _make_mapper(bounds: tuple[float, float, float, float], canvas_size: int):
    minx, miny, maxx, maxy = bounds
    width, height = maxx - minx, maxy - miny
    span = max(width, height, 1e-9)
    pad = span * PADDING_FRAC
    minx, maxx = minx - pad, minx + span + pad
    miny, maxy = miny - pad, miny + span + pad
    scale = canvas_size / (span + 2 * pad)

    def to_px(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return ((x - minx) * scale, canvas_size - (y - miny) * scale)

    return to_px

def _draw_polygon(draw: ImageDraw.ImageDraw, polygon: ShapelyPolygon, to_px) -> None:
    pixel_coords = [to_px(pt) for pt in polygon.exterior.coords]
    draw.polygon(pixel_coords, fill=FIGURE_FILL, outline=FIGURE_OUTLINE)

def _render_panel(figure: ShapelyPolygon) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    to_px = _make_mapper(figure.bounds, PANEL_SIZE)

    draw.rectangle([0, 0, PANEL_SIZE - 1, PANEL_SIZE - 1], outline=CANVAS_OUTLINE, width=CANVAS_OUTLINE_WIDTH)
    _draw_polygon(draw, figure, to_px)
    return canvas

def _render_mystery_panel() -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, PANEL_SIZE - 1, PANEL_SIZE - 1], outline=CANVAS_OUTLINE, width=CANVAS_OUTLINE_WIDTH)
    font = ImageFont.load_default(size=MYSTERY_FONT_SIZE)
    bbox = draw.textbbox((0, 0), "?", font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((PANEL_SIZE - text_w) / 2 - bbox[0], (PANEL_SIZE - text_h) / 2 - bbox[1]),
        "?",
        fill=MYSTERY_COLOR,
        font=font,
    )
    return canvas

def render_question_image(panels: list[ShapelyPolygon]) -> Image.Image:
    rendered = [_render_panel(p) for p in panels] + [_render_mystery_panel()]

    n = len(rendered)
    width = n * PANEL_SIZE + (n + 1) * PANEL_PADDING
    height = PANEL_SIZE + 2 * PANEL_PADDING
    strip = Image.new("RGB", (width, height), BACKGROUND)
    for i, panel in enumerate(rendered):
        x = PANEL_PADDING + i * (PANEL_SIZE + PANEL_PADDING)
        strip.paste(panel, (x, PANEL_PADDING))
    return strip

def render_option_image(candidate: ShapelyPolygon) -> Image.Image:
    return _render_panel(candidate)

def save_png(image: Image.Image, path: str) -> None:
    image.save(path, format="PNG", optimize=True)

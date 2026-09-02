from __future__ import annotations

from dataclasses import dataclass
import math
import re

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .config import EPS


_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "Arial.ttf",
)
RASTER_DASH_LENGTH_PX = 10.0
RASTER_DASH_GAP_PX = 14.0
RGBA_COLOR_PATTERN = re.compile(
    r"rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*([0-9]*\.?[0-9]+)\s*\)"
)


@dataclass(frozen=True)
class RasterViewport:
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    width: int
    height: int
    margins: dict[str, int]

    def map_point(self, point: tuple[float, float]) -> tuple[float, float]:
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        inner_left = self.margins["l"]
        inner_top = self.margins["t"]
        inner_width = max(1, self.width - self.margins["l"] - self.margins["r"])
        inner_height = max(1, self.height - self.margins["t"] - self.margins["b"])
        span_x = max(float(x1) - float(x0), EPS)
        span_y = max(float(y1) - float(y0), EPS)

        px = inner_left + ((float(point[0]) - float(x0)) / span_x) * inner_width
        py = inner_top + ((float(y1) - float(point[1])) / span_y) * inner_height
        return float(px), float(py)


def create_canvas(
    width: int,
    height: int,
    image_scale: float,
    background_color: str,
    margins: dict[str, int],
) -> tuple[Image.Image, ImageDraw.ImageDraw, float, dict[str, int]]:
    scale = max(0.1, float(image_scale))
    export_width = max(1, int(round(width * scale)))
    export_height = max(1, int(round(height * scale)))
    scaled_margins = {
        key: int(round(value * scale))
        for key, value in margins.items()
    }
    image = Image.new(
        "RGBA",
        (export_width, export_height),
        rgba(background_color),
    )
    return image, ImageDraw.Draw(image, "RGBA"), scale, scaled_margins


def rgba(color: str, alpha: int | None = 255) -> tuple[int, int, int, int]:
    match = RGBA_COLOR_PATTERN.fullmatch(color)
    if match is not None:
        red, green, blue, embedded_alpha = match.groups()
        rgba_alpha = float(embedded_alpha)
        if rgba_alpha <= 1.0:
            base_alpha = round(255 * rgba_alpha)
        else:
            base_alpha = round(rgba_alpha)
        if alpha is not None:
            base_alpha = round(base_alpha * max(0.0, min(255.0, float(alpha))) / 255.0)
        return (
            max(0, min(255, int(red))),
            max(0, min(255, int(green))),
            max(0, min(255, int(blue))),
            max(0, min(255, int(base_alpha))),
        )

    resolved_alpha = 255 if alpha is None else alpha
    return (*ImageColor.getrgb(color), max(0, min(255, int(resolved_alpha))))


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    clamped_size = max(6, int(round(size)))
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, clamped_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_round_cap(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    line_width: int,
    fill: tuple[int, int, int, int],
) -> None:
    radius = line_width / 2.0
    draw.ellipse(
        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
        fill=fill,
    )


def draw_line(
    draw: ImageDraw.ImageDraw,
    viewport: RasterViewport,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str,
    width: float,
    dash: str = "solid",
    alpha: int | None = 255,
    rounded_caps: bool = False,
) -> None:
    start = viewport.map_point(p1)
    end = viewport.map_point(p2)
    line_width = max(1, int(round(width)))
    fill = rgba(color, alpha)

    if dash == "solid":
        draw.line((start, end), fill=fill, width=line_width)
        if rounded_caps and line_width > 2:
            _draw_round_cap(draw, start, line_width, fill)
            _draw_round_cap(draw, end, line_width, fill)
        return

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1.0:
        draw.line((start, end), fill=fill, width=line_width)
        return

    dash_length = max(RASTER_DASH_LENGTH_PX, 2.5 * line_width)
    gap_length = max(RASTER_DASH_GAP_PX, 3.5 * line_width)
    step = dash_length + gap_length
    ux = dx / length
    uy = dy / length
    position = 0.0
    while position < length:
        dash_end = min(length, position + dash_length)
        segment_start = (
            start[0] + ux * position,
            start[1] + uy * position,
        )
        segment_end = (
            start[0] + ux * dash_end,
            start[1] + uy * dash_end,
        )
        draw.line((segment_start, segment_end), fill=fill, width=line_width)
        if rounded_caps and line_width > 2:
            _draw_round_cap(draw, segment_start, line_width, fill)
            _draw_round_cap(draw, segment_end, line_width, fill)
        position += step


def draw_polygon(
    draw: ImageDraw.ImageDraw,
    viewport: RasterViewport,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    fill_color: str,
) -> None:
    draw.polygon([viewport.map_point(point) for point in points], fill=rgba(fill_color))


def draw_centered_text_box(
    draw: ImageDraw.ImageDraw,
    viewport: RasterViewport,
    center: tuple[float, float],
    text: str,
    *,
    font_size: float,
    text_color: str,
    fill_color: tuple[int, int, int, int],
    outline_color: str,
    padding: float,
    spacing: float = 2.0,
) -> None:
    font = load_font(max(6, int(round(font_size))))
    anchor = viewport.map_point(center)
    text_box = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        align="center",
        spacing=max(0, int(round(spacing))),
    )
    box_padding = max(0, int(round(padding)))
    left = anchor[0] - (text_box[2] - text_box[0]) / 2 - box_padding
    top = anchor[1] - (text_box[3] - text_box[1]) / 2 - box_padding
    right = anchor[0] + (text_box[2] - text_box[0]) / 2 + box_padding
    bottom = anchor[1] + (text_box[3] - text_box[1]) / 2 + box_padding
    draw.rectangle(
        (left, top, right, bottom),
        fill=fill_color,
        outline=rgba(outline_color),
        width=1,
    )
    draw.multiline_text(
        (anchor[0], anchor[1]),
        text,
        font=font,
        fill=rgba(text_color),
        align="center",
        anchor="mm",
        spacing=max(0, int(round(spacing))),
    )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    viewport: RasterViewport,
    center: tuple[float, float],
    text: str,
    *,
    font_size: float,
    text_color: str,
    anchor: str = "mm",
) -> None:
    draw.text(
        viewport.map_point(center),
        text,
        font=load_font(max(6, int(round(font_size)))),
        fill=rgba(text_color),
        anchor=anchor,
    )


def draw_circle(
    draw: ImageDraw.ImageDraw,
    viewport: RasterViewport,
    center: tuple[float, float],
    *,
    radius: float,
    fill_color: str,
    outline_color: str,
    outline_width: float,
) -> None:
    cx, cy = viewport.map_point(center)
    circle = (
        cx - radius,
        cy - radius,
        cx + radius,
        cy + radius,
    )
    draw.ellipse(
        circle,
        fill=rgba(fill_color),
        outline=rgba(outline_color),
        width=max(1, int(round(outline_width))),
    )

"""Color helpers for paper front/back rendering."""

from __future__ import annotations

sequence_colors_dict = {
    "red": "#F03C36",
    "green": "#56F260",
    "lightyellow": "#FCF686",
    "camel": "#FFB53A",
    "yellow": "#FFF657",
    "orange": "#F26712",
    "black": "#2F393C",
    "pink": "#F877C0",
    "lightpink": "#FFE6E9",
    "brown": "#6A3816",
    "blue": "#48B9FF",
}

def resolve_paper_color(color: str) -> str:
    """Map supported sequence color names to hex values."""
    normalized = color.strip().lower()
    return sequence_colors_dict.get(normalized, color)

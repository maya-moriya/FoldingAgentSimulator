"""Shared helpers for choosing how Plotly should render a figure."""

from __future__ import annotations

import plotly.io as pio


def has_nbformat() -> bool:
    """Plotly's inline notebook renderer needs nbformat; fall back if it's missing."""
    try:
        import nbformat  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_plotly_renderer() -> str | None:
    """Pick a Plotly renderer override, or None to leave Plotly's default alone."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except Exception:
        shell = None

    if shell is None:
        return None

    if not has_nbformat():
        return "browser"

    if pio.renderers.default in {"", "browser"}:
        return "plotly_mimetype"

    return None

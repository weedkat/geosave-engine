"""Drawing geodata — one description, rendered by bokeh or matplotlib.

`elements` turns an array into a holoviews object; `style` holds the color
policy it follows. Type methods (`explore`, `plot`) are thin delegates.
"""
from .elements import (
    Kind,
    ViewOptions,
    fig_to_array,
    plot,
    resolve_kind,
    shared_limits,
    to_element,
    to_static_element,
)
from .style import DEFAULT_STYLE, RenderStyle

__all__ = [
    "DEFAULT_STYLE",
    "Kind",
    "RenderStyle",
    "ViewOptions",
    "fig_to_array",
    "plot",
    "resolve_kind",
    "shared_limits",
    "to_element",
    "to_static_element",
]

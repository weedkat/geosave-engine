"""Legend: what a label raster's pixel values mean. See Legend for details."""
from __future__ import annotations

from typing import ClassVar

from geosave_engine.geodata.extensions.base import GeoExtension
from geosave_engine.utils.colorize import Palette


class Legend(GeoExtension):
    """What a label raster's pixel values mean, and how they colour.

    Keyed to pixel values, not to bands, so it survives every operation that
    changes which bands are present — selection, renaming, reprojection.
    Band roles live in `RenderHints`, which does not.

    Args:
        class_map: `{pixel value: class name}` for a label raster.
        color_map: `{pixel value: hex or RGB}` for a label raster.

    Examples:
        >>> raster.rebase(legend={"class_map": {0: "bg", 1: "palm"}})
    """

    NAMESPACE: ClassVar[str] = "legend"

    class_map: dict[int, str] | None = None
    color_map: Palette | None = None

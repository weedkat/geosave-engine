"""GeoSave Engine's geodata surface, reached as `gs.read_raster`, `gs.raster`, `gs.stack`.

Each name is fetched from `geodata` on first use, so importing the CLI never
pays for the geospatial stack it does not touch.

Examples:
    >>> import geosave_engine as gs
    >>> gs.read_raster("scene.tif").gs.variables
    ('B04', 'B08')
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosave_engine.geodata import (
        DataArray,
        DataTree,
        Dataset,
        GeoDataFrame,
        LAYOUTS,
        GeoAnchor,
        GeoArray,
        GeoRaster,
        GeoStack,
        GeoVector,
        write_tree,
        configure_gdal,
        io,
        raster,
        read_raster,
        read_stack,
        read_vector,
        stack,
        transform,
    )

__all__ = [
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoDataFrame",
    "LAYOUTS",
    "GeoAnchor",
    "GeoArray",
    "GeoRaster",
    "GeoStack",
    "GeoVector",
    "write_tree",
    "configure_gdal",
    "io",
    "raster",
    "read_raster",
    "read_stack",
    "read_vector",
    "stack",
    "transform",
]


def __getattr__(name: str) -> object:
    """Fetch one public geodata name, importing `geodata` the first time.

    Args:
        name: Attribute being read off this package.

    Returns:
        The name as `geodata` exports it.

    Raises:
        AttributeError: `name` is not part of the public surface.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from geosave_engine import geodata

    return getattr(geodata, name)

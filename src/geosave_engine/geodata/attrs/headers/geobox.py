"""Create the coordinate attrs header an odc-geo grid describes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from odc.geo.xr import xr_coords

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS

from ..header import AttrsHeader

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def create_header(geobox: GeoBox) -> AttrsHeader:
    """Describe a grid's spatial coordinates: odc's attrs plus their CF meaning.

    Args:
        geobox: Grid carrying a CRS.

    Returns:
        Header naming only the grid's coordinates, under `y` and `x`, each
        holding odc's `units`, `resolution`, and `crs` plus CF's
        `standard_name` and `axis`.

    Raises:
        ValueError: The grid carries no CRS.

    Examples:
        >>> create_header(utm_geobox).coords["x"].to_attrs()
        {'units': 'metre', 'resolution': 10.0, 'crs': 'EPSG:32633',
         'standard_name': 'projection_x_coordinate', 'axis': 'X'}
    """
    crs = geobox.crs
    if crs is None:
        raise ValueError(
            "geobox carries no CRS, so its axes measure pixels rather than "
            "ground position; assign one before describing them"
        )
    y_dim, x_dim = SPATIAL_DIMENSIONS
    coords = xr_coords(geobox, always_yx=True)
    if crs.geographic:
        y_name, x_name = "latitude", "longitude"
    else:
        y_name, x_name = "projection_y_coordinate", "projection_x_coordinate"

    return AttrsHeader.from_attrs(
        coords={
            y_dim: {
                **coords[y_dim].attrs,
                "standard_name": y_name,
                "axis": "Y",
            },
            x_dim: {
                **coords[x_dim].attrs,
                "standard_name": x_name,
                "axis": "X",
            },
        }
    )

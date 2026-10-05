"""Create the coordinate attrs header an odc-geo grid describes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from odc.geo.xr import xr_coords

from ..header import AttrsHeader

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def create_header(geobox: GeoBox) -> AttrsHeader:
    """Describe a grid's spatial coordinates: odc's attrs plus their CF meaning.

    Args:
        geobox: Grid carrying a CRS.

    Returns:
        Header naming only the grid's coordinates, under the grid's dimension
        names, each holding odc's `units`, `resolution`, and `crs` plus CF's
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
    names = (
        ("latitude", "longitude")
        if crs.geographic
        else ("projection_y_coordinate", "projection_x_coordinate")
    )
    coords = xr_coords(geobox)
    return AttrsHeader.from_attrs(
        coords={
            str(dimension): {
                **coords[dimension].attrs,
                "standard_name": name,
                "axis": axis,
            }
            for dimension, name, axis in zip(
                geobox.dimensions, names, ("Y", "X"), strict=True
            )
        }
    )

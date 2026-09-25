"""Create CF coordinate attrs from an odc-geo grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..header import AttrsHeader

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def create_header(geobox: GeoBox) -> AttrsHeader:
    """Create the coordinate header described by a grid's CRS.

    Args:
        geobox: Grid carrying a CRS.

    Returns:
        Header whose coordinates use the grid's dimension names.

    Raises:
        ValueError: The grid carries no CRS.

    Examples:
        >>> create_header(utm_geobox).coords["y"].to_attrs()
        {'standard_name': 'projection_y_coordinate', 'units': 'metre', 'axis': 'Y'}
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
    return AttrsHeader.from_attrs(
        coords={
            str(dimension): {"standard_name": name, "units": unit, "axis": axis}
            for dimension, name, unit, axis in zip(
                geobox.dimensions, names, crs.units, ("Y", "X"), strict=True
            )
        }
    )

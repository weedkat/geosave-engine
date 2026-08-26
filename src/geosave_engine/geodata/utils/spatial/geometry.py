"""One geometry, however it was spelled, as a shapely object."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import shapely

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

# Every spelling of a geometry this library accepts — GeoJSON dict, WKT string, or shapely object.
type SomeGeometry = dict[str, Any] | str | BaseGeometry


def to_shapely(geometry: SomeGeometry) -> BaseGeometry:
    """One geometry, however it was spelled, as a shapely object.

    Args:
        geometry: GeoJSON geometry dict, WKT string, or shapely geometry.

    Returns:
        The shapely geometry.

    Raises:
        ValueError: `geometry` is WKT that can't be parsed, or is empty.
    """
    if isinstance(geometry, str):
        try:
            geom = shapely.from_wkt(geometry)
        except shapely.errors.ShapelyError as e:
            raise ValueError(f"Could not parse wkt: {e}") from e
    elif isinstance(geometry, dict):
        geom = shapely.geometry.shape(geometry)
    else:
        geom = geometry

    if geom.is_empty:
        raise ValueError("geometry must not be empty")
    return geom

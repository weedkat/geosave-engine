"""Normalize supported geometry representations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import shapely
from odc.geo.geom import Geometry as OdcGeometry
from shapely.geometry.base import BaseGeometry

type SomeGeometry = Mapping[str, Any] | str | BaseGeometry | OdcGeometry


def to_shapely(geometry: SomeGeometry) -> BaseGeometry:
    """Convert a GeoJSON mapping, WKT string, or Shapely geometry.

    Args:
        geometry: Geometry representation to normalize.

    Returns:
        Non-empty Shapely geometry.

    Raises:
        ValueError: If WKT cannot be parsed or the geometry is empty.
    """
    if isinstance(geometry, str):
        try:
            normalized = shapely.from_wkt(geometry)
        except shapely.errors.ShapelyError as error:
            raise ValueError(f"could not parse WKT: {error}") from error
    elif isinstance(geometry, Mapping):
        normalized = shapely.geometry.shape(geometry)
    elif isinstance(geometry, OdcGeometry):
        normalized = geometry.geom
    elif isinstance(geometry, BaseGeometry):
        normalized = geometry
    else:
        raise TypeError(
            "geometry must be GeoJSON, WKT, Shapely, or an ODC Geometry; "
            f"got {type(geometry).__name__}"
        )

    if normalized.is_empty:
        raise ValueError("geometry must not be empty")
    if not normalized.is_valid:
        raise ValueError(
            f"geometry must be valid: {shapely.is_valid_reason(normalized)}"
        )
    return normalized

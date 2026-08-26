"""GeoBox comparison, shared by GeoAnchor/GeoTile/GeoStack."""
from __future__ import annotations

from odc.geo.geobox import GeoBox


def geobox_matches(a: GeoBox, b: GeoBox) -> bool:
    """Check if two geoboxes describe the same pixel grid.

    Affine compared with a small float tolerance (`Affine.almost_equals`),
    not exact equality — a grid rebuilt from coordinates can differ from
    the original by float noise far below one pixel.

    Args:
        a: First geobox.
        b: Second geobox.

    Returns:
        True if a and b have the same shape, CRS, and affine.
    """
    return a.shape == b.shape and a.crs == b.crs and a.affine.almost_equals(b.affine)


def geobox_crs(geobox: GeoBox) -> str:
    """A geobox's CRS as a string, refusing the None case that would silently misplace features.

    Args:
        geobox: Pixel grid whose CRS is wanted.

    Returns:
        The geobox's CRS, e.g. `"EPSG:32633"` — a string, which geopandas
        and odc both read the same way.

    Raises:
        ValueError: `geobox` has no CRS.
    """
    if geobox.crs is None:
        raise ValueError("geobox has no CRS — can't place features on it")
    return str(geobox.crs)

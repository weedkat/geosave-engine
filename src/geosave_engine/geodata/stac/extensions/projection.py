"""Projection extension: the grid a file's pixels sit on."""

from __future__ import annotations

import pystac
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import ProjectionExtension


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write a raster's grid as Projection fields.

    Args:
        asset: Asset already added to its Item.
        raster: Raster carrying a grid with a CRS.

    Raises:
        ValueError: The raster carries no locatable grid.

    Examples:
        >>> write(asset, scene)
        >>> ProjectionExtension.ext(asset).code
        'EPSG:32749'
    """
    geobox = raster.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError(
            f"{asset.href} carries no locatable grid, which a STAC asset states; "
            f"keep it in an ordinary reference table instead"
        )
    epsg = geobox.crs.epsg
    ProjectionExtension.ext(asset, add_if_missing=True).apply(
        code=None if epsg is None else f"EPSG:{epsg}",
        wkt2=geobox.crs.to_wkt() if epsg is None else None,
        shape=list(geobox.shape),
        transform=list(geobox.transform)[:6],
    )

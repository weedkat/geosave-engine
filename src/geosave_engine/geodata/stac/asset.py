"""Build native STAC Assets describing saved raster files."""

from __future__ import annotations

from datetime import UTC, datetime as DateTime
from os import PathLike
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import pystac
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension

from geosave_engine.geodata.attrs.headers.stac import band_fields
from geosave_engine.geodata.conventions import TIME_COORDINATE
from geosave_engine.geodata.io import read_raster
from geosave_engine.geodata.io.storage import absolute_location, gdal_path

if TYPE_CHECKING:
    from geosave_engine.geodata.core.raster import RasterDriver

# What STAC calls the file each driver writes.
_DRIVER_MEDIA_TYPES = {
    "cog": pystac.MediaType.COG,
    "zarr": pystac.MediaType.ZARR,
    "netcdf": pystac.MediaType.NETCDF,
}

# What STAC calls a file read from disk, by its suffix.
_SUFFIX_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".zarr": pystac.MediaType.ZARR,
    ".nc": pystac.MediaType.NETCDF,
    ".nc4": pystac.MediaType.NETCDF,
    ".cdf": pystac.MediaType.NETCDF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
}


def default_key(raster: xr.Dataset) -> str:
    """Name the asset holding a whole raster.

    Args:
        raster: Raster one file or store holds.

    Returns:
        The raster's variable where it has exactly one, else `"image"`.
    """
    variables = raster.gs.variables
    if len(variables) == 1:
        return variables[0]
    return "image"


def from_raster(
    raster: xr.Dataset, href: str | PathLike[str], *, driver: RasterDriver
) -> pystac.Asset:
    """Describe pixels saved at `href`, from the raster that holds them.

    No file is opened, so the raster has to be the one written there.

    Args:
        raster: Raster the file or store holds.
        href: Local path or URL the raster was saved at.
        driver: Format it was saved in.

    Returns:
        Asset with an absolute href, media type, the `data` role, its grid as
        projection fields, one raster and EO band per variable, and its first
        and last instant where the raster is dated.

    Raises:
        ValueError: The raster carries no locatable grid.

    Examples:
        >>> from_raster(scene, "samples/forest.zarr", driver="zarr").media_type
        'application/vnd+zarr'
    """
    return _describe(raster, absolute_location(href), _DRIVER_MEDIA_TYPES[driver])


def from_path(path: str | PathLike[str]) -> pystac.Asset:
    """Describe one saved raster file or store from its own header.

    Args:
        path: File or store `read_raster` opens.

    Returns:
        Asset as `from_raster` builds it, over what the file itself states.

    Raises:
        ValueError: `path` is a folder or another format than a raster file or
            store, or the file carries no locatable grid.

    Examples:
        >>> from_path("samples/forest/forest_20250601T103031.tif").media_type
        'image/tiff; application=geotiff; profile=cloud-optimized'
    """
    href = absolute_location(path)
    media_type = _SUFFIX_MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if media_type is None:
        raise ValueError(
            f"{path} is not one raster file or store, which an asset is; pass "
            f"each file a writer returned"
        )

    # A GeoTIFF says in its own structure tags whether it is cloud optimized.
    if media_type == pystac.MediaType.GEOTIFF:
        with rasterio.open(gdal_path(href)) as src:
            layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        if layout == "COG":
            media_type = pystac.MediaType.COG

    with read_raster(path) as raster:
        return _describe(raster, href, media_type)


def _describe(raster: xr.Dataset, href: str, media_type: str) -> pystac.Asset:
    """Build the Asset stating a raster's grid, bands and time.

    Args:
        raster: Raster the file holds.
        href: Absolute path or URL of the file.
        media_type: STAC media type of the file.

    Returns:
        Asset carrying projection, raster and EO fields.

    Raises:
        ValueError: The raster carries no locatable grid.
    """
    geobox = raster.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError(
            f"{href} carries no locatable grid, which a STAC asset states; "
            f"keep it in an ordinary reference table instead"
        )

    asset = pystac.Asset(href, media_type=media_type, roles=["data"])

    # Projection fields state the grid; a CRS without an EPSG code is spelled as WKT.
    projection = ProjectionExtension.ext(asset)
    if geobox.crs.epsg is None:
        projection.wkt2 = geobox.crs.to_wkt()
    else:
        projection.code = f"EPSG:{geobox.crs.epsg}"
    projection.shape = list(geobox.shape)
    projection.transform = list(geobox.transform)[:6]

    # Raster fields state how each band is stored; EO fields state what it is.
    raster_bands = []
    eo_bands = []
    for name in raster.gs.variables:
        fields = band_fields(raster[name])
        raster_band = RasterBand.create(
            data_type=DataType(fields["data_type"]),
            nodata=fields.get("nodata"),
            scale=fields.get("scale"),
            offset=fields.get("offset"),
            unit=fields.get("unit"),
        )
        raster_bands.append(raster_band)
        eo_band = Band.create(
            name=fields["name"], description=fields.get("description")
        )
        eo_bands.append(eo_band)
    RasterExtension.ext(asset).bands = raster_bands
    EOExtension.ext(asset).bands = eo_bands

    # A dated raster states the first and last instant it covers, in UTC.
    start: DateTime | None = None
    end: DateTime | None = None
    if TIME_COORDINATE in raster.dims:
        span = raster.gs.timespan
        if span is not None:
            start, end = span
    elif TIME_COORDINATE in raster.coords:
        # One scene holds one instant, as a scalar coordinate at microseconds.
        label = raster[TIME_COORDINATE].values.astype("datetime64[us]")
        start = label.item()
        end = start
    if start is not None and end is not None:
        asset.common_metadata.start_datetime = start.replace(tzinfo=UTC)
        asset.common_metadata.end_datetime = end.replace(tzinfo=UTC)
    return asset

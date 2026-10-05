"""What one STAC item row holds."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from os import PathLike
from pathlib import PurePosixPath

import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from typing import TYPE_CHECKING, cast
from odc.geo.geom import Geometry

from geosave_engine.geodata.attrs.models.stac import StacMetadata
from geosave_engine.geodata.utils.datetime import naive_utc
from geosave_engine.geodata.utils.geo.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    from geosave_engine.geodata import GeoDataFrame

# One raster of a record: where it is stored, or a STAC asset stating `href`.
type Asset = str | PathLike[str] | Mapping[str, object]

STAC_VERSION = "1.1.0"
PROJECTION_EXTENSION = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
MEDIA_TYPES = {
    ".tif": "image/tiff; application=geotiff",
    ".tiff": "image/tiff; application=geotiff",
    ".zarr": "application/vnd+zarr",
    ".nc": "application/x-netcdf",
    ".nc4": "application/x-netcdf",
    ".cdf": "application/x-netcdf",
}
# Every column a GeoSave STAC table owns, in the order a row states them.
ITEM_COLUMNS = (
    "id",
    "type",
    "stac_version",
    "stac_extensions",
    "links",
    "datetime",
    "start_datetime",
    "end_datetime",
    "proj:code",
    "proj:wkt2",
    "proj:shape",
    "proj:transform",
    "assets",
    "sources",
    "bbox",
    "geometry",
)


def file_assets(data: xr.Dataset | xr.DataTree) -> dict[str, dict[str, object]]:
    """Recover each opened raster's source href and optional group selector."""
    rasters = data.gs.rasters if isinstance(data, xr.DataTree) else {"": data}
    assets = {}
    for name, raster in rasters.items():
        source = raster.encoding.get("source")
        if source is None:
            raise ValueError(
                f"{name or 'this raster'} was not read from a file; "
                "write it, then register the path with GeoVector.from_assets"
            )
        fields = {"href": source}
        if group := raster.encoding.get("group"):
            fields["group"] = group
        assets[name or PurePosixPath(str(source)).stem] = fields
    return assets


def record(
    stored: xr.Dataset | xr.DataTree,
    *,
    assets: Mapping[str, Asset],
    id: str | None = None,
    datetime: dt.datetime | None = None,
    geometry: SomeGeometry | None = None,
    properties: Mapping[str, object] | None = None,
) -> GeoDataFrame:
    """Build a catalog row from an authoritative opened storage snapshot.

    Callers own opening, validation and closing; this function reads metadata
    only. It does not verify that a runtime object still matches its file.
    """
    properties = properties or {}
    collisions = sorted(set(properties) & (set(ITEM_COLUMNS) | {"raster_metadata"}))
    if collisions:
        raise ValueError(f"properties collide with item columns {collisions}")
    metadata = raster_metadata(stored)
    if isinstance(stored, xr.Dataset):
        metadata = {name: metadata["image"] for name in assets}
    row = {
        **item(stored, assets=assets, id=id, datetime=datetime),
        "raster_metadata": metadata,
        **properties,
    }
    footprint = stored.gs.anchor.geobox.extent if geometry is None else geometry
    columns = {name: pd.Series([value]) for name, value in row.items()}
    for name in ("datetime", "start_datetime", "end_datetime"):
        value = row[name]
        instant = None if value is None else naive_utc(cast("dt.datetime", value))
        columns[name] = pd.Series(pd.to_datetime([instant], utc=True))
    crs = (footprint.crs if isinstance(footprint, Geometry) else None) or "EPSG:4326"
    frame = gpd.GeoDataFrame(columns, geometry=[to_shapely(footprint)], crs=crs)
    return cast("GeoDataFrame", frame.to_crs("EPSG:4326"))


def item(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
    *,
    assets: Mapping[str, Asset],
    id: str | None = None,
    datetime: dt.datetime | None = None,
) -> dict[str, object]:
    """Describe one geolocated xarray object as a STAC item row.

    Args:
        data: Geolocated array, raster, or single-grid stack.
        assets: Where each raster is stored, by layer name. A value is a path
            or URL, or a STAC asset mapping stating `href`. A key names a
            layer of a stack; a lone raster takes any key.
        id: Item identifier. None uses the anchor stem.
        datetime: Item instant. None leaves it null, which the timespan in
            `start_datetime` and `end_datetime` then stands in for.

    Returns:
        The row's columns in STAC order, without its geometry.

    Raises:
        ValueError: `data` has no shared locatable grid, an asset names no
            layer or states no href, or the item has no time.

    Examples:
        >>> item(scene, assets={"image": "scene.tif"})["proj:code"]
        'EPSG:32633'
    """
    anchor = data.gs.anchor
    span = anchor.timespan
    if datetime is None and span is None:
        raise ValueError(
            "this raster carries no time and a STAC item needs one; pass datetime="
        )

    epsg = anchor.crs.epsg
    # STAC names a grid CRS by authority code, falling back to its WKT.
    grid: dict[str, object] = (
        {"proj:code": f"EPSG:{epsg}"}
        if epsg is not None
        else {"proj:wkt2": anchor.crs.to_wkt()}
    )
    return {
        "id": anchor.stem if id is None else id,
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [PROJECTION_EXTENSION],
        "links": [],
        "datetime": datetime,
        "start_datetime": None if span is None else span[0],
        "end_datetime": None if span is None else span[1],
        **grid,
        "proj:shape": (anchor.geobox.height, anchor.geobox.width),
        "proj:transform": tuple(anchor.geobox.transform)[:6],
        "assets": _assets(data, assets),
        "sources": sources(data),
    }


def sources(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
) -> list[dict[str, object]] | None:
    """List the provider items a raster or stack was loaded from.

    Args:
        data: Raster, band, or stack carrying `StacMetadata` in its attrs.

    Returns:
        One entry per provider item across every layer, each once, holding
        its captured properties, its id, and its datetime as ISO text. None
        where no layer was loaded from a catalog.

    Examples:
        >>> [entry["id"] for entry in sources(sample)]
        ['S2A_T49MHM_20250603', 'S2A_T49MHM_20250608']
    """
    rasters = (
        list(data.gs.rasters.values()) if isinstance(data, xr.DataTree) else [data]
    )
    loaded = [
        model
        for model in (raster.gs.attrs.root.get(StacMetadata) for raster in rasters)
        if model is not None
    ]
    if not loaded:
        return None
    merged, _ = StacMetadata.merge(loaded)
    return [
        {**entry.properties, "id": entry.id, "datetime": entry.datetime.isoformat()}
        for entry in merged.stac_items or ()
    ]


def _assets(
    data: xr.DataArray | xr.Dataset | xr.DataTree, assets: Mapping[str, Asset]
) -> dict[str, dict[str, object]]:
    """Describe each stored raster as a STAC asset."""
    if isinstance(data, xr.DataTree):
        layers = data.gs.rasters
        unknown = [name for name in assets if name not in layers]
        if unknown:
            raise ValueError(
                f"assets {unknown} name no layer of this stack; its layers are "
                f"{list(layers)}"
            )
        bands = {name: layers[name].gs.variables for name in assets}
    else:
        bands = {name: data.gs.variables for name in assets}

    stac_assets: dict[str, dict[str, object]] = {}
    for name, asset in assets.items():
        fields = dict(asset) if isinstance(asset, Mapping) else {"href": asset}
        if "href" not in fields:
            raise ValueError(f"asset {name!r} states no href")
        href = fields["href"]
        if isinstance(href, PathLike):
            href = os.fspath(href)
        defaults: dict[str, object] = {"href": href}
        media = MEDIA_TYPES.get(PurePosixPath(str(href)).suffix.lower())
        if media is not None:
            defaults["type"] = media
        defaults["roles"] = ["data"]
        defaults["bands"] = [{"name": band} for band in bands[name]]
        # Fields the caller states win over the ones filled in here.
        stac_assets[name] = {**defaults, **fields, "href": href}
    return stac_assets


def raster_metadata(
    data: xr.Dataset | xr.DataArray | xr.DataTree,
) -> dict[str, object]:
    """Describe acquisition order and bands without reading pixels.

    Args:
        data: Native raster or stack supplying time coordinates and band names.

    Returns:
        Metadata by raster name, retaining each raster's timestamp order.
    """
    rasters = data.gs.rasters if isinstance(data, xr.DataTree) else {"image": data}
    metadata = {}
    for name, raster in rasters.items():
        times = raster.coords.get("time")
        labels = [] if times is None else np.atleast_1d(times.values)
        metadata[name] = {
            "times": np.datetime_as_string(labels).tolist()
            if times is not None and np.issubdtype(times.dtype, np.datetime64)
            else list(labels),
            "bands": list(raster.data_vars)
            if isinstance(raster, xr.Dataset)
            else [raster.name],
        }
    return metadata

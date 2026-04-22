"""DynamicWorld-specific ingest helpers.

Contains only the DW pipeline's row-to-query mapping and the EarthSearch
asset normalization shim.  Generic Sentinel-2 masking and manifest I/O
live in geosave_engine.ingestion.
"""
from __future__ import annotations

import os
from datetime import timedelta

# Must be set before any rasterio/GDAL import.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_TCP_KEEPALIVE", "YES")
os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_RETRY_COUNT", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_NUM_THREADS", "1")

import pystac
import rioxarray  # noqa: F401
import shapely.geometry

from geosave_engine.stac_query import Sentinel2Query
from geosave_engine.utils.datetime import datetime_range_buffer
from geosave_engine.utils.geom import wkt_to_geojson

E84_ASSET_MAP: dict[str, str] = {
    "coastal": "B01",
    "blue":    "B02",
    "green":   "B03",
    "red":     "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir":     "B08",
    "nir08":   "B8A",
    "nir09":   "B09",
    "cirrus":  "B10",
    "swir16":  "B11",
    "swir22":  "B12",
}


def query_from_dw_row(row, *, include_orbit_state: bool) -> Sentinel2Query:
    """Build a Sentinel-2 STAC query from a DynamicWorld row.

    No MGRS grid:code filter — boundary samples straddle two tiles and need
    both returned. The bbox + datetime pair is enough to find every tile
    overlapping the AOI on the target acquisition day.
    """
    dt_range = datetime_range_buffer(
        row["date"],
        delta_before=timedelta(days=1),
        delta_after=timedelta(days=1),
    )
    bbox = shapely.geometry.shape(wkt_to_geojson(str(row["geometry"]), str(row["crs"]))).bounds

    query = Sentinel2Query(
        collections=["sentinel-2-l1c"],
        bbox=bbox,
        datetime=dt_range,
    )

    if include_orbit_state:
        query = query.orbit_state(row["S2_SENSING_ORBIT_DIRECTION"])
    return query


def normalize_and_fix_e84_item(item: pystac.Item) -> pystac.Item:
    """Map EarthSearch asset keys to Bxx names and force L1C href bucket."""
    renamed: dict[str, pystac.Asset] = {}
    for e84_key, band in E84_ASSET_MAP.items():
        if e84_key not in item.assets:
            continue
        asset = item.assets[e84_key]
        if "sentinel-s2-l2a" in asset.href:
            asset.href = asset.href.replace("sentinel-s2-l2a", "sentinel-s2-l1c")
        renamed[band] = asset
    if renamed:
        item.assets = renamed
    return item

"""TIFF file utilities: metadata reading, datetime parsing, filename conventions."""
from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyproj
import rasterio
import shapely.geometry
import shapely.ops

# Filename convention: <prefix>_<lon>_<lat>-<YYYYMMDD>.tif
# Must match Sentinel2L1C.filename_regex used by TorchGeo dataset discovery.
_TIFF_ID_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9_]+?)"
    r"_(?P<lon>-?\d+(?:\.\d+)?)"
    r"_(?P<lat>-?\d+(?:\.\d+)?)"
    r"-(?P<date>\d{8})\.tif$"
)
_TIFF_DATE_RE = re.compile(r"(?<!\d)(?P<date>\d{8})(?!\d)")


@dataclasses.dataclass(frozen=True)
class TiffMetadata:
    """Spatial and temporal metadata extracted from a raster TIFF file."""

    datetime: datetime
    crs: str                                    # e.g. "EPSG:32636"
    bounds: tuple[float, float, float, float]   # (minx, miny, maxx, maxy) in native CRS
    geometry: Any                               # shapely Polygon in WGS84


@dataclasses.dataclass(frozen=True)
class TiffId:
    """Components encoded in a canonical geosave TIFF filename."""

    prefix: str
    lon: float
    lat: float
    date: datetime


def build_tiff_filename(meta: "TiffMetadata", prefix: str) -> str:
    """Format a canonical TIFF filename: ``<prefix>_<lon>_<lat>-<YYYYMMDD>.tif``.

    Longitude/latitude are derived from the centroid of ``meta.geometry`` (WGS84);
    the date is taken from ``meta.datetime``.
    """
    if "_" in prefix or "-" in prefix:
        raise ValueError(f"prefix may not contain '_' or '-': {prefix!r}")
    centroid = meta.geometry.centroid
    return f"{prefix}_{centroid.x:.10f}_{centroid.y:.10f}-{meta.datetime.strftime('%Y%m%d')}.tif"


def parse_tiff_id(path: Path) -> TiffId:
    """Parse a canonical TIFF filename into its components."""
    match = _TIFF_ID_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"filename does not match canonical TIFF pattern: {Path(path).name!r}")
    return TiffId(
        prefix=match["prefix"],
        lon=float(match["lon"]),
        lat=float(match["lat"]),
        date=datetime.strptime(match["date"], "%Y%m%d").replace(tzinfo=timezone.utc),
    )


def parse_tiff_datetime(path: Path) -> datetime:
    """Parse acquisition date from TIFF filename suffix.

    Accepts canonical names like ``dw_<lon>_<lat>-YYYYMMDD.tif`` and tolerant
    variants such as macOS AppleDouble sidecars (``._`` prefix) or auxiliary
    suffixes after the date token.
    """
    name = Path(path).name
    stem = name[2:] if name.startswith("._") else Path(path).stem
    match = _TIFF_DATE_RE.search(stem)
    if not match:
        raise ValueError(f"cannot parse acquisition date from TIFF filename: {name!r}")
    try:
        return datetime.strptime(match["date"], "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"cannot parse acquisition date from TIFF filename: {name!r}") from exc


def read_tiff_metadata(path: Path) -> TiffMetadata:
    """Read CRS, bounds, and WGS84 geometry from a TIFF. Datetime is parsed from the filename."""
    path = Path(path)
    acquisition_dt = parse_tiff_datetime(path)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"TIFF has no CRS metadata: {path}")
        crs    = src.crs.to_string()
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        geometry    = shapely.ops.transform(
            transformer.transform,
            shapely.geometry.box(*bounds),
        )
    return TiffMetadata(datetime=acquisition_dt, crs=crs, bounds=bounds, geometry=geometry)

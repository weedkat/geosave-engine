"""TIFF file utilities: metadata reading and datetime parsing."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyproj
import rasterio
import shapely.geometry
import shapely.ops


@dataclasses.dataclass(frozen=True)
class TiffMetadata:
    """Spatial and temporal metadata extracted from a raster TIFF file."""

    datetime: datetime
    crs: str                                    # e.g. "EPSG:32636"
    bounds: tuple[float, float, float, float]   # (minx, miny, maxx, maxy) in native CRS
    geometry: Any                               # shapely Polygon in WGS84


def parse_tiff_datetime(path: Path) -> datetime:
    """Parse acquisition date from TIFF filename suffix (expects …-YYYYMMDD.tif)."""
    date_token = Path(path).stem.rsplit("-", 1)[-1]
    try:
        return datetime.strptime(date_token, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(
            f"cannot parse acquisition date from TIFF filename: {Path(path).name!r}"
        ) from exc


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

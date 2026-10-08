"""Shared Sentinel-2 builder for feature tests."""

from __future__ import annotations

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster

BANDS = ("B01", "B02", "B04", "B05", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


def sentinel(chunks: dict[str, int] | None = None) -> xr.Dataset:
    """Build prepared reflectance, an SCL band, and per-date sun azimuth.

    Args:
        chunks: Dask chunking to apply, or None to stay eager.

    Returns:
        Two dates of 32 by 32 pixels at 10 m, every band carrying a
        `long_name` so a test can see whether it leaks.
    """
    grid = GeoBox.from_bbox((0, 0, 320, 320), crs="EPSG:32633", resolution=10)
    values = np.random.default_rng(4).uniform(0.1, 0.8, (2, 32, 32)).astype("float32")
    dims = ("time", "y", "x")
    built = raster(
        {band: (dims, values + 0.01 * offset) for offset, band in enumerate(BANDS)}
        | {"SCL": (dims, np.full(values.shape, 4, dtype="uint8"))},
        grid,
        coords={"time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")},
    )
    built = built.assign_coords(sun_azimuth=("time", [45.0, 135.0]), site="plot-1")
    built.sun_azimuth.attrs["units"] = "degree"
    for name in built.data_vars:
        built[name].attrs["long_name"] = f"band {name}"
    return built if chunks is None else built.chunk(chunks)

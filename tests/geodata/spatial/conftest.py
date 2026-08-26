"""Builders for Spatial tests — every fixture is pure geometry, no network, no files.

Grids are built from an explicit Affine rather than a constructor so a test
can state the exact pixel origin it needs; `anchor="edge"` snapping would
otherwise move an origin a test is asserting about.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.spatial import GeoAnchor, GeoRaster, GeoVector

UTM = "EPSG:32633"
ORIGIN_X = 500_000.0
ORIGIN_Y = 5_000_000.0


def make_geobox(
    *,
    width: int = 64,
    height: int = 64,
    resolution: float = 10.0,
    x0: float = ORIGIN_X,
    y0: float = ORIGIN_Y,
    crs: str = UTM,
) -> GeoBox:
    """North-up grid whose lower-left corner sits exactly at `(x0, y0)`."""
    return GeoBox((height, width), Affine(resolution, 0, x0, 0, -resolution, y0 + height * resolution), crs)


def make_anchor(*, time: Any = "2024-01-15", **grid: Any) -> GeoAnchor:
    """Anchor over `make_geobox(**grid)`, dated unless `time=None`."""
    return GeoAnchor(geobox=make_geobox(**grid)).rebase(timespan=time)


def make_raster(
    *,
    bands: Sequence[str] = ("B04", "B08"),
    dtype: str = "uint16",
    nodata: float | int | None = 0,
    times: Sequence[datetime] | None = None,
    vector: GeoVector | None = None,
    time: Any = "2024-01-15",
    **grid: Any,
) -> GeoRaster:
    """In-memory raster of ones over `make_geobox(**grid)`.

    Passing `times` makes it four-dimensional `(time, band, y, x)`.
    """
    anchor = make_anchor(time=time, **grid)
    shape: tuple[int, ...] = (len(bands), anchor.height, anchor.width)
    if times is not None:
        shape = (len(times), *shape)
    pixels = np.ones(shape, dtype=dtype)
    raster = anchor.to_raster(pixels, bands=list(bands), times=list(times) if times else None)
    if nodata is not None:
        raster = raster.rebase(nodata=nodata)
    return raster if vector is None else raster.rebase(vector=vector)


def make_lazy_raster(
    *,
    bands: Sequence[str] = ("B04",),
    dtype: str = "uint16",
    nodata: float | int | None = 0,
    chunk: int = 16,
    **grid: Any,
) -> GeoRaster:
    """Dask-backed raster, for asserting an operation never materializes pixels."""
    import dask.array as dsk

    anchor = make_anchor(**grid)
    data = xr.DataArray(
        dsk.ones((len(bands), anchor.height, anchor.width), dtype=dtype, chunks=(len(bands), chunk, chunk)),
        dims=("band", "y", "x"),
        coords={**dict(xr_coords(anchor.geobox, always_yx=True)), "band": list(bands)},
    )
    raster = anchor.to_raster(data)
    return raster if nodata is None else raster.rebase(nodata=nodata)


def make_vector(*, x0: float = ORIGIN_X + 100, y0: float = ORIGIN_Y + 100, size: float = 100) -> GeoVector:
    """One square polygon inside the default grid's extent, on `UTM`."""
    ring = [
        [x0, y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0, y0 + size],
        [x0, y0],
    ]
    return GeoVector.from_geometry({"type": "Polygon", "coordinates": [ring]}, crs=UTM)


def is_lazy(raster: GeoRaster) -> bool:
    """Whether a raster's pixels are still an unevaluated dask graph."""
    return hasattr(raster.data.data, "dask")


@pytest.fixture
def geobox() -> GeoBox:
    return make_geobox()


@pytest.fixture
def anchor() -> GeoAnchor:
    return make_anchor()


@pytest.fixture
def raster() -> GeoRaster:
    return make_raster()


@pytest.fixture
def vector() -> GeoVector:
    return make_vector()

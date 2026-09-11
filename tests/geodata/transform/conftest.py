from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords


UTM = "EPSG:32633"


def geobox(
    bbox: tuple[float, float, float, float] = (
        300_000.0,
        5_000_000.0,
        300_320.0,
        5_000_320.0,
    ),
    *,
    crs: str = UTM,
    resolution: float = 10.0,
) -> GeoBox:
    """Build one axis-aligned grid for a transform test.

    Args:
        bbox: Extent as `(left, bottom, right, top)` in `crs` units.
        crs: CRS the grid sits in.
        resolution: Pixel size in `crs` units.

    Returns:
        Grid covering `bbox`, snapped to `resolution`.
    """
    return GeoBox.from_bbox(bbox, crs=crs, resolution=resolution)


def build(
    box: GeoBox,
    start: str = "2024-01-01",
    *,
    times: int = 2,
    packed: bool = True,
    labelled: bool = False,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Build one raster on `box` for a transform test.

    Args:
        box: Grid the raster sits on.
        start: First time label, one day apart from there.
        times: Length of the time axis.
        packed: Give `red` CF packing over stored digital numbers.
        labelled: Add a categorical `cls` variable carrying a class map.
        chunks: Dask chunking to apply, or None to stay eager.

    Returns:
        Raster Dataset holding `red`, and `cls` when `labelled`.
    """
    shape = (times, *box.shape)
    data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "red": (("time", *box.dimensions), np.full(shape, 1000, "uint16"))
    }
    if labelled:
        data["cls"] = (("time", *box.dimensions), np.full(shape, 1, "uint8"))

    coords = dict(xr_coords(box))
    coords["time"] = xr.DataArray(
        pd.date_range(start, periods=times, freq="D").values, dims="time"
    )
    raster = xr.Dataset(data, coords=coords, attrs={"title": "scene", "license": "CC0"})
    raster["red"].attrs.update({"long_name": "Red", "units": "1"})
    if packed:
        raster["red"].attrs["scale_factor"] = 1e-4
    if labelled:
        raster["cls"].attrs["class_map"] = {0: "bg", 1: "crop"}

    stamped = raster.gs.write_crs()
    return stamped.chunk(chunks) if chunks else stamped

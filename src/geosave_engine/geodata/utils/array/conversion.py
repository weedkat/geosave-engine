"""Bind raw numpy pixels to a geobox and explicit coordinates."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as dt
from typing import Any

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords


def bind_pixels(
    geobox: GeoBox,
    array: np.ndarray,
    *,
    bands: Sequence[str],
    times: Sequence[dt] | None = None,
) -> xr.DataArray:
    """Put raw pixels on a geobox with explicit band and time coordinates.

    Two-dimensional input becomes one named band. Three-dimensional
    input is `(band, y, x)`. Four-dimensional input is
    `(time, band, y, x)` and requires `times`.

    Args:
        geobox: Spatial reference for the array's (y, x) axes.
        array: NumPy array with 2-4 dimensions; last axes are `(y, x)`.
        bands: One name per band. A 2D array requires exactly one.
        times: Observation datetimes for a 4D array.

    Returns:
        DataArray with dims/coords matching the array's shape.

    Raises:
        ValueError: Shape doesn't match the geobox, dimensionality is
            unsupported, or band/time coordinates do not match it.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Raw pixels must be a numpy.ndarray, got {type(array).__name__}")
    if isinstance(bands, str):
        raise TypeError("bands must be a sequence of names; use ('class',) for one band")

    arr = array
    if arr.ndim not in (2, 3, 4):
        raise ValueError(f"Expected a 2D, 3D, or 4D array, got {arr.ndim}D")
    if arr.shape[-2:] != geobox.shape:
        raise ValueError(f"Pixel grid is {arr.shape[-2:]}, but anchor grid is {geobox.shape}")

    band_names = list(bands)
    base_coords: dict[Any, Any] = dict(xr_coords(geobox, always_yx=True))
    if arr.ndim == 2:
        if len(band_names) != 1:
            raise ValueError(f"2D pixels require exactly one band name, got {len(band_names)}")
        if times is not None:
            raise ValueError("times only applies to 4D pixels shaped (time, band, y, x)")
        arr = arr[np.newaxis, ...]
        return xr.DataArray(
            arr,
            dims=("band", "y", "x"),
            coords={**base_coords, "band": band_names},
        )

    if arr.ndim == 3:
        if len(band_names) != arr.shape[0]:
            raise ValueError(f"Expected {arr.shape[0]} bands, got {len(band_names)}")
        if times is not None:
            raise ValueError("times only applies to 4D pixels shaped (time, band, y, x)")
        return xr.DataArray(arr, dims=("band", "y", "x"), coords={**base_coords, "band": band_names})

    if times is None:
        raise ValueError("times is required for 4D pixels shaped (time, band, y, x)")
    if len(band_names) != arr.shape[1]:
        raise ValueError(f"Expected {arr.shape[1]} bands, got {len(band_names)}")
    if len(times) != arr.shape[0]:
        raise ValueError(f"Expected {arr.shape[0]} times, got {len(times)}")
    time_coord = [np.datetime64(value, "ns") for value in times]
    return xr.DataArray(
        arr,
        dims=("time", "band", "y", "x"),
        coords={**base_coords, "band": band_names, "time": time_coord},
    )

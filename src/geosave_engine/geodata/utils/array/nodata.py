"""Nodata masking for xr.DataArray."""
from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr


def same_nodata(left: float | int | None, right: float | int | None) -> bool:
    """Whether two nodata declarations match, treating NaN as equal."""
    if left is None or right is None:
        return left is right
    return bool(left == right or (np.isnan(left) and np.isnan(right)))


def cast_nodata(value: float | int | None, dtype: np.dtype) -> float | int | None:
    """Represent one nodata value in a target pixel dtype.

    Args:
        value: Nodata sentinel, NaN included, or None.
        dtype: Target numpy dtype.

    Returns:
        Sentinel expressed as a scalar of `dtype`, or None.

    Raises:
        ValueError: The target dtype cannot represent the sentinel.
    """
    if value is None:
        return None

    if np.issubdtype(dtype, np.integer):
        if not np.isfinite(value) or int(value) != value:
            raise ValueError(f"nodata {value!r} is not an integer representable by {dtype}")
        limits = np.iinfo(dtype)
        if not limits.min <= value <= limits.max:
            raise ValueError(f"nodata {value!r} is outside {dtype}'s {limits.min}..{limits.max} range")
        return int(value)

    if np.issubdtype(dtype, np.floating):
        converted = dtype.type(value)
        if np.isfinite(value) and not np.isfinite(converted):
            raise ValueError(f"nodata {value!r} is outside {dtype}'s finite range")
        return float(converted)

    raise ValueError(f"nodata is only supported for integer or floating pixels, got {dtype}")


def mask_nodata(da: xr.DataArray) -> xr.DataArray:
    """Turn da's own declared nodata pixels into real NaN.

    `.where()` promotes dtype only as far as needed to hold NaN safely
    (uint8 -> float32, int64 -> float64, ...), not a blind float32 cast.

    Args:
        da: Array to mask.

    Returns:
        da unchanged if it has no declared nodata or its nodata already is NaN.
    """
    nodata = da.rio.nodata
    # np.isnan handles any numeric dtype (float32/float64/int) — an isinstance(float) guard here missed np.float32, the common on-disk dtype.
    if nodata is None or np.isnan(nodata):
        return da
    return da.where(da != nodata)  # fill with NaN where da equals its own nodata

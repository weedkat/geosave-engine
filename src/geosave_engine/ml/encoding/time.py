"""Validation shared by model inputs that read acquisition times."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

import geosave_engine.geodata  # noqa: F401 — registers the .gs accessors


def time_labels(data: xr.Dataset | xr.DataArray) -> list[datetime]:
    """Read finite datetime labels in their original frame order.

    Args:
        data: Raster or band with a scalar or one-dimensional time coordinate.

    Returns:
        One timestamp per frame, without interpreting temporal bucket bounds.

    Raises:
        TypeError: Input is not a Dataset or DataArray.
        ValueError: Time labels are missing, empty, invalid, or not datetimes.
    """
    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError("Select a Dataset or DataArray before extracting model inputs")
    if "time" not in data.coords:
        raise ValueError("Model inputs require a 'time' coordinate")
    values = data.coords["time"].values
    if values.ndim > 1 or values.size == 0:
        raise ValueError("Model inputs require non-empty scalar or 1D time labels")
    if not np.issubdtype(values.dtype, np.datetime64) or np.isnat(values).any():
        raise ValueError("Model inputs require datetime64 time labels without NaT")
    return [
        datetime.fromisoformat(str(np.datetime_as_string(value, unit="us")))
        for value in np.atleast_1d(values)
    ]

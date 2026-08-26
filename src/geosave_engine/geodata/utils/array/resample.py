"""Time-bucketing reduction for xr.DataArray."""
from __future__ import annotations

from datetime import datetime as dt
from datetime import timedelta
from typing import Callable, Literal

import numpy as np
import xarray as xr

from geosave_engine.geodata.utils.array.nodata import mask_nodata
from geosave_engine.geodata.utils.datetime import Freq, freq_offset

ReduceMethod = Literal["first", "last", "max", "min", "mean", "median", "sum", "std", "var", "count"]

# resample_time methods safe to restore to the original dtype/nodata after resampling — never a blend.
SELECTOR_METHODS = frozenset({"first", "last", "max", "min"})
_REDUCE_METHODS = SELECTOR_METHODS | frozenset({"mean", "median", "sum", "std", "var", "count"})


def resample_time(
    da: xr.DataArray,
    freq: Freq,
    method: ReduceMethod | Callable[..., xr.DataArray] = "last",
    *,
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    origin: str | dt = "start_day",
    offset: str | timedelta | None = None,
    restore_coord_dims: bool = False,
) -> xr.DataArray:
    """Bucket da's time dim to freq, collapsing each bucket to one time step.

    Thin wrapper over `xr.DataArray.resample(time=freq)`. Nodata masked to
    NaN first (`mask_nodata`), so it never pollutes a bucket's aggregate —
    this also makes a named selector method (first/last/max/min) skip a
    nodata step and surface the next real one in its bucket, resample's
    own `skipna=True` default. A bucket with zero real observations comes
    back NaN — genuinely missing, not fabricated; `.resample()` always
    fills every bucket across da's own time range, gaps included, never
    drops one for being empty. For a selector method — always a literal
    source value, never a blend — the result is cast back to da's own
    dtype/nodata when it had one declared; any other method (mean/median/
    sum/std/var/count/a custom callable) stays promoted float with NaN
    nodata, since a blended value may not fit the original dtype.

    Args:
        da: Array to bucket, needs a `time` dim.
        freq: Target cadence — a base unit ("D", "W", "M", "Q", "Y") or a
            multiple of one, e.g. `"5D"`. Month/quarter/year bucket labels
            land on that bucket's own end date (e.g. "Q" labels a quarter
            with its last day, not its first) under the default closed/label.
        method: Named DataArrayResample reduction, or a callable passed to `.reduce()`.
        closed: Which bucket edge is inclusive. None uses `.resample()`'s own per-freq default.
        label: Which bucket edge becomes its coordinate. None uses `.resample()`'s own per-freq default.
        origin: Anchor point bucket edges snap to — forwarded to `.resample()`'s own `origin` kwarg.
        offset: Shift `origin` further — forwarded to `.resample()`'s own `offset` kwarg.
        restore_coord_dims: Forwarded to `.resample()` as-is.

    Returns:
        New DataArray, one time step per bucket, chronological.

    Raises:
        ValueError: da has no `time` dim, freq isn't a base unit or a
            multiple of one, or method is a string outside `ReduceMethod`.
    """
    if "time" not in da.dims:
        raise ValueError("resample_time() needs da to have a 'time' dim")
    offset_alias = freq_offset(freq)
    if isinstance(method, str) and method not in _REDUCE_METHODS:
        raise ValueError(f"Unknown resample_time method {method!r}, expected one of {sorted(_REDUCE_METHODS)} or a callable")

    original_dtype = da.dtype
    original_nodata = da.rio.nodata

    resampled = mask_nodata(da).resample(
        time=offset_alias, closed=closed, label=label,
        origin=origin, offset=offset, restore_coord_dims=restore_coord_dims,
    )
    reduced = getattr(resampled, method)() if isinstance(method, str) else resampled.reduce(method)

    # NaN only eligible for float
    if isinstance(method, str) and method in SELECTOR_METHODS and original_nodata is not None:
        reduced = reduced.fillna(original_nodata).astype(original_dtype).rio.write_nodata(original_nodata, inplace=True)
    elif np.issubdtype(reduced.dtype, np.floating):
        reduced = reduced.rio.write_nodata(np.nan, inplace=True)

    return reduced

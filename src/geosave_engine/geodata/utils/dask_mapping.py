"""Run a NumPy function over spatial arrays with chunk overlap, lazily."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS, TIME_COORDINATE


def map_spatial_overlap(
    func: Callable[..., np.ndarray],
    *fields: xr.DataArray,
    depth: int,
    dtype: str | np.dtype[Any],
    boundary: str | int | float | bool = "reflect",
    **func_kwargs: Any,
) -> xr.DataArray:
    """Map a NumPy function over spatial arrays with chunk overlap.

    NumPy-backed fields run eagerly. Dask-backed fields use overlap on `y` and
    `x` and retain the first field's coordinates.

    Args:
        func: Function mapping equally shaped NumPy blocks to one output block.
        *fields: Bands selected from one Dataset, with the same dimensions.
            Dimension order follows the first field, which uses `y`, `x`,
            optionally preceded by `time`.
        depth: Halo width in pixels on each spatial side.
        dtype: Output dtype.
        boundary: Value or policy used to pad Dask chunk boundaries.
        **func_kwargs: Literal keyword arguments passed to `func`.

    Returns:
        DataArray on the first field's dimensions and coordinates.

    Raises:
        ValueError: Depth is negative, inputs use unsupported dimensions, or
            spatial chunks are smaller than `depth`.
    """
    if depth < 0:
        raise ValueError(f"Halo depth must not be negative, got {depth}")
    reference = fields[0]
    dims = tuple(str(dim) for dim in reference.dims)
    if dims not in (SPATIAL_DIMENSIONS, (TIME_COORDINATE, *SPATIAL_DIMENSIONS)):
        raise ValueError(
            f"Field has dimensions {dims}; expected {SPATIAL_DIMENSIONS}, "
            f"optionally preceded by {TIME_COORDINATE!r}"
        )

    # A Dataset shares indexes but allows each variable its own dimension order.
    fields = tuple(field.transpose(*reference.dims) for field in fields)
    if not any(isinstance(field.data, da.Array) for field in fields):
        values: np.ndarray | da.Array = np.asarray(
            func(*(field.values for field in fields), **func_kwargs), dtype=dtype
        )
    else:

        def apply(*blocks: np.ndarray) -> np.ndarray:
            return np.asarray(func(*blocks, **func_kwargs), dtype=dtype)

        overlap = tuple(depth if dim in SPATIAL_DIMENSIONS else 0 for dim in dims)
        values = da.map_overlap(
            apply,
            *(da.asarray(field.data) for field in fields),
            depth=overlap,
            boundary=boundary,
            allow_rechunk=False,
            dtype=np.dtype(dtype),
            meta=np.array((), dtype=dtype),
        )

    # xarray names a DataArray after its dask array's key unless told otherwise.
    return xr.DataArray(values, dims=reference.dims, coords=reference.coords).rename(
        None
    )

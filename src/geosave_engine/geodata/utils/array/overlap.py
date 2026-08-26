"""map_overlap: run a neighborhood function per dask chunk without seams. See map_overlap."""
from __future__ import annotations

from typing import Any, Callable

import dask.array as darr
import numpy as np
import xarray as xr

# How a chunk's halo is filled at the array's own edge, where no neighbor exists.
Boundary = str | int | float


def map_overlap(
    func: Callable[..., np.ndarray],
    *arrays: xr.DataArray,
    depth: int,
    dtype: str | np.dtype[Any],
    boundary: Boundary = "reflect",
    **kwargs: Any,
) -> xr.DataArray:
    """Apply a function that reads neighboring pixels, chunk by chunk.

    Applied per chunk without a halo, a neighborhood function is wrong at
    every chunk edge. This grows each chunk by `depth`, applies `func`, then
    trims, so the chunked result equals the whole-array one.

    Args:
        func: Takes one NumPy block per array, in the order given, and
            returns one block of the same `(y, x)` shape.
        *arrays: Inputs on one grid, at least one. The first names the
            result's dims and coords.
        depth: Pixels of halo each side, at least the function's own filter
            radius. Too small silently seams; too large only costs time.
        dtype: Result dtype.
        boundary: How the halo is filled at the array's own edge —
            "reflect", "none", or a constant.
        **kwargs: Passed to `func` on every block.

    Returns:
        DataArray on the first input's dims and coords, lazy when the
        inputs are.

    Raises:
        ValueError: No array was given, or `depth` is negative.

    Examples:
        >>> mask = map_overlap(binary_opening, cloud, depth=1, dtype="uint8")
    """
    if not arrays:
        raise ValueError("map_overlap() needs at least one array")
    if depth < 0:
        raise ValueError(f"depth must not be negative, got {depth}")

    reference = arrays[0]
    blocks = [array.data for array in arrays]
    if not any(isinstance(block, darr.Array) for block in blocks):
        values = func(*(np.asarray(block) for block in blocks), **kwargs)
        return xr.DataArray(values.astype(dtype), dims=reference.dims, coords=reference.coords)

    def apply(*chunks: np.ndarray) -> np.ndarray:
        return np.asarray(func(*chunks, **kwargs), dtype=dtype)

    overlapped = darr.map_overlap(
        apply,
        *(darr.asarray(block) for block in blocks),
        depth=depth,
        boundary=boundary,
        dtype=np.dtype(dtype),
        meta=np.array((), dtype=dtype),
    )
    return xr.DataArray(overlapped, dims=reference.dims, coords=reference.coords)

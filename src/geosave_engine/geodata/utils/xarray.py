"""Lazy xarray operations shared by geodata modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr


def map_spatial_overlap(
    func: Callable[..., np.ndarray],
    *fields: xr.DataArray,
    depth: int,
    dtype: str | np.dtype[Any],
    boundary: str | int | float | bool = "reflect",
    **func_kwargs: Any,
) -> xr.DataArray:
    """Map a NumPy function over aligned spatial arrays with chunk overlap.

    NumPy-backed fields run eagerly. Dask-backed fields use overlap on the two
    trailing spatial dimensions and retain the first field's coordinates.

    Args:
        func: Function mapping equally shaped NumPy blocks to one output block.
        *fields: Arrays with identical ordered dimensions and indexes.
        depth: Halo width in pixels on each spatial side.
        dtype: Output dtype.
        boundary: Value or policy used to pad Dask chunk boundaries.
        **func_kwargs: Literal keyword arguments passed to `func`.

    Returns:
        DataArray on the first field's dimensions and coordinates.

    Raises:
        ValueError: Inputs are absent, misaligned, use unsupported dimensions,
            or have spatial chunks smaller than `depth`.
    """
    if not fields:
        raise ValueError("Need at least one input field")
    if depth < 0:
        raise ValueError(f"Halo depth must not be negative, got {depth}")

    reference = fields[0]
    dims = tuple(str(dim) for dim in reference.dims)
    if len(dims) not in (2, 3) or (len(dims) == 3 and dims[0] != "time"):
        raise ValueError(
            f"Field has dimensions {dims}; expected two spatial dimensions, "
            "optionally preceded by 'time'"
        )
    if any(tuple(str(dim) for dim in field.dims) != dims for field in fields[1:]):
        raise ValueError(
            f"Input fields must use ordered dimensions {dims}; transpose them first"
        )
    try:
        fields = xr.align(*fields, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(f"Input fields are not exactly aligned: {exc}") from exc

    dask_fields = [field for field in fields if isinstance(field.data, da.Array)]
    if not dask_fields:
        values: np.ndarray | da.Array = np.asarray(
            func(*(field.values for field in fields), **func_kwargs), dtype=dtype
        )
    else:
        donor = dask_fields[0]
        spatial_dims = dims[-2:]
        for dim in spatial_dims:
            smallest = min(donor.chunksizes[dim])
            if depth > smallest:
                raise ValueError(
                    f"Halo depth {depth} exceeds smallest {dim} chunk {smallest}; "
                    f"rechunk so every {dim} chunk is at least {depth}"
                )
        chunks = {dim: donor.chunksizes[dim] for dim in donor.dims}
        lazy = [field.chunk(chunks) for field in fields]

        def apply(*blocks: np.ndarray) -> np.ndarray:
            return np.asarray(func(*blocks, **func_kwargs), dtype=dtype)

        overlap = tuple(depth if dim in spatial_dims else 0 for dim in dims)
        values = da.map_overlap(
            apply,
            *(field.data for field in lazy),
            depth=overlap,
            boundary=boundary,
            dtype=np.dtype(dtype),
            meta=np.array((), dtype=dtype),
        )

    return xr.DataArray(values, dims=reference.dims, coords=reference.coords)

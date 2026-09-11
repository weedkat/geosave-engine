"""Apply derived-field functions to eager or chunked spatial arrays."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from geosave_engine.geodata.core.array import array


def map_blocks_with_halo(
    func: Callable[..., np.ndarray],
    *fields: xr.DataArray,
    depth: int,
    dtype: str | np.dtype[Any],
    boundary: str | int | float | bool = "reflect",
    **func_kwargs: Any,
) -> xr.DataArray:
    """Apply ``func`` to aligned spatial fields, chunk-wise with a halo.

    NumPy-backed fields run eagerly on one block. Dask-backed fields map over
    chunks with a ``depth``-pixel halo on both spatial dimensions and are
    rechunked to the first Dask field's grid.

    Args:
        func: Maps one block per field to one output block of equal spatial shape.
        *fields: DataArrays sharing dims, shape, and indexes, spanning two
            spatial dimensions optionally preceded by ``time``.
        depth: Non-negative halo width in pixels per spatial side.
        dtype: Output dtype; ``func``'s result is cast to it.
        boundary: Halo fill for the Dask path, passed to ``da.map_overlap``.
        **func_kwargs: Forwarded to ``func`` unchanged.

    Returns:
        DataArray with the first field's dims and coords holding ``func``'s output.

    Raises:
        ValueError: If no fields are given, ``depth`` is negative, the fields do
            not span two trailing spatial dimensions on one aligned grid, or a
            spatial chunk is smaller than ``depth``.
    """
    if not fields:
        raise ValueError("Need at least one input field")
    if depth < 0:
        raise ValueError(f"Halo depth must not be negative, got {depth}")

    reference = fields[0]
    dims = tuple(str(dim) for dim in reference.dims)
    # Spatial axes come last, whatever the CRS names them.
    if len(dims) not in (2, 3) or (len(dims) == 3 and dims[0] != "time"):
        raise ValueError(
            f"Field has dimensions {dims}; expected two spatial dimensions, "
            "optionally preceded by 'time'"
        )
    spatial = dims[-2:]
    try:
        fields = xr.align(*fields, join="exact")
    except ValueError as exc:
        raise ValueError(f"Input fields are not on one aligned grid: {exc}") from exc

    dask_fields = [field for field in fields if isinstance(field.data, da.Array)]
    if not dask_fields:
        values = np.asarray(
            func(*(field.values for field in fields), **func_kwargs), dtype=dtype
        )
        return array(values, reference.odc.geobox, **reference.gs.axes)

    donor = dask_fields[0]
    target_chunks = {dim: donor.chunksizes[dim] for dim in donor.dims}
    for dim in spatial:
        smallest = min(donor.chunksizes[dim])
        if depth > smallest:
            raise ValueError(
                f"Halo depth {depth} exceeds smallest {dim} chunk {smallest}; "
                f"rechunk so every {dim} chunk is at least {depth}"
            )
    lazy = [field.chunk(target_chunks) for field in fields]

    def apply(*chunks: np.ndarray) -> np.ndarray:
        return np.asarray(func(*chunks, **func_kwargs), dtype=dtype)

    spatial_depth = tuple(depth if dim in spatial else 0 for dim in dims)
    overlapped = da.map_overlap(
        apply,
        *(field.data for field in lazy),
        depth=spatial_depth,
        boundary=boundary,
        dtype=np.dtype(dtype),
        meta=np.array((), dtype=dtype),
    )
    return array(overlapped, reference.odc.geobox, **reference.gs.axes)

"""Convert declared raster inputs and row context to model tensors."""

from collections.abc import Mapping
from typing import Any, overload

import numpy as np
import pandas as pd
import torch
import xarray as xr

from geosave_engine.model.spec import ModelSpec


@overload
def to_tensor(
    data: xr.Dataset | xr.DataArray, *, dtype: str | torch.dtype | None = None
) -> torch.Tensor: ...


@overload
def to_tensor(
    data: xr.DataTree, *, dtype: str | torch.dtype | None = None
) -> dict[str, torch.Tensor]: ...


def to_tensor(
    data: xr.Dataset | xr.DataArray | xr.DataTree,
    *,
    dtype: str | torch.dtype | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Convert a native raster or stack into contiguous model tensors.

    Select and order Dataset variables with xarray before conversion. Leading
    axes retain their order, followed by bands and the spatial dimensions.
    Read-only pixel buffers are copied so tensor operations can safely write.

    Args:
        data: Prepared raster, band, or stack of named rasters.
        dtype: Torch dtype or its string name. None preserves the pixel dtype.
            A dtype NumPy cannot hold, such as bfloat16, reads as float32 first.

    Returns:
        Tensor for a Dataset or DataArray, or group names mapped to tensors
        for a DataTree. An individual band retains its existing band axis.

    Raises:
        ValueError: A dtype name is unknown or variables cannot be stacked.
        TypeError: The pixel dtype cannot be converted to a tensor.

    Examples:
        >>> to_tensor(scene[["red", "nir"]], dtype="float32").shape
        torch.Size([2, 256, 256])
    """
    if isinstance(data, xr.DataTree):
        return {
            name: to_tensor(raster, dtype=dtype)
            for name, raster in data.gs.rasters.items()
        }

    if isinstance(dtype, str):
        target_dtype = getattr(torch, dtype, None)
        if not isinstance(target_dtype, torch.dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
    else:
        target_dtype = dtype

    if target_dtype is None:
        values = np.require(data.gs.to_numpy(), requirements=["C", "W"])
        try:
            return torch.as_tensor(values)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot convert numpy dtype {values.dtype} to a torch tensor"
            ) from error

    try:
        reading_dtype = torch.empty(0, dtype=target_dtype).numpy().dtype
    except TypeError:
        reading_dtype = np.dtype("float32")
    return torch.as_tensor(
        np.require(data.gs.to_numpy(dtype=reading_dtype), requirements=["C", "W"]),
        dtype=target_dtype,
    )


def model_inputs(
    spec: ModelSpec,
    rasters: Mapping[str, xr.Dataset | xr.DataArray],
    row: pd.Series,
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare identical model inputs for a training or inference sample.

    Args:
        spec: Raster bindings and optional row-based context recipe.
        rasters: Prepared input rasters, with the sample's window already applied.
        row: Metadata describing the actual selected frames and spatial window.
        context: Explicit cached encoding for this row and recipe.

    Returns:
        Named tensors retaining raster time/band axes.
    """
    values = spec.model_inputs(rasters, row, context=context)
    return {
        name: to_tensor(value)
        if isinstance(value, (xr.Dataset, xr.DataArray))
        else torch.as_tensor(value)
        for name, value in values.items()
    }

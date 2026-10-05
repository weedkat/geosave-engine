"""Convert declared raster inputs and row context to model tensors."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
import torch
import xarray as xr

from geosave_engine.model.spec import ModelSpec


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
        name: value.gs.to_tensor()
        if isinstance(value, (xr.Dataset, xr.DataArray))
        else torch.as_tensor(value)
        for name, value in values.items()
    }

"""Batch native raster samples and reconstruct dense model outputs."""

from collections.abc import Callable, Mapping
from itertools import chain
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import xarray as xr

from geosave_engine.geodata.transform.tiling import Tiles

from .sampling import RasterSamples, sample_windows
from .spec import InferenceSpec


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def infer(
    prepared: xr.DataTree,
    *,
    model: nn.Module,
    settings: InferenceSpec,
    context: Mapping[str, Any] | None = None,
    model_context: Callable[[xr.DataTree], Mapping[str, Any]] | None = None,
    batch_size: int = 1,
    device: str | torch.device | None = None,
) -> tuple[xr.DataArray, ...]:
    """Run dense predictions, returning one merged raster per temporal window.

    Args:
        prepared: Named prepared rasters in a native DataTree.
        model: Loaded module returning a floating tensor shaped B,C,H,W.
        settings: Tensor encoding, temporal windows and spatial tiling.
        context: Additional model arguments shared by every batch.
        model_context: Optional per-tile context extractor, called before reading.
        batch_size: Maximum tiles in one model invocation.
        device: Explicit model device; None uses its current device or CPU.

    Returns:
        Georeferenced merged logits, ordered by temporal window.

    Raises:
        ValueError: Inputs, context, batch size or output shape are incompatible.
        TypeError: The model or its output has an unsupported type.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Inference requires a loaded torch.nn.Module")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    settings = InferenceSpec.model_validate(settings)
    context = dict(context) if context is not None else {}
    if any(not isinstance(key, str) for key in context):
        raise TypeError("Context keys must be strings naming model arguments")
    if collision := context.keys() & settings.inputs.keys():
        raise ValueError(f"Model context arguments collide: {sorted(collision)}")
    windows = sample_windows(prepared, settings=settings)
    first = next(chain(model.parameters(), model.buffers()), None)
    target = (
        torch.device(device)
        if device is not None
        else (first.device if first is not None else torch.device("cpu"))
    )
    if device is not None:
        model.to(target)
    constant_context = _to_device(context, target)
    training = [(module, module.training) for module in model.modules()]
    results = []
    try:
        model.eval()
        with torch.inference_mode():
            for window in windows:
                tiles = Tiles(
                    [window],
                    settings.tiling.tile_shape,
                    overlap=settings.tiling.overlap,
                    mode=settings.tiling.mode,
                )
                samples = RasterSamples(
                    tiles,
                    settings,
                    model_context=model_context,
                    context_keys=frozenset(context),
                )
                merger = tiles.merger(window=settings.tiling.window)
                for batch in DataLoader(samples, batch_size=batch_size, shuffle=False):
                    inputs = _to_device(batch["inputs"], target)
                    derived = _to_device(batch["context"], target)
                    output = model(**inputs, **constant_context, **derived)
                    if (
                        not isinstance(output, torch.Tensor)
                        or not output.is_floating_point()
                    ):
                        raise TypeError(
                            "Model output must be a floating B,C,H,W tensor"
                        )
                    expected = (len(batch["index"]), *settings.tiling.tile_shape)
                    if (
                        output.ndim != 4
                        or output.shape[1] < 1
                        or (output.shape[0], *output.shape[-2:]) != expected
                    ):
                        raise ValueError(
                            f"Model output shape {tuple(output.shape)} must be ({expected[0]}, C, {expected[1]}, {expected[2]})"
                        )
                    output = output.detach().cpu()
                    if output.dtype == torch.bfloat16:
                        output = output.float()
                    merger.add(
                        dict(zip(batch["index"].tolist(), output.numpy(), strict=True))
                    )
                merged = merger.merge()[0].rename("logits")
                merged = merged.assign_coords(band=np.arange(merged.sizes["band"]))
                times = [
                    data.coords["time"].values.reshape(-1)
                    for data in window.gs.rasters.values()
                    if "time" in data.dims
                ]
                if times:
                    labels = np.concatenate(times)
                    merged.attrs.update(
                        time_coverage_start=str(labels.min()),
                        time_coverage_end=str(labels.max()),
                    )
                results.append(merged)
    finally:
        for module, was_training in training:
            module.training = was_training
    return tuple(results)

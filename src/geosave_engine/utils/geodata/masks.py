"""Mask raster utilities: union, single-band write."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr


def union_masks(*masks: np.ndarray) -> np.ndarray:
    """Logical OR over equally-shaped boolean masks → uint8 (0/1)."""
    if not masks:
        raise ValueError("union_masks requires at least one mask")
    combined = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        if m.shape != combined.shape:
            raise ValueError(f"mask shape mismatch: {m.shape} vs {combined.shape}")
        combined |= m.astype(bool)
    return combined.astype(np.uint8)


def write_single_band_mask(
    mask: np.ndarray,
    reference: xr.DataArray,
    path: Path,
    ingestor,
) -> None:
    """Wrap ``mask`` as a DataArray sharing ``reference``'s CRS/transform and save.

    ``ingestor`` is any object exposing ``save(data_array, path)`` (e.g.
    ``BaseIngestor`` subclass). Writes a tiled, compressed single-band GeoTIFF.
    """
    layer = xr.DataArray(
        mask.astype(np.uint8),
        dims=["y", "x"],
        coords={"y": reference.y, "x": reference.x},
    )
    layer = layer.rio.write_crs(reference.rio.crs)
    layer = layer.rio.write_transform(reference.rio.transform())
    ingestor.save(layer, path)

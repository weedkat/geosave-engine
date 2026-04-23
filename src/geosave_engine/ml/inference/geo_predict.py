from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import GeoSlice
from torchgeo.samplers import GridGeoSampler


def _to_hw(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    v = int(value)
    return v, v


class GeoPredictRasterDataset(RasterDataset):
    """TorchGeo raster dataset for patch-based prediction over GeoTIFF COGs."""

    filename_glob = "*.tif"
    filename_regex = r".*"
    is_image = True
    separate_files = False

    def __init__(
        self,
        paths: str | PathLike[str] | list[str | PathLike[str]],
        bands: list[str] | tuple[str, ...] | None = None,
        cache: bool = True,
    ) -> None:
        super().__init__(paths=paths, bands=bands, cache=cache)

    def __getitem__(self, index: GeoSlice) -> dict[str, Any]:
        sample = super().__getitem__(index)

        if hasattr(index, "minx") and hasattr(index, "maxx"):
            minx = float(getattr(index, "minx"))
            maxx = float(getattr(index, "maxx"))
            miny = float(getattr(index, "miny"))
            maxy = float(getattr(index, "maxy"))
        else:
            bounds = sample["bounds"]
            minx = float(bounds[0])
            maxx = float(bounds[1])
            miny = float(bounds[2])
            maxy = float(bounds[3])

        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0
        candidates = self.index.cx[center_x:center_x, center_y:center_y]
        if candidates.empty:
            candidates = self.index.cx[minx:maxx, miny:maxy]

        scene_path = ""
        if not candidates.empty:
            scene_path = str(candidates.iloc[0].filepath)

        sample["scene_path"] = scene_path
        sample["scene_id"] = Path(scene_path).stem if scene_path else "unknown"
        sample["bbox"] = torch.tensor([minx, maxx, miny, maxy], dtype=torch.float64)
        return sample


def build_grid_sampler(
    dataset: GeoPredictRasterDataset,
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> GridGeoSampler:
    """Create a deterministic grid sampler for inference patches."""
    size = _to_hw(patch_size)
    step = _to_hw(stride) if stride is not None else size
    return GridGeoSampler(dataset, size=size, stride=step)

__all__ = ["GeoPredictRasterDataset", "build_grid_sampler"]

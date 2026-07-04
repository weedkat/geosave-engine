from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from geosave_engine.geodata.core import ZarrSource, source_from_dict
from geosave_engine.ml.data import GeoDataModule

from modules.pipeline import (
    CloudMaskPipeline,
    LabelPipeline,
    NdviPipeline,
    Sentinel2Pipeline,
    Sentinel2RGBPipeline,
)

_OUTPUT_KEY: dict[str, str | tuple[str, torch.dtype]] = {
    "sentinel_2_l1c": "image",
    "cloud_mask":     ("mask",  torch.bool),
    "ndvi":           ("ndvi",  torch.float32),
    "dynamicworld":   ("label", torch.int64),
}
_RGB_OUTPUT_KEY: dict[str, str | tuple[str, torch.dtype]] = {
    "sentinel_2_l1c": "image",
    "dynamicworld":   ("label", torch.int64),
}
_RGB_SEL_BANDS: dict[str, list[str]] = {
    "sentinel_2_l1c": ["B04", "B03", "B02"],
}
_ALL_CONTEXT_FIELDS: list[str] = [
    "crs", "transform", "coordinate", "time", "datetime", "bbox_wgs84", "stac_item_ids",
]


class GeosaveDataModule(GeoDataModule):
    """Semantic-segmentation datamodule for Sentinel-2 / DynamicWorld.

    Ingestion runs in ``prepare_data`` only when ``ingest=True``.
    Each split is specified as a source dict under ``sources``.

    Args:
        root: Base directory. Split subdirs created inside.
        sources: Map of split name → source config dict.
            Example: ``{"train": {"type": "geotiff", "src": "data/raw/train/"}, ...}``.
        rgb: RGB-only mode. Uses Sentinel2RGBPipeline (3 bands, faster ingest).
            Skips cloud mask and NDVI. Use ``in_channels: 3`` in model config.
        context_fields: GeoTile metadata fields per sample. Defaults to all fields.
            Valid: ``crs``, ``transform``, ``coordinate``, ``time``,
            ``datetime``, ``bbox_wgs84``, ``stac_item_ids``.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.
        predict_sampler: Sampler strategy for predict stage.
        patch_size: Spatial patch size in pixels (grid sampler only).
        stride: Stride between patches. Defaults to ``patch_size``.
        ingest: Run ingestion in ``prepare_data`` when ``True``.
        max_tiles: Stop ingestion after this many tiles. ``None`` processes all.
    """

    def __init__(
        self,
        root: str | Path,
        sources: dict[str, dict] | None = None,
        rgb: bool = False,
        context_fields: list[str] | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        predict_sampler: Literal["prechipped", "grid"] = "prechipped",
        patch_size: int = 1024,
        stride: int | None = None,
        ingest: bool = False,
        max_tiles: int | None = None,
    ) -> None:
        super().__init__(
            root=root,
            output_key=_RGB_OUTPUT_KEY if rgb else _OUTPUT_KEY,
            sources=sources,
            sel_bands=_RGB_SEL_BANDS if rgb else None,
            context_fields=context_fields if context_fields is not None else _ALL_CONTEXT_FIELDS,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            predict_sampler=predict_sampler,
            patch_size=patch_size,
            stride=stride,
            ingest=ingest,
            max_tiles=max_tiles,
        )
        self.rgb = rgb

    def prepare_data(self) -> None:
        if not self.ingest:
            return
        for split, src_dict in self.sources.items():
            source = source_from_dict(src_dict)
            split_root = self.root / split
            if self.rgb:
                Sentinel2RGBPipeline(split_root).ingest_from(source, max_item=self.max_tiles)
            else:
                zarr_src = ZarrSource(src=split_root / Sentinel2Pipeline.layer_name)
                Sentinel2Pipeline(split_root).ingest_from(source, max_item=self.max_tiles)
                CloudMaskPipeline(split_root).ingest_from(zarr_src, max_item=self.max_tiles)
                NdviPipeline(split_root).ingest_from(zarr_src, max_item=self.max_tiles)
            if split != "predict":
                LabelPipeline(split_root).ingest_from(source, max_item=self.max_tiles)

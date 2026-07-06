from __future__ import annotations

from pathlib import Path
from typing import Literal

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from geosave_engine.geodata.core import GeoTile
from geosave_engine.geodata.datasets import GeoDataset, GridSampler, PreChippedSampler, stack_samples


class HLSS30Dataset(GeoDataset):
    """GeoDataset for HLS S30 prediction tiles."""

    output_key = {"hls_s30": "image"}

    def context(self, tiles: dict[str, GeoTile]) -> dict:
        """Spatial metadata passed through to predict_step output.

        Returns:
            {
                "crs": str,
                "transform": Affine,
                "coordinate": tuple[float, float],
            }
        """
        ref_tile = next(iter(tiles.values()))
        return {
            "crs": ref_tile.crs,
            "transform": ref_tile.affine,
            "coordinate": ref_tile.centroid,
        }


class GraniteBiomassDataModule(LightningDataModule):
    """Prediction-only datamodule for GraniteGeospatialBiomass.

    Reads already-ingested HLS S30 tiles and serves them for sliding-window
    prediction. No train/val/test splits. Ingestion (pre-downloaded GeoTIFFs ->
    ``root/predict/hls_s30``) runs separately via ``geosave ingest -c
    configs/ingest.yaml`` — this module only loads and predicts.

    Args:
        root: Base directory; reads from ``root/predict/hls_s30``.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.
        predict_sampler: ``"prechipped"`` for pre-cut tiles; ``"grid"`` for on-the-fly patches.
        patch_size: Spatial patch size in pixels (grid sampler only).
        stride: Stride between patches. Defaults to ``patch_size``.
    """

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        predict_sampler: Literal["prechipped", "grid"] = "prechipped",
        patch_size: int = 224,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.predict_sampler_type = predict_sampler
        self.patch_size = patch_size
        self.stride = stride or patch_size

    def setup(self, stage: str | None = None) -> None:
        if stage != "predict":
            raise ValueError(
                f"GraniteBiomassDataModule is prediction-only; got stage={stage!r}"
            )
        sampler = (
            GridSampler(self.patch_size, self.stride)
            if self.predict_sampler_type == "grid"
            else PreChippedSampler()
        )
        self.predict_dataset = HLSS30Dataset(self.root / "predict", sampler=sampler)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=stack_samples,
        )

from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from geosave_engine.geodata.datasets import GeoDataset, stack_samples

from .data_pipeline import HLSS30Pipeline, predict_pipeline


class GraniteBiomassDataModule(LightningDataModule):
    """Prediction-only datamodule for GraniteGeospatialBiomass.

    Ingests HLS S30 tiles from pre-downloaded GeoTIFFs and serves them
    for sliding-window prediction. No train/val/test splits.

    Args:
        root: Base directory. ``predict/`` subdir created inside.
        geotiff_src: Path to directory of 6-band HLS S30 GeoTIFFs. Filename
            stems must end in ``-YYYYMMDD`` or ``-YYYYMMDD-YYYYMMDD``.
            Required when ``ingest=True``.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.
        ingest: Run ingestion in ``prepare_data`` when ``True``.
        max_tiles: Stop ingestion after this many tiles. None processes all.
    """

    def __init__(
        self,
        root: str | Path,
        geotiff_src: str | Path | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        ingest: bool = False,
        max_tiles: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.geotiff_src = Path(geotiff_src) if geotiff_src else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.ingest = ingest
        self.max_tiles = max_tiles
        self.pipeline = HLSS30Pipeline()

    def prepare_data(self) -> None:
        if not self.ingest:
            return
        if self.geotiff_src is None:
            raise ValueError("geotiff_src must be set when ingest=True")
        predict_pipeline(self.root / "predict", self.geotiff_src, max_item=self.max_tiles)

    def setup(self, stage: str | None = None) -> None:
        if stage != "predict":
            raise ValueError(
                f"GraniteBiomassDataModule is prediction-only; got stage={stage!r}"
            )
        self.predict_dataset = GeoDataset(self.root / "predict", context_fn=self.pipeline.context)

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

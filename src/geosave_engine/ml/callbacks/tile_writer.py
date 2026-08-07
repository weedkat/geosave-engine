from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from geosave_engine.geodata.tile import GeoStack, GeoTile


class TilePredictionWriter(BasePredictionWriter):
    """
    """

    def __init__(
        self,
        root: str | Path,
        model_name: str,
        resolution: float,
        layers: Sequence[str] | None = None,
    ) -> None: ...

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve `<root>/<model_name>/`, reset this run's open-mosaic cache."""
        ...

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None: ...

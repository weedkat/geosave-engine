from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from geosave_engine.geodata.tile import GeoStack, GeoTile
from geosave_engine.utils.colorize import Palette

_RECOGNIZED_KEYS = ("logits", "proba", "pred")


class DensePredictionWriter(BasePredictionWriter):
    """
    """

    def __init__(
        self,
        root: str | Path,
        model_name: str,
        image_key: str,
        resolution: float,
        class_map: dict[int, str] | None = None,
        color_map: Palette | None = None,
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
    ) -> None:
        """Raises:
            KeyError: None of _RECOGNIZED_KEYS present as a Tensor in prediction.
        """
        ...

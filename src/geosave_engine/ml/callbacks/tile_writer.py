from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from geosave_engine.geodata.spatial import GeoStack, GeoTile
from geosave_engine.utils.colorize import Palette

WriteMode = Literal["geosave", "xcube", "cog"]
_RECOGNIZED_KEYS = ("logits", "proba", "pred")


class TilePredictionWriter(BasePredictionWriter):
    """Persist one GeoStack per predicted anchor, layer by layer, across batches.

    Always persists `batch["tiles"]` (the model's own input layers). If
    `prediction` also carries any of `_RECOGNIZED_KEYS` as a Tensor, each
    gets wrapped into its own GeoTile layer, anchored via `image_key` —
    that's the only difference between "just log inputs" and "log a dense
    prediction" runs, no separate writer needed for either.

    Args:
        root: Output root — layers land under `<root>/<model_name>/<job_id>/`.
        model_name: Subdirectory name for this model's runs.
        resolution: Pixel size in meters for any prediction-derived layer's geobox.
        mode: Output format — `"geosave"` (native multi-group `GeoStack.to_zarr`),
            `"xcube"` (flat `GeoStack.to_xcube`), or `"cog"` (per-layer `GeoTile.to_cog`).
        image_key: `batch["tiles"]` key whose geobox/geotag anchors
            prediction-derived layers. Required only if `prediction` is ever used.
        layers: `batch["tiles"]` keys to persist; None keeps all.
        class_map: Baked into a prediction-derived layer's `plot_meta`.
        color_map: Baked into a prediction-derived layer's `plot_meta`.
        job_id: Run identifier under `model_name`. None derives one at `setup()`.

    Raises:
        ValueError: `mode` isn't one of the recognized values.
    """

    def __init__(
        self,
        root: str | Path,
        model_name: str,
        resolution: float,
        mode: WriteMode = "geosave",
        image_key: str | None = None,
        layers: Sequence[str] | None = None,
        class_map: dict[int, str] | None = None,
        color_map: Palette | None = None,
        job_id: str | None = None,
    ) -> None: ...

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve `<root>/<model_name>/<job_id>/`, reset this run's already-written cache."""
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
        """Gather this batch's layers, group by anchor stem, persist per stem.

        Raises:
            ValueError: `prediction` has a recognized key but `image_key` is unset.
        """
        ...

    def _predicted_layers(self, prediction: Any, anchor: GeoTile) -> dict[str, GeoTile]:
        """Wrap prediction's recognized tensor keys into GeoTiles anchored on `anchor`.

        Args:
            prediction: Model output. {} if none of `_RECOGNIZED_KEYS` present as a Tensor.
            anchor: Tile supplying the geobox/geotag for each wrapped layer.

        Returns:
            Recognized key to GeoTile, `class_map`/`color_map` baked into `plot_meta`.
        """
        raise NotImplementedError

    def _persist(self, stem: str, stack: GeoStack) -> None:
        """Write stack's new layers for one anchor per `self.mode`, skipping ones already on disk."""
        ...

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write `<root>/<model_name>/<job_id>/metadata.json` (model_name, job_id, mode, resolution)."""
        ...

from __future__ import annotations

import json
import logging
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.geotile import GeoTile, mosaic

log = logging.getLogger(__name__)


def _tile_from_context(context: dict[str, Any], i: int, shape: tuple[int, int]) -> GeoTile:
    """Reconstruct a header GeoTile for sample i from stacked batch context.

    Args:
        context: ``batch["context"]`` dict; non-tensor values are lists of length B.
        i: Sample index within the batch.
        shape: ``(H, W)`` spatial shape of the prediction tensor.

    Raises:
        KeyError: If ``crs``, ``transform``, or ``datetime`` are missing from context.
    """
    geobox = GeoBox(shape=shape, affine=context["transform"][i], crs=context["crs"][i])
    return GeoTile(geobox=geobox, datetime=dt.fromisoformat(context["datetime"][i]))


class MosaicBuilder:
    """Merge per-tile prediction COGs into a single mosaic COG.

    Loads all ``pred_label.tif`` and ``pred_proba.tif`` under ``tiles_dir``,
    merges via :func:`geosave_engine.geodata.core.geotile.mosaic`, then writes
    the result to ``mosaic_dir``.

    Args:
        tiles_dir: Directory containing per-tile subdirectories.
        mosaic_dir: Output directory for mosaic files.
    """

    def __init__(self, tiles_dir: Path, mosaic_dir: Path) -> None:
        self.tiles_dir = tiles_dir
        self.mosaic_dir = mosaic_dir

    def build(self) -> None:
        """Merge all prediction tiles into mosaic COGs."""
        for layer in ("pred_label.tif", "pred_proba.tif"):
            paths = sorted(self.tiles_dir.rglob(layer))
            if not paths:
                log.warning("No tiles found for mosaic layer %s", layer)
                continue
            tiles = [GeoTile.from_geotiff(p, load_data=True) for p in paths]
            out_path = self.mosaic_dir / layer
            mosaic(tiles).to_cog(out_path)
            log.info("Mosaic written: %s", out_path)


class PredictionWriter(BasePredictionWriter):
    """Write per-tile predictions and input layers to a structured output directory.

    Writes ``pred_label.tif``, ``pred_proba.tif``, optional input layer COGs,
    and a ``context.json`` sidecar per tile. Writes ``manifest.json`` at the
    end of predict. Optionally merges all tiles into a spatial mosaic.

    Requires ``batch["context"]`` to contain: ``crs``, ``transform``, ``datetime``.
    Optional provenance keys: ``bbox_wgs84``, ``stac_item_ids``.

    Output layout::

        <output_dir>/
          <model_name>/
            tiles/
              pred_<lon>_<lat>_<date>_<res>/
                pred_label.tif
                pred_proba.tif
                inputs/
                  image.tif       # one file per key in input_keys
                context.json
            mosaic/               # only when mosaic=True
              pred_label.tif
              pred_proba.tif
            manifest.json

    Args:
        output_dir: Base directory. Model-named subdirectory created inside.
        input_keys: Batch tensor keys to copy as input layer COGs.
        mosaic: Merge all prediction tiles into mosaic COGs after predict.
        model_name: Model name for the output directory and manifest.
            Falls back to checkpoint stem, then a timestamp.

    Examples:
        # LightningCLI YAML:
        callbacks:
          - class_path: geosave_engine.ml.callbacks.PredictionWriter
            init_args:
              output_dir: predictions/
              input_keys: [image]
              mosaic: true
    """

    def __init__(
        self,
        output_dir: str | Path,
        input_keys: list[str] | None = None,
        mosaic: bool = False,
        model_name: str | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.input_keys: list[str] = input_keys or ["image"]
        self.mosaic = mosaic
        self._model_name_override = model_name
        self._tiles_dir: Path | None = None
        self._model_name: str = ""
        self._written_stems: list[str] = []

    def _resolve_model_name(self, trainer: Trainer) -> str:
        if self._model_name_override:
            return self._model_name_override
        if trainer.ckpt_path:
            return Path(trainer.ckpt_path).stem
        return f"predict_{dt.now().strftime('%Y%m%dT%H%M%S')}"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "predict":
            return
        self._model_name = self._resolve_model_name(trainer)
        self._tiles_dir = self.output_dir / self._model_name / "tiles"
        self._tiles_dir.mkdir(parents=True, exist_ok=True)
        self._written_stems = []

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
        if self._tiles_dir is None:
            raise RuntimeError("PredictionWriter.setup() was not called before write_on_batch_end")
        preds, max_probs = prediction  # both [B, H, W]
        context: dict[str, Any] = batch.get("context", {})
        h, w = preds.shape[-2:]
        for i in range(preds.shape[0]):
            stem = self._write_tile(i, preds[i], max_probs[i], batch, context, (h, w), trainer, self._tiles_dir)
            self._written_stems.append(stem)

    def _write_tile(
        self,
        i: int,
        pred: torch.Tensor,
        max_prob: torch.Tensor,
        batch: dict[str, Any],
        context: dict[str, Any],
        shape: tuple[int, int],
        trainer: Trainer,
        tiles_dir: Path,
    ) -> str:
        base = _tile_from_context(context, i, shape)

        lon, lat = base.centroid
        res = int(base.resolution)
        date_part = context["datetime"][i][:10].replace("-", "")
        stem = f"pred_{lon:.6f}_{lat:.6f}_{date_part}_{res}m"

        tile_dir = tiles_dir / stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        base.with_np(pred.cpu().numpy().astype(np.uint8), ["pred_label"]).to_cog(
            tile_dir / "pred_label.tif"
        )
        base.with_np(max_prob.cpu().numpy().astype(np.float32), ["pred_proba"]).to_cog(
            tile_dir / "pred_proba.tif"
        )

        saved_inputs: dict[str, str] = {}
        inputs_dir = tile_dir / "inputs"
        for key in self.input_keys:
            if key not in batch or not isinstance(batch[key], torch.Tensor):
                continue
            arr = batch[key][i].cpu().numpy().astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            bands = [f"{key}_{j}" for j in range(arr.shape[0])]
            base.with_np(arr, bands).to_cog(inputs_dir / f"{key}.tif")
            saved_inputs[key] = f"inputs/{key}.tif"

        stac_ids: list[str] = list(context.get("stac_item_ids", [None] * (i + 1))[i] or [])
        bbox: list[float] = list(context["bbox_wgs84"][i]) if "bbox_wgs84" in context else list(base.wgs84_bbox)

        ctx: dict[str, Any] = {
            "tile_stem": stem,
            "crs": base.crs,
            "datetime": context["datetime"][i],
            "bbox_wgs84": bbox,
            "stac_item_ids": stac_ids,
            "model_name": self._model_name,
            "model_checkpoint": str(trainer.ckpt_path) if trainer.ckpt_path else None,
            "layers": {
                "pred_label": "pred_label.tif",
                "pred_proba": "pred_proba.tif",
                **saved_inputs,
            },
        }
        with open(tile_dir / "context.json", "w") as f:
            json.dump(ctx, f, indent=2)

        log.debug("Tile written: %s", tile_dir)
        return stem

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._tiles_dir is None:
            return

        manifest: dict[str, Any] = {
            "model_name": self._model_name,
            "model_checkpoint": str(trainer.ckpt_path) if trainer.ckpt_path else None,
            "tiles": self._written_stems,
            "mosaic": self.mosaic,
        }
        manifest_path = self._tiles_dir.parent / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Manifest: %s (%d tiles)", manifest_path, len(self._written_stems))

        if self.mosaic:
            mosaic_dir = self._tiles_dir.parent / "mosaic"
            MosaicBuilder(self._tiles_dir, mosaic_dir).build()

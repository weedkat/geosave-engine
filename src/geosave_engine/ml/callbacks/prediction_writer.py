from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import zarr
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from geosave_engine.geodata.tile import GeoStack, GeoTile
from geosave_engine.utils.colorize import Palette

log = logging.getLogger(__name__)

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
_RECOGNIZED_KEYS = ("logits", "proba", "pred")


def _write_layers(tiles_dir: Path, layers: dict[str, GeoTile], write_zarr: bool, write_cog: bool) -> None:
    """Write one anchor's layers as zarr/cog, skipping ones already present.

    Args:
        tiles_dir: Run's tiles/ directory.
        layers: Layer name to GeoTile, one anchor's worth — every value shares the same stem.
        write_zarr, write_cog: Which format(s) to write.
    """
    stem = next(iter(layers.values())).stem
    stack_path = tiles_dir / f"{stem}.zarr"
    cog_dir = tiles_dir / stem
    existing = set(zarr.open_group(stack_path, mode="r").group_keys()) if stack_path.exists() else set()
    if (not write_zarr or set(layers) <= existing) and (not write_cog or all((cog_dir / k).exists() for k in layers)):
        log.info("Skipping %s — already written", stem)
        return
    if write_zarr:
        GeoStack(**layers).save(stack_path, mode="append")
    if write_cog:
        timestamp = next(iter(layers.values())).start.strftime(_TIMESTAMP_FORMAT)
        for key, tile in layers.items():
            tile.to_cog(cog_dir / key / f"{timestamp}.tif")


def _write_run_metadata(run_dir: Path, model_name: str, job_id: str, checkpoint: str | None) -> None:
    """Write/refresh this run's metadata.json.

    Both TilePredictionWriter and DensePredictionWriter call this — same tiny
    shape either way, harmless if both run in the same predict call.
    """
    metadata = {
        "model_name": model_name,
        "job_id": job_id,
        "checkpoint": checkpoint,
        "created_at": datetime.now().isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


class TilePredictionWriter(BasePredictionWriter):
    """Persist real GeoTiles straight from batch["tiles"], one <stem> per anchor.

    Writes whatever's in batch["tiles"] (or a filtered subset via `layers`) as-is —
    already real, already-named GeoTiles, nothing to convert. Adds each tile's own
    `rgb_subset()` alongside automatically when derivable. Entirely decoupled from
    predict_step's output — this is about the model's *input*, not its output.
    Composable with `DensePredictionWriter` (or any other writer) via
    `GeoStack.save(mode="append")`, so pointing both at the same
    output_dir/model_name/job_id is safe regardless of callback order.

    Output layout::

        <output_dir>/<model_name>/<job_id>/tiles/
          <stem>.zarr              # write_zarr
          <stem>/<layer>/<ts>.tif  # write_cog
        metadata.json

    Re-running predict against the same run directory skips layers already
    written. Never auto-attached — opt in via `trainer.callbacks:`. `model_name`
    left unset is back-filled by `GeosaveCLI` (see
    `GeosaveCLI._fill_prediction_writer_model_name`).

    Args:
        output_dir: Base directory. `<model_name>/<job_id>` created inside.
        model_name: Same `model_name:` top-level config key training uses.
        job_id: This predict invocation's id — caller's own job/request id, or a timestamp.
        write_zarr: Write as one zarr store, one group per layer (ml-preferred).
        write_cog: Also write as COG, one per timestep (serving-preferred).
        layers: batch["tiles"] keys to persist. None persists every key present.

    Raises:
        ValueError: Both write_zarr and write_cog are False.
    """

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        job_id: str | None = None,
        write_zarr: bool = True,
        write_cog: bool = False,
        layers: Sequence[str] | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        if not write_zarr and not write_cog:
            raise ValueError(f"{type(self).__name__} needs write_zarr or write_cog True — got neither")
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.write_zarr = write_zarr
        self.write_cog = write_cog
        self.layers = layers
        self._job_id_override = job_id
        self.job_id: str
        self._tiles_dir: Path
        self._run_dir: Path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve job_id, create `<output_dir>/<model_name>/<job_id>/tiles/`."""
        super().setup(trainer, pl_module, stage)
        self.job_id = self._job_id_override or datetime.now().strftime(_TIMESTAMP_FORMAT)
        self._run_dir = self.output_dir / self.model_name / self.job_id
        self._tiles_dir = self._run_dir / "tiles"
        self._tiles_dir.mkdir(parents=True, exist_ok=True)

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
        tiles_by_layer: dict[str, list[GeoTile]] = batch["tiles"]
        wanted = self.layers if self.layers is not None else list(tiles_by_layer)
        n = len(next(iter(tiles_by_layer.values())))
        for i in range(n):
            layers: dict[str, GeoTile] = {}
            for key in wanted:
                tile = tiles_by_layer[key][i]
                layers[key] = tile
                rgb = tile.rgb_subset()
                if rgb is not None:
                    layers[f"{key}_rgb"] = rgb
            _write_layers(self._tiles_dir, layers, self.write_zarr, self.write_cog)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write `metadata.json`: model_name, job_id, checkpoint, created_at."""
        checkpoint = str(trainer.ckpt_path) if trainer.ckpt_path else None
        _write_run_metadata(self._run_dir, self.model_name, self.job_id, checkpoint)


class DensePredictionWriter(BasePredictionWriter):
    """Persist predict_step's dense per-pixel output as zarr and/or COG.

    Recognizes predict_step output keys "logits"/"proba"/"pred" — whichever are
    present, same small fixed vocabulary `ThresholdCalibrator`/`DensePredictionLogger`
    already read off validation/test outputs. Covers pixelwise regression and
    semantic segmentation identically; anything else predict_step returns is
    ignored (a task can freely return extra keys for other callbacks). Spatial
    identity comes from `batch["tiles"][image_key]`, not predict_step's own output.
    Composable with `TilePredictionWriter` via `GeoStack.save(mode="append")`.

    Each output tile is `rebase()`d off its reference anchor before being built, so
    it carries this run's provenance (model_name/job_id/checkpoint/created_at, in
    `geotag.metadata`) and, if given, class_map/color_map (in `geotag.plot_meta` —
    the tile's own visualization hints, no separate lookup needed to plot it).

    Output layout: same as `TilePredictionWriter` — the two are meant to point at
    the same output_dir/model_name/job_id when used together.

    Args:
        output_dir: Base directory. `<model_name>/<job_id>` created inside.
        model_name: Same `model_name:` top-level config key training uses.
        image_key: `batch["tiles"]` key giving each sample's reference anchor.
        job_id: This predict invocation's id — caller's own job/request id, or a timestamp.
        write_zarr: Write as one zarr store, one group per layer (ml-preferred).
        write_cog: Also write as COG, one per timestep (serving-preferred).
        class_map: `{pixel value: class name}`, baked into each output tile's
            `plot_meta.class_map`. None leaves it unset.
        color_map: `{pixel value: hex/RGB}`, baked into each output tile's
            `plot_meta.color_map`. None leaves it unset.

    Raises:
        ValueError: Both write_zarr and write_cog are False.
    """

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        image_key: str,
        job_id: str | None = None,
        write_zarr: bool = True,
        write_cog: bool = False,
        class_map: dict[int, str] | None = None,
        color_map: Palette | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        if not write_zarr and not write_cog:
            raise ValueError(f"{type(self).__name__} needs write_zarr or write_cog True — got neither")
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.image_key = image_key
        self.write_zarr = write_zarr
        self.write_cog = write_cog
        self.class_map = class_map
        self.color_map = color_map
        self._job_id_override = job_id
        self.job_id: str
        self._tiles_dir: Path
        self._run_dir: Path
        self._plot_meta_update: dict[str, Any]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve job_id, create `<output_dir>/<model_name>/<job_id>/tiles/`."""
        super().setup(trainer, pl_module, stage)
        self.job_id = self._job_id_override or datetime.now().strftime(_TIMESTAMP_FORMAT)
        self._run_dir = self.output_dir / self.model_name / self.job_id
        self._tiles_dir = self._run_dir / "tiles"
        self._tiles_dir.mkdir(parents=True, exist_ok=True)
        self._plot_meta_update = {
            key: value
            for key, value in {"class_map": self.class_map, "color_map": self.color_map}.items()
            if value is not None
        }

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
        found = {
            key: value for key, value in prediction.items() if key in _RECOGNIZED_KEYS and isinstance(value, torch.Tensor)
        }
        if not found:
            raise KeyError(
                f"DensePredictionWriter needs at least one of {_RECOGNIZED_KEYS} in predict_step's "
                f"output as a Tensor, got keys {list(prediction)}"
            )

        checkpoint = str(trainer.ckpt_path) if trainer.ckpt_path else None
        metadata_update = {
            "model_name": self.model_name,
            "job_id": self.job_id,
            "checkpoint": checkpoint,
            "created_at": datetime.now().isoformat(),
        }

        anchors: list[GeoTile] = batch["tiles"][self.image_key]
        for i, anchor in enumerate(anchors):
            tagged = anchor.rebase(metadata=metadata_update, plot_meta=self._plot_meta_update or None)
            layers = {key: tagged.to_geotile(tensor[i].detach().cpu().numpy()) for key, tensor in found.items()}
            _write_layers(self._tiles_dir, layers, self.write_zarr, self.write_cog)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write `metadata.json`: model_name, job_id, checkpoint, created_at."""
        checkpoint = str(trainer.ckpt_path) if trainer.ckpt_path else None
        _write_run_metadata(self._run_dir, self.model_name, self.job_id, checkpoint)

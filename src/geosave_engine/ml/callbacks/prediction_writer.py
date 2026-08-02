from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from geosave_engine.geodata.tile import GEOSTACK_SUFFIX, GeoAnchor, GeoStack, GeoTile

log = logging.getLogger(__name__)

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
_METADATA_ATTRS = ("class_map", "color_map", "band_map", "ignore_index")


class PredictionWriter(BasePredictionWriter):
    """Write per-anchor predictions as GeoStacks, one layer per `predict_step` output key.

    `predict_step` (any task) must return `dict[str, Tensor]` plus one
    `"anchors"` key (`list[GeoAnchor]`, one per sample) — every other key's
    tensor `[B, H, W]` or `[B, C, H, W]`, keyed by whatever layer names that
    task wants saved (e.g. `pred_label`/`pred_proba` for segmentation, a
    single `prediction` for regression). Spatial identity comes from
    `prediction["anchors"]`, not `batch` — same reasoning as
    `ThresholdCalibrator` reading `outputs['logits']`/`['label']` instead of
    reaching into `batch`: this callback has no business knowing a task's
    `image_key`/which batch layer counts as the anchor source. That's the
    task's own call (usually `batch["anchors"][self.image_key]`) — it
    already knows both; this callback just requires the result show up in
    its output, fails fast with a clear message if it doesn't.

    One predicted anchor becomes one `<stem>.geostack/` folder holding
    every returned layer as zarr, plus a `.tif` COG per layer (one COG per
    timestep, for any layer with a time dimension — a COG can't hold one
    itself). Building a combined tiled layer (mosaic, MosaicJSON, whatever
    a map client wants) is a serving-side concern that reads this output —
    not this callback's job.

    Re-running predict against the same run directory skips anchors whose
    `.geostack` folder already exists — the folder itself is the only
    state, no separate tracking file.

    Output layout::

        <output_dir>/
          <model_name>/
            <predict_id>/             # one directory per predict invocation
              tiles/
                <stem>.geostack/
                  <layer_name>.zarr   # every key predict_step returned
                  <layer_name>/
                    <timestamp>.tif   # one COG per timestep (usually just one)
              metadata.json

    This callback is never auto-attached — deployment-specific args
    (`output_dir` especially) have no sensible universal default, so it's
    opt-in via `trainer.callbacks:` in config, same as any other callback.
    `model_name` is the one exception worth knowing about: leave it out of
    your config entirely and `GeosaveCLI` back-fills it from the top-level
    `model_name:` key (see `GeosaveCLI._fill_prediction_writer_model_name`)
    — one source of truth, not a second value to keep in sync by hand. It's
    still a required constructor arg (fails fast, no silent default) for
    anyone wiring this callback outside `GeosaveCLI` (a plain script, a
    different CLI) where that back-fill never runs.

    Args:
        output_dir: Base directory. `<model_name>/<predict_id>` created
            inside — tenancy prefixes above that (user/project id, if any)
            are the caller's own responsibility to build into this path,
            not something this callback knows about.
        model_name: The caller's own name for this model — matches the
            same `model_name:` top-level key training uses (this library
            provides the framework/pipeline, not model identity — never
            guessed from a checkpoint path). Required; `GeosaveCLI`
            back-fills it automatically when this callback is declared in
            config without one, so in practice you rarely type it here.
        input_keys: Batch layer names to also persist alongside the
            predictions (e.g. the source imagery, for QA/traceability).
            Off by default — imagery is usually the bulk of the bytes, and
            it already exists wherever the predict-root `GeoStackDataset` reads
            it from, so duplicating it is opt-in, not automatic.
        predict_id: Override — identifies this predict invocation, distinct
            from `model_name` itself (this is one *inference run* against
            an already-trained model, not the model's own identity). The
            caller's own job/request id if it has one (opaque string, no
            format imposed — ownership of that id belongs to the caller,
            not this library). Otherwise a timestamp — deliberately not
            derived from whatever logger happens to be attached (fragile:
            depends on logging config that has nothing to do with where
            predictions get written, and silently changes behavior if a
            user swaps loggers or disables logging).
    """

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        input_keys: Sequence[str] | None = None,
        # override
        predict_id: str | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.input_keys = input_keys
        self._predict_id_override = predict_id
        self.predict_id: str
        self._tiles_dir: Path
        self._run_dir: Path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve predict_id, create `<output_dir>/<model_name>/<predict_id>/tiles/`."""
        super().setup(trainer, pl_module, stage)

        self.predict_id = self._predict_id_override or datetime.now().strftime(_TIMESTAMP_FORMAT)
        self._run_dir = self.output_dir / self.model_name / self.predict_id
        self._tiles_dir = self._run_dir / "tiles"
        self._tiles_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_tile(anchor: GeoAnchor, array: torch.Tensor) -> GeoTile:
        """Build one GeoTile from one sample's `[H, W]` or `[C, H, W]` tensor.

        Band names aren't knowable here (`PredictionWriter` doesn't have
        task-level band semantics) — `band_{i}` for a multi-band array;
        a `[H, W]` array needs no band name at all (no 'band' dim).
        """
        np_array = array.detach().cpu().numpy()
        if np_array.ndim == 2:
            return anchor.to_geotile(np_array)
        return anchor.to_geotile(np_array, [f"band_{i}" for i in range(np_array.shape[0])])

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
        """Validate `prediction`, pull anchors from it, write one GeoStack per sample.

        Raises:
            KeyError: `prediction` isn't a dict, or has no `"anchors"` key.
            TypeError: `prediction`'s other keys aren't all `Tensor`.
        """
        if not isinstance(prediction, dict) or "anchors" not in prediction:
            raise KeyError(
                "PredictionWriter requires predict_step to return a dict with an "
                "'anchors' key (list[GeoAnchor], one per sample) — add it to your "
                "task's predict_step output, e.g. output['anchors'] = batch['anchors'][self.image_key]"
            )
        anchors = prediction["anchors"]
        layers_out = {key: value for key, value in prediction.items() if key != "anchors"}
        if not all(isinstance(value, torch.Tensor) for value in layers_out.values()):
            raise TypeError(
                f"PredictionWriter expects predict_step to return dict[str, Tensor] plus "
                f"'anchors', got value types {[type(v).__name__ for v in layers_out.values()]}"
            )

        for i, anchor in enumerate(anchors):
            stem = anchor.stem
            stack_path = self._tiles_dir / f"{stem}{GEOSTACK_SUFFIX}"
            if stack_path.exists():
                log.info("Skipping %s — already predicted", stem)
                continue

            layers = {key: self._to_tile(anchor, tensor[i]) for key, tensor in layers_out.items()}
            for key in self.input_keys or []:
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    layers[key] = self._to_tile(anchor, value[i])

            GeoStack(**layers).save(stack_path)

            timestamp = anchor.start.strftime(_TIMESTAMP_FORMAT)
            for key, tile in layers.items():
                tile.to_cog(stack_path / key / f"{timestamp}.tif")

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write `metadata.json`: model_name, predict_id, checkpoint, created_at,
        plus class_map/color_map/band_map/ignore_index read off `pl_module`
        when present (duck-typed — absent for tasks that don't have them,
        e.g. regression)."""
        metadata: dict[str, Any] = {
            "model_name": self.model_name,
            "predict_id": self.predict_id,
            "checkpoint": str(trainer.ckpt_path) if trainer.ckpt_path else None,
            "created_at": datetime.now().isoformat(),
        }
        for attr in _METADATA_ATTRS:
            value = getattr(pl_module, attr, None)
            if value is not None:
                metadata[attr] = value

        (self._run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

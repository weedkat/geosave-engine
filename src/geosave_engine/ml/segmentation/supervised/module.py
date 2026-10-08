from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast
from dataclasses import dataclass, field

import numpy as np
from tiler import Merger

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_warn
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave_engine.ml.segmentation.metrics import SemanticSegmentationMetrics
from .data import _read_target
from geosave_engine.ml.builders import (
    build_criterion,
    build_optimizer,
    build_scheduler,
)
from geosave_engine.model.chain import ModelChain
from geosave_engine.model.registry import build_model
from geosave_engine.geodata.warnings import GeoSaveWarning
from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS


@dataclass
class _Evaluation:
    """Per-loader completion and native accumulation state."""

    remaining: set[str]
    mergers: dict[str, Merger] = field(default_factory=dict)
    halos: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


class Module(LightningModule):
    """Train and evaluate a segmentation model chain on tile batches.

    A batch is ``(model_inputs, target, ids, valid)``. Training scores each tile;
    validation and test merge tile logits and score each raster whole, so
    their datasets supply original parents and the window table their ids
    name. Scene evaluation currently requires a single device;
    tile-sharded distributed evaluation cannot complete the local scenes.

    Args:
        model_chain: Ordered stage construction specifications. Each selects a
            registered name or class_path with optional init_args. The last
            stage states the class count as `num_classes`.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Loss class and arguments. Defaults to CrossEntropyLoss with
            ignore_index; an explicit criterion ignore_index takes precedence.
        optimizer: Optimizer class, arguments, and stage groups. Defaults to
            AdamW with a learning rate of 1e-3.
        lr_scheduler: Scheduler class, arguments, and Lightning metadata.
            None disables scheduling.
        metrics: Metric names in dot notation (e.g. ``["iou.macro", "f1.macro"]``).
        class_thresholds: Per-class confidence threshold, one per class the
            last stage states. None initializes every class to 0.5. Checkpoint
            loading restores saved thresholds; an explicitly configured
            calibration callback may update them.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave_engine.ml.segmentation.supervised.Module
          init_args:
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: dpt}
              head:
                name: segmentation
                init_args:
                  classes: [background, oil_palm]
    """

    model: ModelChain
    num_classes: int
    class_thresholds: torch.Tensor

    def __init__(
        self,
        *,
        model_chain: dict[str, dict[str, Any]],
        ignore_index: int = 255,
        criterion: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        class_thresholds: list[float] | None = None,
    ) -> None:
        super().__init__()

        if not model_chain:
            raise ValueError("Supply at least one model stage")
        if criterion is None:
            criterion = {"name": "cross_entropy"}
        if optimizer is None:
            optimizer = {"name": "adamw", "init_args": {"lr": 1e-3}}
        self.save_hyperparameters()
        self.model_chain = model_chain

        self.ignore_index = ignore_index

        self.optimizer_spec = deepcopy(optimizer)
        self.scheduler_spec = deepcopy(lr_scheduler)
        self.metrics_config = metrics
        self._initial_class_thresholds = class_thresholds

        criterion_spec = deepcopy(criterion)
        criterion_spec["init_args"] = {
            "ignore_index": ignore_index,
            **criterion_spec.get("init_args", {}),
        }
        self.criterion = build_criterion(criterion_spec)

    def configure_model(self):
        """Construct the model, then the thresholds and metrics its classes size.

        Raises:
            TypeError: The last stage states no integer `num_classes`.
            ValueError: `class_thresholds` does not hold one entry per class.
        """
        if hasattr(self, "model"):
            return

        last = list(self.model_chain)[-1]
        self.model = build_model(deepcopy(self.model_chain))

        head = self.model.get_submodule(last)
        num_classes = getattr(head, "num_classes", None)
        if not isinstance(num_classes, int):
            raise TypeError(
                f"stage {last!r} ({type(head).__name__}) states no num_classes; "
                f"the last stage names how many classes it predicts"
            )
        self.num_classes = num_classes
        thresholds = self._initial_class_thresholds
        if thresholds is not None and len(thresholds) != num_classes:
            raise ValueError(
                f"class_thresholds must have {num_classes} entries, "
                f"got {len(thresholds)}"
            )
        self.register_buffer(
            "class_thresholds",
            torch.full((num_classes,), 0.5)
            if thresholds is None
            else torch.tensor(thresholds),
        )

        # Lightning runs setup before configure_model, so metrics are built here.
        metrics = SemanticSegmentationMetrics(
            num_classes=num_classes,
            ignore_index=self.ignore_index,
            metrics=self.metrics_config,
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(self.optimizer_spec, self.model)

        if self.scheduler_spec is None:
            return optimizer

        return cast(
            OptimizerLRScheduler,
            {
                "optimizer": optimizer,
                "lr_scheduler": build_scheduler(self.scheduler_spec, optimizer),
            },
        )

    def forward(self, **model_inputs: Any) -> torch.Tensor:
        """Return raw per-pixel logits for prepared model inputs.

        Args:
            **model_inputs: Prepared named model tensors. ``image`` holds the
                selected encoder's input layout, including a time axis when
                required; contextual tensors are ordinary keys.

        Returns:
            Raw per-pixel logits.

        Raises:
            TypeError: If the configured model does not return one tensor.
        """
        logits = self.model(**model_inputs)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "Semantic segmentation models must return logits as a tensor"
            )
        return logits

    def training_step(
        self,
        batch: tuple[dict[str, Any], torch.Tensor, Sequence[str], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        model_inputs, target, _, _ = batch

        logits = self(**model_inputs)
        loss = self.criterion(logits, target)

        self.train_metrics.update(logits, target)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=target.shape[0],
        )
        self.log_dict(
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=target.shape[0],
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        """Start merging afresh, dropping rasters a cut-short run left open."""
        self._start_evaluation()

    def on_test_epoch_start(self) -> None:
        """Start merging afresh, dropping rasters a cut-short run left open."""
        self._start_evaluation()

    def _start_evaluation(self) -> None:
        """Require local scene completion before initializing merge state."""
        if self._trainer is not None and self.trainer.world_size > 1:
            raise ValueError(
                "scene evaluation requires a single device so each rank can "
                "complete scenes; distributed scene partitioning is not supported"
            )
        # Only continuous logits are assembled; categorical targets stay native.
        self._evaluation: dict[int, _Evaluation] = {}
        self._raster_count = 0

    def on_validation_epoch_end(self) -> None:
        """Say so when a run cut short scored nothing, so no `val_loss` exists."""
        if (
            self._evaluation
            and not self._raster_count
            and not self.trainer.sanity_checking
        ):
            rank_zero_warn(
                "validation ended before any raster was whole, so nothing was "
                "scored and val_loss was not logged; a raster is scored once "
                "all its tiles have been read, so let validation read at least "
                "one whole raster (raise limit_val_batches)",
                category=GeoSaveWarning,
            )

    def _merge(
        self,
        logits: torch.Tensor,
        ids: Sequence[str],
        valid: torch.Tensor,
        loader: Any,
        dataloader_idx: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Assemble logits and pair completed parents with their original labels."""
        if isinstance(loader, (list, tuple)):
            loader = loader[dataloader_idx]
        dataset = loader.dataset
        if dataloader_idx not in self._evaluation:
            self._evaluation[dataloader_idx] = _Evaluation(set(dataset.reference.id))
        state = self._evaluation[dataloader_idx]
        rows = dataset.reference.loc[list(ids)]
        pixels = logits.detach().float().cpu().numpy()
        masks = valid.cpu().numpy() & np.isfinite(pixels).all(axis=1)
        for sample_id, (_, row), values, mask in zip(
            ids, rows.iterrows(), pixels, masks, strict=True
        ):
            if sample_id not in state.remaining:
                raise ValueError(f"prediction {sample_id!r} was already accumulated")
            parent_id = row.parent
            if parent_id not in state.mergers:
                # The layout a cut used is rebuilt from the same shape and settings.
                tiler, state.halos[parent_id] = dataset.spec.chips.layout(
                    tuple(dataset.parents[parent_id].gs.anchor.geobox.shape)
                )
                state.mergers[parent_id] = Merger(
                    tiler,
                    logits=self.num_classes + 1,
                    window=dataset.spec.chips.window,
                    save_visits=False,
                    data_dtype=np.float64,
                    weights_dtype=np.float64,
                )
            # Coverage uses the same native taper as logits, excluding invalid tiles.
            contribution = np.concatenate([np.where(mask, values, 0), mask[None]])
            state.mergers[parent_id].add(int(row.chip), contribution)
            state.remaining.remove(sample_id)

        completed = []
        for parent_id in rows.parent.unique():
            expected = dataset.reference.loc[
                dataset.reference.parent == parent_id, "id"
            ]
            if state.remaining.intersection(expected):
                continue
            sums = state.mergers.pop(parent_id).merge(
                extra_padding=state.halos.pop(parent_id),
                normalize_by_weights=False,
            )
            coverage = sums[-1]
            values = np.full(sums[:-1].shape, np.nan, dtype=np.float32)
            np.divide(sums[:-1], coverage, out=values, where=coverage > 0)
            parent = dataset.parents[parent_id]
            image = next(iter(parent.gs.rasters.values()))
            labels = parent[dataset.target].dataset
            if image.gs.geobox != labels.gs.geobox or tuple(
                image.sizes[d] for d in SPATIAL_DIMENSIONS
            ) != tuple(labels.sizes[d] for d in SPATIAL_DIMENSIONS):
                raise ValueError(
                    f"target grid differs from prediction for {parent_id!r}"
                )
            target = _read_target(labels, self.ignore_index).to(self.device)
            raster_logits = torch.from_numpy(values).to(self.device)
            covered = torch.isfinite(raster_logits).all(dim=0)
            if target.shape != covered.shape:
                raise ValueError(
                    f"target shape differs from prediction for {parent_id!r}"
                )
            target = target.masked_fill(~covered, self.ignore_index)
            self._raster_count += 1
            if (target != self.ignore_index).any():
                completed.append(
                    (torch.nan_to_num(raster_logits).unsqueeze(0), target.unsqueeze(0))
                )

        return completed

    def validation_step(
        self,
        batch: tuple[dict[str, Any], torch.Tensor, Sequence[str], torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Merge validation tiles and score each raster once it is whole.

        Args:
            batch: ``(model_inputs, target, ids, valid)`` prepared for the model chain.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning validation loader number.

        Returns:
            Tile logits and labels for validation callbacks.
        """
        model_inputs, target, ids, valid = batch
        logits = self(**model_inputs)

        loaders = self.trainer.val_dataloaders
        for raster_logits, raster_target in self._merge(
            logits, ids, valid, loaders, dataloader_idx
        ):
            self.log(
                "val_loss",
                self.criterion(raster_logits, raster_target),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
            )
            self.val_metrics.update(raster_logits, raster_target)
            self.log_dict(
                self.val_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )
        return {"logits": logits, "label": target}

    def test_step(
        self,
        batch: tuple[dict[str, Any], torch.Tensor, Sequence[str], torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Merge test tiles and score each raster once it is whole.

        Args:
            batch: ``(model_inputs, target, ids, valid)`` prepared for the model chain.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning test loader number.

        Returns:
            Tile logits and labels for test callbacks.
        """
        model_inputs, target, ids, valid = batch
        logits = self(**model_inputs)

        loaders = self.trainer.test_dataloaders
        for raster_logits, raster_target in self._merge(
            logits, ids, valid, loaders, dataloader_idx
        ):
            self.test_metrics.update(raster_logits, raster_target)
            self.log_dict(
                self.test_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )
        return {"logits": logits, "label": target}

    def predict_step(
        self,
        batch: tuple[dict[str, Any], Sequence[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, Sequence[str]]:
        """Return raw tile logits beside their reference IDs.

        Args:
            batch: ``(model_inputs, ids)`` from a method-owned prediction Dataset.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning prediction loader number.

        Returns:
            ``(logits, ids)``. IDs identify rows in the dataset's reference.
        """
        model_inputs, ids = batch
        return self(**model_inputs), ids

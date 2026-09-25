from copy import deepcopy
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.model_chain import ModelChain
from geosave_engine.ml.registry import (
    build_criterion,
    build_model,
    build_optimizer,
    build_scheduler,
)


def softmax_argmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over the class dim, then argmax + top-class confidence.

    Args:
        logits: `[B, num_classes, H, W]` raw model output.

    Returns:
        `(preds [B, H, W] argmax class, max_probs [B, H, W] top-class confidence)`.
    """
    probs = logits.softmax(dim=1)
    max_probs, preds = probs.max(dim=1)
    return preds, max_probs


def apply_thresholds(
    logits: torch.Tensor,
    thresholds: torch.Tensor,
    ignore_index: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax + per-class confidence threshold + optional nodata mask.

    Args:
        logits: `[B, num_classes, H, W]` raw model output.
        thresholds: `[num_classes]` per-class confidence threshold.
        ignore_index: Class index assigned to low-confidence/masked pixels.
        mask: Optional boolean `[B, H, W]` nodata mask. Masked pixels -> ignore_index.

    Returns:
        `(preds [B, H, W], max_probs [B, H, W] float32)`.
    """
    preds, max_probs = softmax_argmax(logits)

    pixel_thresholds = torch.index_select(thresholds, 0, preds.reshape(-1)).view_as(
        preds
    )
    preds = torch.where(
        max_probs >= pixel_thresholds, preds, preds.new_full((), ignore_index)
    )

    if mask is not None:
        preds = torch.where(mask.bool(), preds.new_full((), ignore_index), preds)

    return preds, max_probs


class SemanticSegmentationTask(LightningModule):
    """Train and evaluate segmentation models from prepared image tensors.

    Inputs must already use the dtype, bands, and numerical representation the
    selected model expects. Each supervised batch is ``(model_inputs, target)``
    where model inputs are passed to the model chain by name.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        model_chain: Ordered stage construction specifications. Each selects a
            registered name or class_path with optional init_args. Defaults to
            DINOv3, DPT, and DenseHead. The first stage receives in_channels and
            input_size; the last receives num_classes. These task dimensions
            override stage init_args.
        input_size: Training crop size and default model input size.
        ignore_index: Class index excluded from loss and metrics.
        criterion: Loss class and arguments. Defaults to CrossEntropyLoss with
            ignore_index; an explicit criterion ignore_index takes precedence.
        optimizer: Optimizer class, arguments, and stage groups. Defaults to
            AdamW with a learning rate of 1e-3.
        lr_scheduler: Scheduler class, arguments, and Lightning metadata.
            None disables scheduling.
        metrics: Metric names in dot notation (e.g. ``["iou.macro", "f1.macro"]``).
        class_thresholds: Per-class confidence threshold, one per output class.
            None initializes every class to 0.5. Checkpoint loading restores saved
            thresholds; an explicitly configured calibration callback may update them.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave_engine.ml.tasks.SemanticSegmentationTask
          init_args:
            in_channels: 2
            num_classes: 2
            model_chain:
              encoder: {name: dinov3}
              decoder: {name: dpt}
              head: {name: dense}
    """

    model: ModelChain
    class_thresholds: torch.Tensor

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        model_chain: dict[str, dict[str, Any]] | None = None,
        input_size: int | tuple[int, int] = 224,
        ignore_index: int = 255,
        criterion: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        class_thresholds: list[float] | None = None,
    ) -> None:
        super().__init__()

        if model_chain is None:
            model_chain = {
                "encoder": {"name": "dinov3"},
                "decoder": {"name": "dpt"},
                "head": {"name": "dense"},
            }
        if not model_chain:
            raise ValueError("Supply at least one model stage")
        if criterion is None:
            criterion = {"name": "cross_entropy"}
        if optimizer is None:
            optimizer = {"name": "adamw", "init_args": {"lr": 1e-3}}
        self.save_hyperparameters()
        self.model_chain = model_chain

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        self.ignore_index = ignore_index

        self.optimizer_spec = deepcopy(optimizer)
        self.scheduler_spec = deepcopy(lr_scheduler)
        self.metrics_config = metrics
        if class_thresholds is not None and len(class_thresholds) != self.num_classes:
            raise ValueError(
                f"class_thresholds must have {self.num_classes} entries, "
                f"got {len(class_thresholds)}"
            )
        self._initial_class_thresholds = class_thresholds

        criterion_spec = deepcopy(criterion)
        criterion_spec["init_args"] = {
            "ignore_index": ignore_index,
            **criterion_spec.get("init_args", {}),
        }
        self.criterion = build_criterion(criterion_spec)

    def configure_model(self) -> None:
        """Construct the model and prediction thresholds."""
        if hasattr(self, "model"):
            return

        stage_names = list(self.model_chain)
        first, last = stage_names[0], stage_names[-1]
        stage_specs = deepcopy(self.model_chain)
        stage_specs[first]["init_args"] = {
            **stage_specs[first].get("init_args", {}),
            "in_channels": self.in_channels,
            "input_size": self.input_size,
        }
        stage_specs[last]["init_args"] = {
            **stage_specs[last].get("init_args", {}),
            "num_classes": self.num_classes,
        }
        self.model = build_model(stage_specs)

        initial = (
            torch.tensor(self._initial_class_thresholds)
            if self._initial_class_thresholds is not None
            else torch.full((self.num_classes,), 0.5)
        )
        self.register_buffer("class_thresholds", initial)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(self.optimizer_spec, self.model)

        if self.scheduler_spec is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": build_scheduler(self.scheduler_spec, optimizer),
        }

    def setup(self, stage: str | None = None) -> None:
        metrics = SemanticSegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            metrics=self.metrics_config,
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, **model_inputs: Any) -> torch.Tensor:
        """Run a prepared tile batch through the model chain.

        Args:
            **model_inputs: Prepared named model tensors. ``image`` holds the
                selected encoder's input layout, including a time axis when
                required; contextual tensors are ordinary keys.

        Returns:
            ``[B, num_classes, H, W]`` logits.
        """
        result = self.model(**model_inputs)
        return result if isinstance(result, torch.Tensor) else result["logits"]

    def postprocess(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign classes after tile logits have been stitched into scenes.

        Args:
            logits: ``[B, num_classes, H, W]`` raw model output.
            mask: Optional boolean ``[B, H, W]`` nodata mask. Masked pixels → ignore_index.

        Returns:
            ``(pred_label [B, H, W] uint8, pred_proba [B, H, W] float32)``.
        """
        preds, max_probs = apply_thresholds(
            logits, self.class_thresholds, self.ignore_index, mask
        )

        preds = preds.to(torch.uint8)
        max_probs = max_probs.to(torch.float32)

        return preds, max_probs

    def training_step(
        self, batch: tuple[dict[str, Any], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        model_inputs, target = batch
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

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

    def validation_step(
        self,
        batch: tuple[dict[str, Any], torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute loss and metrics over prepared validation tiles.

        Args:
            batch: ``(model_inputs, target)`` prepared for the model chain.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning validation loader number.

        Returns:
            Tile logits and labels for validation callbacks.
        """
        model_inputs, target = batch
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        logits = self(**model_inputs)
        loss = self.criterion(logits, target)

        self.val_metrics.update(logits, target)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return {"logits": logits, "label": target}

    def test_step(
        self,
        batch: tuple[dict[str, Any], torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute metrics over prepared test tiles.

        Args:
            batch: ``(model_inputs, target)`` prepared for the model chain.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning test loader number.

        Returns:
            Tile logits and labels for test callbacks.
        """
        model_inputs, target = batch
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        logits = self(**model_inputs)

        self.test_metrics.update(logits, target)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False)
        return {"logits": logits, "label": target}

    def predict_step(
        self,
        batch: tuple[dict[str, Any], Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, Any]:
        """Predict tile logits for geodata stitching without class assignment.

        Args:
            batch: ``(model_inputs, index)`` from TileDataset.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning prediction loader number.

        Returns:
            ``(logits, index)``. Merge logits before applying
            postprocess; indices are local to each loader's Tiles collection.
        """
        model_inputs, index = batch
        return self(**model_inputs), index

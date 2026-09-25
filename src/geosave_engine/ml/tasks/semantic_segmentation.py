from __future__ import annotations

import logging
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave_engine.ml.registry import (
    BuildSpec,
    StageSpec,
    build_loss,
    build_optimizer,
    build_scheduler,
)
from geosave_engine.ml.postprocessing.segmentation import apply_thresholds
from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.models.contract import ModelChain
from geosave_engine.ml.transforms import ImageAugmenter

log = logging.getLogger(__name__)


def _validate_dense_map(name: str, mapping: dict[int, str]) -> None:
    """Raise if `mapping`'s keys aren't exactly `0..len(mapping)-1`.

    Args:
        name: Param name, for the error message.
        mapping: Map to check (`class_map` or `band_map`).

    Raises:
        ValueError: Keys aren't a dense 0-based range — a hand-typed gap or
            duplicate would otherwise silently misalign class/channel indices.
    """
    expected = set(range(len(mapping)))
    if set(mapping) != expected:
        raise ValueError(
            f"{name} keys must be dense 0..{len(mapping) - 1}, got {sorted(mapping)}"
        )


class SemanticSegmentationTask(LightningModule):
    """Train and evaluate segmentation models from prepared image tensors.

    Inputs must already use the dtype, bands, and numerical representation the
    selected model expects. Model context arrives in batch["model_context"].

    Args:
        stages: Ordered stage construction specifications. Each selects a
            registered name or class_path with optional init_args. Defaults to
            DINOv3, DPT, and DenseHead. The first stage receives in_channels and
            input_size; the last receives num_classes. Explicit init_args override
            these task defaults, and constructors must accept the resulting arguments.
        input_size: Training crop size and default model input size.
        image_key: Batch key holding the input image tensor.
        label_key: Batch key holding the label tensor.
        ignore_index: Class index excluded from loss and metrics.
        class_map: ``{class_id: class_name}`` for every output class, dense
            from 0. Required — ``num_classes`` is ``len(class_map)``.
        band_map: ``{channel_idx: band_name}`` for every input channel, dense
            from 0. Required — ``in_channels`` is ``len(band_map)``.
        loss: Loss construction specification. None selects CELoss.
        optimizer: Optimizer construction specification. None selects AdamW.
        scheduler: Scheduler construction specification. None disables scheduling.
        metrics: Metric names in dot notation (e.g. ``["iou.macro", "f1.macro"]``).
        augmentations: Kornia augmentation config list.
        class_thresholds: Per-class confidence threshold, one per class_map entry.
            None initializes every class to 0.5. Checkpoint loading restores saved
            thresholds; an explicitly configured calibration callback may update them.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave_engine.ml.tasks.SemanticSegmentationTask
          init_args:
            stages:
              encoder: {name: dinov3}
              decoder: {name: dpt}
              head: {name: dense}
            image_key: sentinel_2_l1c
            label_key: dynamicworld
            class_map: {0: water, 1: trees}
            band_map: {0: B02, 1: B03}
    """

    model: ModelChain
    class_thresholds: torch.Tensor

    def __init__(
        self,
        *,
        stages: dict[str, StageSpec] | None = None,
        class_map: dict[int, str],
        band_map: dict[int, str],
        input_size: int | tuple[int, int] = 224,
        image_key: str = "image",
        label_key: str = "label",
        ignore_index: int = 255,
        loss: BuildSpec | None = None,
        optimizer: BuildSpec | None = None,
        scheduler: BuildSpec | None = None,
        metrics: list[str] | None = None,
        augmentations: list[dict] | None = None,
        class_thresholds: list[float] | None = None,
    ) -> None:
        super().__init__()

        if stages is None:
            stages = {
                "encoder": {"name": "dinov3"},
                "decoder": {"name": "dpt"},
                "head": {"name": "dense"},
            }
        if not stages:
            raise ValueError("Supply at least one model stage")
        loss = {"name": "CELoss"} if loss is None else loss
        optimizer = {"name": "AdamW"} if optimizer is None else optimizer
        self.save_hyperparameters()
        self.stages = stages

        _validate_dense_map("class_map", class_map)
        _validate_dense_map("band_map", band_map)
        self.num_classes = len(class_map)
        self.in_channels = len(band_map)
        self.input_size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        self.image_key = image_key
        self.label_key = label_key
        self.ignore_index = ignore_index
        self.class_map = class_map
        self.band_map = band_map

        self.optimizer_spec = optimizer
        self.scheduler_spec = scheduler
        self.metrics_config = metrics
        self.augmentations = augmentations or []
        if class_thresholds is not None and len(class_thresholds) != self.num_classes:
            raise ValueError(
                f"class_thresholds must have {self.num_classes} entries (one per class_map "
                f"entry), got {len(class_thresholds)}"
            )
        self._initial_class_thresholds = class_thresholds

        self.loss_fn = build_loss(
            {
                **loss,
                "init_args": {
                    "ignore_index": ignore_index,
                    **loss.get("init_args", {}),
                },
            }
        )

    def configure_model(self) -> None:
        """Construct the model, training augmentation, and prediction thresholds."""
        if hasattr(self, "model"):
            return

        stage_names = list(self.stages)
        first, last = stage_names[0], stage_names[-1]
        stages: dict[str, StageSpec] = {
            name: {**spec} for name, spec in self.stages.items()
        }
        stages[first]["init_args"] = {
            "in_channels": self.in_channels,
            "input_size": self.input_size,
            **stages[first].get("init_args", {}),
        }
        stages[last]["init_args"] = {
            "num_classes": self.num_classes,
            **stages[last].get("init_args", {}),
        }
        self.model = ModelChain(stages=stages)

        initial = (
            torch.tensor(self._initial_class_thresholds)
            if self._initial_class_thresholds is not None
            else torch.full((self.num_classes,), 0.5)
        )
        self.register_buffer("class_thresholds", initial)
        self.augmenter = ImageAugmenter(
            augmentations=self.augmentations,
            size=self.input_size,
            data_keys=["image", "mask"],
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(self.optimizer_spec, self.model)

        if self.scheduler_spec is None:
            return optimizer

        scheduler = build_scheduler(self.scheduler_spec, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage: str | None = None) -> None:
        metrics = SemanticSegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            labels=[self.class_map[i] for i in range(self.num_classes)],
            metrics=self.metrics_config,
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, image: torch.Tensor, **ctx: Any) -> torch.Tensor:
        """Run a prepared tile batch through the model chain.

        Args:
            image: Prepared tile tensor in the selected encoder's input layout,
                including a time axis when required.
            **ctx: Extra per-model context (e.g. `temporal_coords=...`,
                `location_coords=...`) forwarded from the dataset adapter to
                the model chain unchanged.
                Only consumed by whichever stage's `@chain_step` method
                actually names the key — unused keys sit in the chain's ctx dict untouched.

        Returns:
            ``[B, num_classes, H, W]`` logits.
        """
        result = self.model(image=image, **ctx)
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

    def _extract_model_context(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Read precomputed model context from a batch.

        Dataset adapters compute this through their context function before
        encoding tensors. Model code only consumes the result.

        Args:
            batch: One model batch containing optional precomputed context.

        Returns:
            Extra keys to forward into `self(image, **model_context)` — `{}`
            when the adapter supplied no context.
        """
        return dict(batch.get("model_context") or {})

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        image, label = batch["layers"][self.image_key], batch["layers"][self.label_key]
        model_context = self._extract_model_context(batch)
        image, label = self.augmenter(image, label)
        label = label.squeeze(1)  # (B, 1, H, W) → (B, H, W)

        logits = self(image, **model_context)
        loss = self.loss_fn(logits, label)

        self.train_metrics.update(logits, label)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=image.shape[0],
        )
        self.log_dict(
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=image.shape[0],
        )

        return loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        """Compute loss and metrics over prepared validation tiles.

        Args:
            batch: Supervised layers and optional per-tile model_context.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning validation loader number.

        Returns:
            Tile logits and labels for validation callbacks.
        """
        image, label = batch["layers"][self.image_key], batch["layers"][self.label_key]
        model_context = self._extract_model_context(batch)
        label = label.squeeze(1)  # (B, 1, H, W) → (B, H, W)

        logits = self(image, **model_context)
        loss = self.loss_fn(logits, label)

        self.val_metrics.update(logits, label)
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
        return {"logits": logits, "label": label}

    def test_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        """Compute metrics over prepared test tiles.

        Args:
            batch: Supervised layers and optional per-tile model_context.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning test loader number.

        Returns:
            Tile logits and labels for test callbacks.
        """
        image, label = batch["layers"][self.image_key], batch["layers"][self.label_key]
        model_context = self._extract_model_context(batch)
        label = label.squeeze(1)  # (B, 1, H, W) → (B, H, W)

        logits = self(image, **model_context)

        self.test_metrics.update(logits, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False)
        return {"logits": logits, "label": label}

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Predict tile logits for geodata stitching without class assignment.

        Args:
            batch: TileDataset batch with image, index, and optional model_context.
            batch_idx: Lightning batch number.
            dataloader_idx: Lightning prediction loader number.

        Returns:
            Raw logits and the input tile indices. Merge logits before applying
            postprocess; indices are local to each loader's Tiles collection.
        """
        image = batch["image"]
        model_context = self._extract_model_context(batch)
        return {"logits": self(image, **model_context), "index": batch["index"]}

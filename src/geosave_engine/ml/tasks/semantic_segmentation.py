from copy import deepcopy
from importlib import import_module
from typing import Any, Literal, Required, TypedDict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave_engine.ml.registry import StageSpec
from geosave_engine.ml.postprocessing.segmentation import apply_thresholds
from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.models.contract import ModelChain



class ModuleSpec(TypedDict, total=False):
    """Importable PyTorch class and its constructor arguments."""

    class_path: Required[str]
    init_args: dict[str, Any]


class OptimizerSpec(ModuleSpec, total=False):
    """Optimizer arguments and overrides keyed by exact model-chain stage names."""

    groups: dict[str, dict[str, Any]]


class LRSchedulerSpec(ModuleSpec, total=False):
    """Scheduler arguments and Lightning scheduling metadata."""

    interval: Literal["step", "epoch"]
    frequency: int
    monitor: str
    strict: bool
    name: str


def _resolve_class(path: str, base: type) -> type:
    module, _, name = path.rpartition(".")
    cls = getattr(import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, base):
        raise TypeError(f"{path!r} must name a {base.__name__} subclass")
    return cls


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
        model_chain: dict[str, StageSpec] | None = None,
        input_size: int | tuple[int, int] = 224,
        ignore_index: int = 255,
        criterion: ModuleSpec | None = None,
        optimizer: OptimizerSpec | None = None,
        lr_scheduler: LRSchedulerSpec | None = None,
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
            criterion = {"class_path": "torch.nn.CrossEntropyLoss"}
        if optimizer is None:
            optimizer = {
                "class_path": "torch.optim.AdamW", "init_args": {"lr": 1e-3}
            }
        self.save_hyperparameters()
        self.model_chain = model_chain

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        self.ignore_index = ignore_index

        self.optimizer_spec = optimizer
        self.scheduler_spec = lr_scheduler
        self.metrics_config = metrics
        if class_thresholds is not None and len(class_thresholds) != self.num_classes:
            raise ValueError(
                f"class_thresholds must have {self.num_classes} entries, "
                f"got {len(class_thresholds)}"
            )
        self._initial_class_thresholds = class_thresholds

        criterion_cls = _resolve_class(criterion["class_path"], nn.Module)
        self.criterion = criterion_cls(
            **{"ignore_index": ignore_index, **criterion.get("init_args", {})}
        )

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
        self.model = ModelChain(stages=stage_specs)

        initial = (
            torch.tensor(self._initial_class_thresholds)
            if self._initial_class_thresholds is not None
            else torch.full((self.num_classes,), 0.5)
        )
        self.register_buffer("class_thresholds", initial)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        groups = self.optimizer_spec.get("groups", {})
        named = self.model._modules
        unknown = groups.keys() - named.keys()
        if unknown:
            raise ValueError(
                f"Unknown model-chain optimizer groups: {sorted(unknown)}"
            )
        selected = {
            id(parameter)
            for stage in groups
            for parameter in named[stage].parameters()
        }
        param_groups = [
            {
                "params": [p for p in named[stage].parameters() if p.requires_grad],
                **options,
            }
            for stage, options in groups.items()
        ]
        remaining = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in selected
        ]
        if remaining:
            param_groups.append({"params": remaining})
        optimizer_cls = _resolve_class(self.optimizer_spec["class_path"], Optimizer)
        optimizer = optimizer_cls(
            param_groups, **self.optimizer_spec.get("init_args", {})
        )

        if self.scheduler_spec is None:
            return optimizer

        scheduler_cls = _resolve_class(self.scheduler_spec["class_path"], LRScheduler)
        scheduler = scheduler_cls(
            optimizer, **self.scheduler_spec.get("init_args", {})
        )
        metadata = {
            key: self.scheduler_spec[key]
            for key in ("interval", "frequency", "monitor", "strict", "name")
            if key in self.scheduler_spec
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, **metadata},
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

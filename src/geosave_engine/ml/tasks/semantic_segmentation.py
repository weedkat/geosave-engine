from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Mapping, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from geosave_engine.geodata.datasets import GeoDataset, stack_samples
from geosave_engine.geodata.datasets.geo_dataset import LayerName
from geosave_engine.geodata.pipeline import GeoPipeline
from geosave_engine.ml.callbacks.calibration import DenseCalibrationCallback
from geosave_engine.ml.core.factory import build_loss, build_model, build_optimizer, build_scheduler
from geosave_engine.ml.core.transforms import ImageAugmenter, ImageProcessor
from geosave_engine.ml.inference.sliding_window import sliding_window_inference
from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.models.contract import ContextChain
from geosave_engine.utils import colorize

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
        raise ValueError(f"{name} keys must be dense 0..{len(mapping) - 1}, got {sorted(mapping)}")


class SemanticSegmentationTask(LightningModule):
    """Standardized, config-only semantic segmentation task.

    Owns model construction, forward pass, sliding-window inference,
    postprocessing, and a generic supervised training loop. Fully usable via
    YAML alone — no subclassing needed for standard supervised segmentation.

    Batch keys default to ``image``/``label``/``mask``/``context`` but are
    configurable (``image_key``/``label_key``/``mask_key``) — point them at
    your GeoDataset's raw layer names directly (e.g. ``image_key="sentinel_2_l1c"``)
    instead of requiring a renaming step upstream.

    For custom training loops (semi-supervised, UniMatchV2, bespoke
    architectures), write an independent LightningModule instead of
    subclassing this — see the semantic_segmentation templates for the
    pattern. This class does not expect to be subclassed.

    Model paths:
        - Chain: ``stages={'encoder': ..., 'decoder': ..., 'head': ...}`` — each
          value a registry key or nn.Module class, built in dict order.
        - Monolith: ``stages={'model': ...}`` — a single nn.Module with one
          ``@model_context`` method. Same code path as the chain, just one
          entry — no separate monolith concept.

    Args:
        stages: Stage name to registry key (or nn.Module class), in build
            order. Defaults to ``{'encoder': 'dinov3', 'decoder': 'dpt', 'head': 'dense'}``.
            The first stage receives ``in_channels``/``input_size``, the last
            receives ``num_classes`` — both by position, not by name, so this
            works the same whether ``stages`` has one entry or several.
            ``in_channels``/``num_classes`` themselves come from ``band_map``/
            ``class_map`` (see below), not passed directly — one source of
            truth, no risk of a hand-typed count drifting from the map.
        input_size: Spatial patch size for sliding-window inference.
        image_key: Batch key holding the input image tensor.
        label_key: Batch key holding the label tensor.
        mask_key: Batch key holding the optional nodata mask.
        ignore_index: Class index excluded from loss and metrics.
        upsample_output: Bilinearly upsample logits to input spatial size.
        class_map: ``{class_id: class_name}`` for every output class, dense
            from 0. Required — ``num_classes`` is ``len(class_map)``.
        band_map: ``{channel_idx: band_name}`` for every input channel, dense
            from 0. Required — ``in_channels`` is ``len(band_map)``.
        color_map: ``{class_id: hex_color}`` for prediction visualization.
        mean_norm: Per-channel normalization mean. Overrides model attribute.
        std_norm: Per-channel normalization std. Overrides model attribute.
        overlap_ratio: Sliding-window patch overlap fraction.
        pad_size: Reflect-padding added on each side before sliding window.
        config: Stage name to that stage's own constructor kwargs (e.g. ``{'encoder': {...}}``).
        loss: Loss function registry key (e.g. ``"CELoss"``).
        optimizer: Optimizer registry key (e.g. ``"AdamW"``).
        scheduler: LR scheduler registry key. ``None`` disables scheduling.
        metrics: Metric names in dot notation (e.g. ``["iou.macro", "f1.macro"]``).
        augmentations: Kornia augmentation config list.
        threshold_calibration_config: kwargs forwarded to ``DenseCalibrationCallback``.
        log_image_every_n_epochs: Epoch frequency for prediction visualization logging.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave_engine.ml.tasks.SemanticSegmentationTask
          init_args:
            stages:
              encoder: dinov3
              decoder: dpt
              head: dense
            image_key: sentinel_2_l1c
            label_key: dynamicworld
            mask_key: cloud_mask
            class_map: {0: water, 1: trees}
            band_map: {0: B02, 1: B03}
        data:
          class_path: geosave_engine.ml.tasks.SemanticSegmentationDataModule
          init_args:
            root: workspace/data
    """

    model: ContextChain

    def __init__(
        self,
        *,
        stages: dict[str, str] | dict[str, type[nn.Module]] | None = None,
        class_map: dict[int, str],
        band_map: dict[int, str],
        input_size: int | tuple[int, int] = 224,
        image_key: str = 'image',
        label_key: str = 'label',
        mask_key: str = 'mask',
        ignore_index: int = 255,
        upsample_output: bool = True,
        color_map: dict | None = None,
        mean_norm: list[float] | None = None,
        std_norm: list[float] | None = None,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
        config: dict | None = None,
        loss: str = 'CELoss',
        optimizer: str = 'AdamW',
        scheduler: str | None = None,
        metrics: list[str] | None = None,
        augmentations: list[dict] | None = None,
        threshold_calibration_config: dict | None = None,
        log_image_every_n_epochs: int = 2,
    ) -> None:
        super().__init__()

        # Reassign the local before save_hyperparameters() (frame-inspection based)
        # so it captures the resolved dict, not None, when the caller omits it.
        stages = stages or {'encoder': 'dinov3', 'decoder': 'dpt', 'head': 'dense'}
        ignore_hparams = ['stages'] if any(isinstance(v, type) for v in stages.values()) else None
        self.save_hyperparameters(ignore=ignore_hparams)
        self.stages = stages

        _validate_dense_map('class_map', class_map)
        _validate_dense_map('band_map', band_map)
        self.num_classes = len(class_map)
        self.in_channels = len(band_map)
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.image_key = image_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.ignore_index = ignore_index
        self.upsample_output = upsample_output
        self.class_map = class_map
        self.band_map = band_map
        self.color_map = color_map
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.overlap_ratio = overlap_ratio
        self.pad_size = pad_size
        self.config = config or {}

        self.loss_name = loss
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.metrics_config = metrics
        self.augmentations = augmentations or []
        self.threshold_calibration_config = threshold_calibration_config or {}
        self.log_image_every_n_epochs = log_image_every_n_epochs

        self.loss_fn = build_loss(loss, {**self.config.get('loss', {}), 'ignore_index': ignore_index})

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_model(self) -> None:
        """Build self.model, preprocessor, augmenter, and class_thresholds buffer.

        The first stage always consumes the raw image, the last always
        produces the final output — true whether ``stages`` has one entry
        (monolith) or several (chain) — so ``in_channels``/``input_size`` and
        ``num_classes`` route by position, not by a fixed stage name.
        """
        if hasattr(self, 'model'):
            return

        stage_names = list(self.stages)
        first, last = stage_names[0], stage_names[-1]

        stage_config = {name: dict(self.config.get(name) or {}) for name in stage_names}
        stage_config[first] = {'in_channels': self.in_channels, 'input_size': self.input_size, **stage_config[first]}
        stage_config[last] = {'num_classes': self.num_classes, **stage_config[last]}

        self.model = build_model(self.stages, stage_config)
        norm_source: nn.Module = getattr(self.model, first)

        self.preprocessor = ImageProcessor(
            in_channels=self.in_channels,
            model=norm_source,
            mean_norm=self.mean_norm,
            std_norm=self.std_norm,
        )
        self.register_buffer('class_thresholds', torch.full((self.num_classes,), 0.5))
        self.augmenter = ImageAugmenter(augmentations=self.augmentations, size=self.input_size)

    def configure_callbacks(self):
        return [DenseCalibrationCallback(**self.threshold_calibration_config)]

    def configure_optimizers(self):
        optimizer = build_optimizer(self.optimizer_name, self.model, self.config.get('optimizer') or {})
        if self.scheduler_name is None:
            return optimizer
        scheduler = build_scheduler(self.scheduler_name, optimizer, self.config.get('scheduler') or {})
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def setup(self, stage: str | None = None) -> None:
        metrics = SemanticSegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            labels=[self.class_map[i] for i in range(self.num_classes)],
            metrics=self.metrics_config,
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    # ------------------------------------------------------------------
    # Model forward
    # ------------------------------------------------------------------

    def _forward_ctx(self, ctx: dict[str, Any]) -> torch.Tensor:
        h, w = ctx['image'].shape[-2:]
        result = self.model(ctx)
        logits: torch.Tensor = result if isinstance(result, torch.Tensor) else result['logits']
        if self.upsample_output and logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits

    def forward_sliding(
        self,
        image: torch.Tensor,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
        context: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Sliding-window inference with Hann blending over a large raster.

        Expects ``image`` to be already preprocessed (``forward`` handles this).

        Args:
            image: ``[B, C, H, W]`` preprocessed float tensor.
            overlap_ratio: Patch overlap fraction in ``[0, 1)``.
            pad_size: Reflect-padding added on each side before patching.
            context: Context dict forwarded to each patch (geo metadata, etc.).

        Returns:
            ``[B, num_classes, H, W]`` logits at full input resolution.
        """
        base_ctx = context or {}

        def model_fn(patch: torch.Tensor) -> torch.Tensor:
            return self._forward_ctx({'image': patch, **base_ctx})

        return sliding_window_inference(model_fn, image, self.input_size, overlap_ratio, pad_size)

    def forward(self, image: torch.Tensor, **context: Any) -> torch.Tensor:
        """Preprocess then run model.

        Training: direct chain call on the input patch.
        Eval/predict: sliding-window inference with Hann blending.

        Args:
            image: ``[B, C, H, W]`` float image tensor.
            **context: Geo metadata forwarded as-is to the model chain.
        """
        image = self.preprocessor(image)
        ctx = {'image': image, **context}

        if self.training:
            return self._forward_ctx(ctx)
        return self.forward_sliding(image, self.overlap_ratio, self.pad_size, context)

    # ------------------------------------------------------------------
    # Training / validation / test / predict
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        image, label = batch[self.image_key], batch[self.label_key]
        image, label = self.augmenter(image, label)
        label = label.squeeze(1)
        logits = self(image, **batch.get('context', {}))
        loss = self.loss_fn(logits, label)
        self.train_metrics.update(logits, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=image.shape[0])
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=image.shape[0])
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        image, label = batch[self.image_key], batch[self.label_key]
        label = label.squeeze(1)
        mask = batch.get(self.mask_key)
        if mask is not None:
            mask = mask.squeeze(1)
        logits = self(image, **batch.get('context', {}))
        loss = self.loss_fn(logits, label)
        self.val_metrics.update(logits, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if batch_idx == 0 and self.current_epoch % self.log_image_every_n_epochs == 0:
            self._log_prediction('val', logits, label, mask)

    def test_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        image, label = batch[self.image_key], batch[self.label_key]
        label = label.squeeze(1)
        mask = batch.get(self.mask_key)
        if mask is not None:
            mask = mask.squeeze(1)
        logits = self(image, **batch.get('context', {}))
        self.test_metrics.update(logits, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False)
        if batch_idx == 0 and self.current_epoch % self.log_image_every_n_epochs == 0:
            self._log_prediction('test', logits, label, mask)

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = batch[self.image_key]
        mask = batch.get(self.mask_key)
        if mask is not None:
            mask = mask.squeeze(1)
        logits = self(image, **batch.get('context', {}))
        return self.postprocess(logits, mask)

    def postprocess(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmax + per-class confidence threshold + optional nodata mask.

        Args:
            logits: ``[B, num_classes, H, W]`` raw model output.
            mask: Optional boolean ``[B, H, W]`` nodata mask. Masked pixels → ignore_index.

        Returns:
            ``(pred_label [B, H, W], pred_proba [B, H, W])``.
        """
        probs = logits.softmax(dim=1)
        max_probs, preds = probs.max(dim=1)

        class_thresholds = cast(torch.Tensor, self.class_thresholds)
        pixel_thresholds = torch.index_select(class_thresholds, 0, preds.reshape(-1)).view_as(preds)
        preds = torch.where(max_probs >= pixel_thresholds, preds, preds.new_full((), self.ignore_index))

        if mask is not None:
            preds = torch.where(mask.bool(), preds.new_full((), self.ignore_index), preds)

        return preds, max_probs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_prediction(
        self,
        prefix: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if not self.color_map:
            warnings.warn(
                f"{type(self).__name__}: no color_map defined; skipping prediction logging.",
                UserWarning,
                stacklevel=2,
            )
            return
        preds, _ = self.postprocess(logits, mask)
        label_rgb = colorize(labels[0], self.color_map)
        pred_rgb = colorize(preds[0], self.color_map)
        step = self.current_epoch
        for lg in self.loggers:
            if isinstance(lg, TensorBoardLogger):
                writer = lg.experiment
                writer.add_image(f'{prefix}/label', label_rgb, step, dataformats='HWC')
                writer.add_image(f'{prefix}/prediction', pred_rgb, step, dataformats='HWC')
            elif isinstance(lg, MLFlowLogger):
                lg.experiment.log_image(lg.run_id, label_rgb, f'{prefix}_label_{step}.png')
                lg.experiment.log_image(lg.run_id, pred_rgb, f'{prefix}_prediction_{step}.png')


class SemanticSegmentationDataModule(LightningDataModule):
    """Generic datamodule pairing with SemanticSegmentationTask.

    Reads already-ingested GeoDataset directories, one per split — raw layer
    names pass through unchanged. Pair with ``SemanticSegmentationTask``'s
    ``image_key``/``label_key``/``mask_key`` to point the task at whatever
    layer names your GeoDataset actually produces. Ingestion itself (running
    your Pipelines) is not this class's job — point each root at a directory
    that already has ``<root>/<layer_name>/*.zarr`` written.

    Each split's root is its own param (not a fixed subfolder name under one
    shared root) — splits routinely live in unrelated places (e.g. a predict
    root pointing at a fresh inference AOI, nothing to do with where
    train/val/test were ingested), so baking in a naming convention would
    just force awkward symlinks/copies to satisfy it.

    Args:
        train_root: GeoDataset directory for the train split. Required for
            ``fit``.
        val_root: GeoDataset directory for the val split. Required for
            ``fit``/``validate``.
        test_root: GeoDataset directory for the test split. Required for
            ``test``.
        predict_root: GeoDataset directory for the predict split. Required
            for ``predict``.
        sel_bands: Layer name → band names to keep; default is all bands.
        dtype_override: Layer name to torch dtype to cast that layer's tensor
            to. Only needed to deviate from the tensor's saved dtype (e.g.
            cast a uint8 label layer to int64 for cross-entropy loss).
        pipeline: GeoPipeline whose ``context()`` supplies per-sample context
            (e.g. crs/transform/datetime for PredictionWriter, or derived
            values like day-of-year for a model that needs it). None omits
            context entirely — instantiating a pipeline just for this must
            not require live ingestion resources (network/credentials); it's
            the pipeline author's job to keep construction cheap.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave_engine.ml.tasks.SemanticSegmentationTask
          init_args:
            image_key: sentinel_2_l1c
            label_key: dynamicworld
            mask_key: cloud_mask
        data:
          class_path: geosave_engine.ml.tasks.SemanticSegmentationDataModule
          init_args:
            train_root: workspace/data/dynamicworld/train
            val_root: workspace/data/dynamicworld/val
            test_root: workspace/data/dynamicworld/test
            pipeline:
              class_path: modules.data_pipeline.Pipeline
    """

    def __init__(
        self,
        *,
        train_root: str | Path | None = None,
        val_root: str | Path | None = None,
        test_root: str | Path | None = None,
        predict_root: str | Path | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        pipeline: GeoPipeline | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.train_root = Path(train_root) if train_root is not None else None
        self.val_root = Path(val_root) if val_root is not None else None
        self.test_root = Path(test_root) if test_root is not None else None
        self.predict_root = Path(predict_root) if predict_root is not None else None
        self.sel_bands = sel_bands
        self.dtype_override = dtype_override
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def _make_dataset(self, name: str, root: Path | None) -> GeoDataset:
        if root is None:
            raise ValueError(f"{name} not set — pass `{name}` to build this split's dataset.")
        context_fn = self.pipeline.context if self.pipeline is not None else None
        return GeoDataset(root, sel_bands=self.sel_bands, dtype_override=self.dtype_override, context_fn=context_fn)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self.train_dataset = self._make_dataset("train_root", self.train_root)
            self.val_dataset = self._make_dataset("val_root", self.val_root)
        elif stage == "validate":
            self.val_dataset = self._make_dataset("val_root", self.val_root)
        elif stage == "test":
            self.test_dataset = self._make_dataset("test_root", self.test_root)
        elif stage == "predict":
            self.predict_dataset = self._make_dataset("predict_root", self.predict_root)
        else:
            raise ValueError(f"Invalid stage: {stage!r}")

    def _loader(self, dataset: GeoDataset, *, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=stack_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_dataset)

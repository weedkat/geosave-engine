from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from litdata import StreamingDataLoader
from torch.utils.data import DataLoader, Dataset

from geosave_engine.geodata.datasets import StackDataset, StoreDataset, stack_samples
from geosave_engine.geodata.datasets.stack import LayerName
from geosave_engine.ml.callbacks.prediction_logger import DensePredictionLogger
from geosave_engine.ml.callbacks.threshold_calibrator import ThresholdCalibrator
from geosave_engine.ml.registry import build_loss, build_model, build_optimizer, build_scheduler
from geosave_engine.ml.inference.sliding_window import split_patches, stitch_patches
from geosave_engine.ml.inference.thresholding import apply_thresholds
from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics
from geosave_engine.ml.models.contract import ContextChain
from geosave_engine.ml.transforms import ImageAugmenter, Normalize

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
    postprocessing, and training. Fully usable via YAML, no subclassing needed.

    Batch keys default to ``image``/``label``/``mask``/``context`` but are
    configurable (``image_key``/``label_key``/``mask_key``) to match your
    StackDataset's own layer names.

    For custom training loops, write an independent LightningModule
    instead — this class does not expect to be subclassed.

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
        threshold_calibration_config: Sweep-tuning kwargs forwarded to the
            auto-attached ``ThresholdCalibrator`` (``threshold_begin``/
            ``threshold_end``/``threshold_steps``/``metric``). ``num_classes``/
            ``ignore_index`` come from ``class_map``/``ignore_index`` above,
            not this dict.
        class_thresholds: Initial per-class confidence threshold, one per
            ``class_map`` entry. ``None`` starts every class at ``0.5`` — a
            placeholder ``ThresholdCalibrator`` overwrites once it calibrates,
            or a real deployed value if you already know good thresholds and
            don't need calibration.
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
    class_thresholds: torch.Tensor

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
        class_thresholds: list[float] | None = None,
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
        if class_thresholds is not None and len(class_thresholds) != self.num_classes:
            raise ValueError(
                f"class_thresholds must have {self.num_classes} entries (one per class_map "
                f"entry), got {len(class_thresholds)}"
            )
        self._initial_class_thresholds = class_thresholds
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

        self.preprocessor = Normalize(
            model=norm_source,
            mean_norm=self.mean_norm,
            std_norm=self.std_norm,
        )

        # 0.5 is a placeholder shape-holder when class_thresholds isn't given, not a
        # real default. A real value only exists after ThresholdCalibrator calibrates
        # it, after the caller passes class_thresholds explicitly, or after Lightning's
        # own load_from_checkpoint (which calls configure_model, then load_state_dict)
        # overwrites this buffer with the checkpoint's saved value.
        initial = (
            torch.tensor(self._initial_class_thresholds)
            if self._initial_class_thresholds is not None
            else torch.full((self.num_classes,), 0.5)
        )
        self.register_buffer('class_thresholds', initial)
        self.augmenter = ImageAugmenter(
            augmentations=self.augmentations, 
            size=self.input_size,
            data_keys=['image', 'mask'],
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
    
    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = [
            ThresholdCalibrator(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                **self.threshold_calibration_config,
            )
        ]
        # No color_map means nothing to render — don't add a callback that
        # would just warn and skip every eligible batch forever.
        if self.color_map:
            callbacks.append(DensePredictionLogger(
                color_map=self.color_map,
                class_map=self.class_map,
                log_image_every_n_epochs=self.log_image_every_n_epochs,
            ))
        return callbacks

    # ------------------------------------------------------------------
    # Model forward
    # ------------------------------------------------------------------

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Resize + normalize. ``forward()`` calls this internally too —

        Exposed standalone for introspection/composition, not as a step
        callers need to remember: ``forward()`` always applies it, so
        there's no way to feed the model un-preprocessed data by mistake.

        Args:
            image: ``[B, C, H, W]`` raw tensor.

        Returns:
            ``[B, C, H, W]`` resized + normalized tensor.
        """
        return self.preprocessor(image)

    def forward(self, image: torch.Tensor, **ctx: Any) -> torch.Tensor:
        """Preprocess then run one tile through the model chain.

        Always exactly one tile in, logits out — no sliding-window branching.
        For an image larger than `input_size`, use `predict()`/`forward_sliding()`
        instead — they handle the sliding window and call this per patch.

        Args:
            image: ``[B, C, H, W]`` float image tensor, exactly `input_size`.
            **ctx: Extra per-model context (e.g. `temporal_coords=...`,
                `location_coords=...` — usually built by `_extract_context`
                from a `pipeline.context()`-supplied batch, not passed by
                hand), forwarded to the model chain unchanged. Only consumed
                by whichever stage's `@model_context` method actually names
                the key — unused keys sit in the chain's ctx dict untouched.

        Returns:
            ``[B, num_classes, H, W]`` logits.
        """
        ctx = {'image': self.preprocess(image), **ctx}
        result = self.model(ctx)
        return result if isinstance(result, torch.Tensor) else result['logits']
    
    def forward_sliding(
        self,
        image: torch.Tensor,
        context: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sliding-window inference for an image larger than `input_size`.

        Always raw logits — what ``validation_step``/``test_step`` need for
        loss/metrics. For a finished prediction, use ``predict()`` instead.

        Args:
            image: ``[B, C, H, W]`` raw tensor, any size.
            context: Forwarded to every patch's ``forward()`` call unchanged.
                Empty if not given.

        Returns:
            ``[B, num_classes, H, W]`` logits at full input resolution.
        """
        context = context or {}
        patches = split_patches(image, self.input_size, self.overlap_ratio, self.pad_size)
        predictions = [self(patch, **context) for patch in patches]
        original_shape = (image.shape[-2], image.shape[-1])
        return stitch_patches(predictions, original_shape, self.input_size, self.overlap_ratio, self.pad_size)

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
            ``(pred_label [B, H, W] uint8, pred_proba [B, H, W] float32)``.
        """
        preds, max_probs = apply_thresholds(logits, self.class_thresholds, self.ignore_index, mask)

        preds = preds.to(torch.uint8)
        max_probs = max_probs.to(torch.float32)

        return preds, max_probs
    
    def predict(
        self,
        image: torch.Tensor,
        context: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finished prediction: sliding-window inference, always postprocessed.

        The public deployment entry point — works standalone, no Trainer
        needed (unlike ``predict_step``, which only runs through
        ``Trainer.predict()``'s own loop).

        Args:
            image: ``[B, C, H, W]`` raw tensor, any size.
            context: Forwarded to every patch's ``forward()`` call unchanged.
                Empty if not given.
            mask: Optional boolean ``[B, H, W]`` nodata mask. Masked pixels → ignore_index.

        Returns:
            ``(pred_label [B, H, W] uint8, pred_proba [B, H, W] float32)``.
        """
        logits = self.forward_sliding(image, context)
        return self.postprocess(logits, mask)

    # ------------------------------------------------------------------
    # Training / validation / test / predict
    # ------------------------------------------------------------------

    def _extract_context(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Every batch key besides image/label/mask/geobox/geotags — the ingested `GeoStack.context`.

        `image_key`/`label_key`/`mask_key` are this task's own tensors,
        `"geobox"`/`"geotags"` are `GeoStack.to_tensor`'s always-present
        identity keys (see `docs/concept/model.md`) — none of these five
        are model context. Everything else in `batch` came from the
        pipeline's own `context()` override (e.g. `temporal_coords`/
        `location_coords`), computed once at ingest time and already baked
        into every saved `StackDataset` sample — nothing to configure here.

        Args:
            batch: One `DataLoader` batch, as `stack_samples` produces it.

        Returns:
            Extra keys to forward into `self(image, **context)` — `{}` if
            the ingesting pipeline's `context()` returned none.
        """
        exclude = {self.image_key, self.label_key, self.mask_key, 'geobox', 'geotags'}
        return {key: value for key, value in batch.items() if key not in exclude}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        image, label = batch[self.image_key], batch[self.label_key]
        context = self._extract_context(batch)
        image, label = self.augmenter(image, label)
        label = label.squeeze(1) # (B, 1, H, W) → (B, H, W)

        logits = self(image, **context)
        loss = self.loss_fn(logits, label)

        self.train_metrics.update(logits, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=image.shape[0])
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=image.shape[0])

        return loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        image, label = batch[self.image_key], batch[self.label_key]
        context = self._extract_context(batch)
        label = label.squeeze(1) # (B, 1, H, W) → (B, H, W)

        logits = self.forward_sliding(image, context)
        loss = self.loss_fn(logits, label)

        self.val_metrics.update(logits, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # DensePredictionLogger reads logits/label via on_validation_batch_end's own
        # outputs arg — raw model output only, no postprocess (class_thresholds isn't
        # calibrated until on_fit_end runs, so applying it mid-training adds no signal over
        # plain argmax).
        return {'logits': logits, 'label': label}

    def test_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        image, label = batch[self.image_key], batch[self.label_key]
        context = self._extract_context(batch)
        label = label.squeeze(1) # (B, 1, H, W) → (B, H, W)

        logits = self.forward_sliding(image, context)

        self.test_metrics.update(logits, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False)
        # DensePredictionLogger reads logits/label via on_test_batch_end's own outputs arg.
        return {'logits': logits, 'label': label}

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        image = batch[self.image_key]
        context = self._extract_context(batch)
        mask = batch.get(self.mask_key)
        if mask is not None:
            mask = mask.squeeze(1) # (B, 1, H, W) → (B, H, W)

        preds, max_probs = self.predict(image, context, mask=mask)
        return {"pred": preds, "proba": max_probs}


class SemanticSegmentationDataModule(LightningDataModule):
    """Generic datamodule pairing with SemanticSegmentationTask.

    Reads already-ingested StackDataset directories, one per split — raw layer
    names pass through unchanged. Pair with ``SemanticSegmentationTask``'s
    ``image_key``/``label_key``/``mask_key`` to point the task at whatever
    layer names your StackDataset actually produces. Ingestion itself (running
    your Pipelines) is not this class's job — point each root at a directory
    that already has ``<root>/<layer_name>/*.zarr`` written.

    Each split's root is its own param (not a fixed subfolder name under one
    shared root) — splits routinely live in unrelated places (e.g. a predict
    root pointing at a fresh inference AOI, nothing to do with where
    train/val/test were ingested), so baking in a naming convention would
    just force awkward symlinks/copies to satisfy it.

    Args:
        train_root: StackDataset directory for the train split. Required for
            ``fit``.
        val_root: StackDataset directory for the val split. Required for
            ``fit``/``validate``.
        test_root: StackDataset directory for the test split. Required for
            ``test``.
        predict_root: StackDataset directory for the predict split. Required
            for ``predict``.
        dataset: `"stack"` (raw `GeoStack` zarr, via `StackDataset`) or
            `"store"` (packed `SampleStore`, via `StoreDataset`) — or pass a
            `Dataset` class directly for anything outside those two.
        dataset_kwargs: Extra kwargs forwarded to `dataset`'s constructor —
            e.g. `required_layers` for `StackDataset`, `cache_dir`/`shuffle`
            for `StoreDataset`. `sel_bands`/`dtype_override` below are
            already forwarded, don't repeat them here.
        sel_bands: Layer name → band names to keep; default is all bands.
        dtype_override: Layer name to torch dtype to cast that layer's tensor
            to. Only needed to deviate from the tensor's saved dtype (e.g.
            cast a uint8 label layer to int64 for cross-entropy loss).
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
    """

    def __init__(
        self,
        *,
        train_root: str | Path | None = None,
        val_root: str | Path | None = None,
        test_root: str | Path | None = None,
        predict_root: str | Path | None = None,
        dataset: Literal["stack", "store"] = "stack",
        dataset_kwargs: dict[str, Any] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
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
        self.dataset = dataset
        self.dataset_kwargs = dataset_kwargs or {}
        self.sel_bands = sel_bands
        self.dtype_override = dtype_override
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def _make_dataset(self, name: str, root: Path | None) -> Dataset:
        if root is None:
            raise ValueError(f"{name} not set — pass `{name}` to build this split's dataset.")
        if self.dataset == "stack":
            return StackDataset(root, sel_bands=self.sel_bands, dtype_override=self.dtype_override, **self.dataset_kwargs)
        if self.dataset == "store":
            return StoreDataset(root, sel_bands=self.sel_bands, dtype_override=self.dtype_override, **self.dataset_kwargs)
        raise ValueError(f"Unknown dataset {self.dataset!r} — must be 'stack' or 'store'")

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

    def _loader(self, dataset: Dataset, *, drop_last: bool = False) -> DataLoader:
        # StoreDataset is a litdata StreamingDataset — a plain DataLoader would iterate it
        # fine but skip litdata's own shuffle/num_workers/checkpoint-resume wiring, which
        # only StreamingDataLoader sets up (it calls dataset.set_shuffle()/set_num_workers()
        # before construction; StreamingDataset even rejects a bare DataLoader outright).
        if self.dataset == "store":
            assert isinstance(dataset, StoreDataset)
            return StreamingDataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=drop_last,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                collate_fn=stack_samples,
            )
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

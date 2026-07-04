from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from geosave_engine.ml.core.transforms import ImageProcessor
from geosave_engine.ml.inference.sliding_window import sliding_window_inference
from geosave_engine.ml.models.contract import ContextChain
from geosave_engine.ml.models.decoder.dpt import DPTDecoder
from geosave_engine.ml.models.decoder.unet import UnetDecoder
from geosave_engine.ml.models.encoder.dinov3 import DINOv3
from geosave_engine.ml.models.head.dense import DenseHead
from geosave_engine.utils import filter_kwargs

log = logging.getLogger(__name__)


class SemanticSegmentationTask(LightningModule):
    """Semantic segmentation model construction and inference base.

    Owns model construction, forward pass, sliding-window inference, and
    postprocessing. Use SupervisedSegmentationTask for a full supervised
    training loop, or subclass and add a custom training_step.

    Model paths:
        - Encoder/decoder/head: pass string registry keys or nn.Module classes.
        - Monolith: pass ``monolith`` — a single nn.Module with one
          ``@model_context`` method. ``self.model`` is the monolith directly.

    Args:
        encoder: Registry key or nn.Module class for encoder.
        decoder: Registry key or nn.Module class for decoder.
        head: Registry key or nn.Module class for head.
        monolith: Pre-built nn.Module overriding encoder/decoder/head.
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        input_size: Spatial patch size for sliding-window inference.
        ignore_index: Class index excluded from loss and metrics.
        upsample_output: Bilinearly upsample logits to input spatial size.
        class_map: ``{class_id: class_name}`` for metrics and visualization.
        band_map: ``{band_name: channel_idx}`` saved in artifact for reconstruction.
        palette: ``{class_id: hex_color}`` for prediction visualization.
        mean_norm: Per-channel normalization mean. Overrides model attribute.
        std_norm: Per-channel normalization std. Overrides model attribute.
        overlap_ratio: Sliding-window patch overlap fraction.
        pad_size: Reflect-padding added on each side before sliding window.
        config: Nested config overrides for model components (encoder/decoder/head/monolith).

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave_engine.ml.tasks.SupervisedSegmentationTask
          init_args:
            encoder: dinov3
            decoder: dpt
            num_classes: 8
            in_channels: 13
    """

    ENCODERS: ClassVar[dict[str, type[nn.Module]]] = {'dinov3': DINOv3}
    DECODERS: ClassVar[dict[str, type[nn.Module]]] = {'dpt': DPTDecoder, 'unet': UnetDecoder}
    HEADS: ClassVar[dict[str, type[nn.Module]]] = {'dense': DenseHead}
    MONOLITHS: ClassVar[dict[str, type[nn.Module]]] = {}

    model: ContextChain

    def __init__(
        self,
        encoder: str | type[nn.Module] = 'dinov3',
        decoder: str | type[nn.Module] = 'dpt',
        head: str | type[nn.Module] = 'dense',
        monolith: str | type[nn.Module] | None = None,
        num_classes: int = 1,
        in_channels: int = 3,
        input_size: int | tuple[int, int] = 224,
        ignore_index: int = 255,
        upsample_output: bool = True,
        class_map: dict | None = None,
        band_map: dict | None = None,
        palette: dict | None = None,
        mean_norm: list[float] | None = None,
        std_norm: list[float] | None = None,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
        config: dict | None = None,
    ) -> None:
        super().__init__()

        _nn_args = {'encoder': encoder, 'decoder': decoder, 'head': head, 'monolith': monolith}
        ignore_hparams = [k for k, v in _nn_args.items() if isinstance(v, type)]
        self.save_hyperparameters(ignore=ignore_hparams or None)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.ignore_index = ignore_index
        self.upsample_output = upsample_output
        self.class_map = class_map
        self.band_map = band_map
        self.palette = palette
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.overlap_ratio = overlap_ratio
        self.pad_size = pad_size
        self.config = config or {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_model(self) -> None:
        """Build self.model, preprocessor, and class_thresholds buffer."""
        if hasattr(self, 'model'):
            return

        encoder = self.hparams.encoder
        decoder = self.hparams.decoder
        head = self.hparams.head
        monolith = self.hparams.monolith

        if monolith is not None:
            mon_cls = self.MONOLITHS[monolith] if isinstance(monolith, str) else monolith
            monolith_instance = mon_cls(**filter_kwargs(mon_cls, {
                'num_classes': self.num_classes,
                'in_channels': self.in_channels,
                'input_size': self.input_size,
                **dict(self.config.get('monolith') or {}),
            }))
            self.model = ContextChain({'model': monolith_instance})
            norm_source: nn.Module = monolith_instance
        else:
            enc_cls = self.ENCODERS[encoder] if isinstance(encoder, str) else encoder
            enc = enc_cls(in_channels=self.in_channels, **dict(self.config.get('encoder') or {}))

            dec_cls = self.DECODERS[decoder] if isinstance(decoder, str) else decoder
            dec = dec_cls(
                encoder_out_channels=enc.out_channels,
                encoder_output_strides=enc.output_strides,
                **dict(self.config.get('decoder') or {}),
            )

            hd_cls = self.HEADS[head] if isinstance(head, str) else head
            hd = hd_cls(
                in_channels=dec.out_channels,
                num_classes=self.num_classes,
                **dict(self.config.get('head') or {}),
            )

            self.model = ContextChain({'encoder': enc, 'decoder': dec, 'head': hd})
            norm_source = enc

        self.preprocessor = ImageProcessor(
            model=norm_source,
            mean_norm=self.mean_norm,
            std_norm=self.std_norm,
        )
        self.register_buffer('class_thresholds', torch.full((self.num_classes,), 0.5))

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
        context: dict[str, Any] | None = None,
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
            **context: Keys forwarded as-is to the model chain.
        """
        image = self.preprocessor(image)
        ctx = {'image': image, **context}

        if self.training:
            return self._forward_ctx(ctx)
        return self.forward_sliding(image, self.overlap_ratio, self.pad_size, context)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = batch['image']
        mask = batch.get('mask')
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

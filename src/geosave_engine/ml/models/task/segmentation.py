from __future__ import annotations

import torch
import torch.nn.functional as F

from typing import ClassVar, Mapping

from geosave_engine.ml.inference.sliding_window import sliding_window_inference
import torch.nn as nn

from geosave_engine.ml.models.contract import ModelContext
from geosave_engine.ml.models.contract.chain import _discover_chain
from geosave_engine.ml.models.encoder.dinov3 import DINOv3
from geosave_engine.ml.models.decoder.dpt import DPTDecoder
from geosave_engine.ml.models.decoder.unet import UnetDecoder
from geosave_engine.ml.models.head.dense import DenseHead


class SemanticSegmentation(nn.Module):
    """Encoder → Decoder → Head segmentation pipeline.

    Wires three nn.Module instances into a chain. Each module must declare
    exactly one @model_context method — ambiguity raises at construction.
    Primary input key is always ``'image'``; ``'logits'`` is the output key.

    Args:
        encoder: Registry key or nn.Module class.
        decoder: Registry key or nn.Module class.
        head: Registry key or nn.Module class.
        in_channels: Input image channel count, forwarded to encoder.
        num_classes: Output channel count, forwarded to head.
        input_size: Spatial patch size for sliding-window inference.
        encoder_config: Extra encoder constructor kwargs.
        decoder_config: Decoder constructor overrides.
        head_config: Head constructor overrides.
        upsample_output: Bilinearly upsample logits to input spatial size.
    """

    ENCODERS: ClassVar[dict[str, type[nn.Module]]] = {
        'dinov3': DINOv3,
    }
    DECODERS: ClassVar[dict[str, type[nn.Module]]] = {
        'dpt': DPTDecoder,
        'unet': UnetDecoder,
    }
    HEADS: ClassVar[dict[str, type[nn.Module]]] = {
        'dense': DenseHead,
    }

    def __init__(
        self,
        encoder: str | type[nn.Module] = 'dinov3',
        decoder: str | type[nn.Module] = 'dpt',
        head: str | type[nn.Module] = 'dense',
        in_channels: int = 3,
        num_classes: int = 1,
        input_size: tuple[int, int] | int = (256, 256),
        encoder_config: Mapping[str, object] | None = None,
        decoder_config: Mapping[str, object] | None = None,
        head_config: Mapping[str, object] | None = None,
        upsample_output: bool = True,
    ):
        super().__init__()
        self.upsample_output = upsample_output
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        enc_cls = self.ENCODERS[encoder] if isinstance(encoder, str) else encoder
        self.encoder = enc_cls(in_channels=in_channels, **dict(encoder_config or {}))

        dec_cls = self.DECODERS[decoder] if isinstance(decoder, str) else decoder
        self.decoder = dec_cls(
            encoder_out_channels=self.encoder.out_channels,
            encoder_output_strides=self.encoder.output_strides,
            **dict(decoder_config or {}),
        )

        hd_cls = self.HEADS[head] if isinstance(head, str) else head
        self.head = hd_cls(
            in_channels=self.decoder.out_channels,
            num_classes=num_classes,
            **dict(head_config or {}),
        )

        self._chain = _discover_chain([self.encoder, self.decoder, self.head])

    def forward(self, image: torch.Tensor, **sample_meta) -> torch.Tensor:
        """Run encoder→decoder→head chain; upsample logits to input resolution.

        Args:
            image: (B, C, H, W) input tensor.
            **sample_meta: Per-sample metadata forwarded to each submodule
                (e.g. ``crs``, ``coordinate``, ``transform``, ``time``).

        Returns:
            (B, num_classes, H, W) logits.
        """
        ctx = ModelContext(inputs={'image': image}, sample_meta=sample_meta)
        return self._forward_ctx(ctx).inputs['logits']

    def _forward_ctx(self, ctx: ModelContext) -> ModelContext:
        h, w = ctx.inputs['image'].shape[-2:]
        for module, method_name in self._chain:
            ctx = getattr(module, method_name)(ctx)
        logits = ctx.inputs['logits']
        if logits.shape[-2:] != (h, w) and self.upsample_output:
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            ctx = ModelContext(
                inputs={'logits': logits},
                sample_meta=ctx.sample_meta,
                metadata=ctx.metadata,
            )
        return ctx

    def forward_sliding(
        self,
        image: torch.Tensor,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
        **sample_meta,
    ) -> torch.Tensor:
        """Sliding-window inference over a large raster with Hann blending.

        Args:
            image: (B, C, H, W) input tensor.
            overlap_ratio: Patch overlap fraction. Must be in [0, 1).
            pad_size: Reflect-padding added on each side before patching.
            **sample_meta: Per-sample metadata forwarded to each patch call.

        Returns:
            (B, num_classes, H, W) logits at full input resolution.
        """
        ctx = ModelContext(inputs={'image': image}, sample_meta=sample_meta)

        def model_fn(patch: torch.Tensor) -> torch.Tensor:
            patch_ctx = ModelContext(
                inputs={'image': patch},
                sample_meta=ctx.sample_meta,
                metadata=ctx.metadata,
            )
            return self._forward_ctx(patch_ctx).inputs['logits']

        return sliding_window_inference(
            model_fn, image, self.input_size, overlap_ratio, pad_size
        )

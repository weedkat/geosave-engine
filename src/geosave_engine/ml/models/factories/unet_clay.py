from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from geosave_engine.ml.models.decoder.unet import UnetDecoder
from geosave_engine.ml.models.encoder.clay import (
    build_clay_v15,
    load_clay_metadata,
    resolve_clay_bands,
)
from geosave_engine.ml.models.head.unet import UnetRegHead, UnetSegHead


_CLAY_LARGE_DIM: int = 1024
_CLAY_PATCH_SIZE: int = 8


class UnetClay(nn.Module):
    """UNet decoder on a Clay v1.5 backbone (encoder-only).

    Pipeline (per forward call):
        x [B, C, H, W]
          -> Clay encoder (datacube)               (CLS-token + spatial tokens captured per chosen block via forward hooks)
          -> reshape each captured token map to    [B, D, H/patch, W/patch]
          -> UnetDecoder fuses the 4 stage maps  -> [B, decoder_channels[-1], H/2, W/2]
          -> head                                 -> [B, num_classes, H/2, W/2]
          -> bilinear interpolate to             -> [B, num_classes, H, W]

    Args:
        modality: Clay-supported sensor / product. Drives band count, wavelengths, gsd.
        bands: optional sub-selection of the modality's pretraining ``band_order``.
            ``None`` (default) uses the full ``band_order``.
        checkpoint_path: optional local Clay v1.5 checkpoint; ``None`` fetches from HF Hub.
        img_size: nominal training resolution. Clay tolerates other multiples of patch size
            via positional-encoding interpolation; stored only for documentation.
        out_indices: 4 transformer block indices to capture (default = even quarters of
            depth-24 large model).
        freeze_encoder: when ``True``, sets ``requires_grad=False`` on all Clay encoder params.
        decoder_channels: per-stage decoder widths (length 4).
        use_norm: insert BatchNorm2d in decoder DoubleConv blocks.
        task: ``'regression'`` (default for biomass/depth) or ``'segmentation'``.
        num_classes: number of output channels (``num_outputs`` for regression).
        head_dropout: Dropout2d probability before the final 1x1 projection.

    Attributes:
        img_mean (list[float]): per-band mean from Clay metadata (for normalization in datamodule).
        img_std  (list[float]): per-band std.
    """

    def __init__(
        self,
        # --- backbone (Clay v1.5) ---
        modality: Literal['sentinel-2-l2a', 'sentinel-1-rtc',
                          'landsat-c2l1', 'landsat-c2l2-sr',
                          'planetscope-sr', 'naip', 'linz', 'modis',
                          'satellogic-MSI-L1D'] = 'sentinel-2-l2a',
        bands: list[str] | None = None,
        checkpoint_path: str | None = None,
        img_size: int = 128,
        out_indices: tuple[int, int, int, int] = (5, 11, 17, 23),
        freeze_encoder: bool = False,
        # --- decoder (UNet) ---
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 32),
        use_norm: bool = True,
        # --- head ---
        task: Literal['regression', 'segmentation'] = 'regression',
        num_classes: int = 1,
        head_dropout: float = 0.0,
    ):
        super().__init__()

        # --- backbone ---
        meta = load_clay_metadata()
        self._platform: str = modality
        self._bands: list[str] = resolve_clay_bands(meta, modality, bands)
        self._in_channels: int = len(self._bands)
        wavelengths_nm = [float(meta[modality]['bands']['wavelength'][b]) * 1000.0
                          for b in self._bands]
        self.register_buffer('_waves',
                             torch.tensor(wavelengths_nm, dtype=torch.float32))
        self.register_buffer('_gsd',
                             torch.tensor(float(meta[modality]['gsd']), dtype=torch.float32))
        self.img_mean: list[float] = [float(meta[modality]['bands']['mean'][b]) for b in self._bands]
        self.img_std:  list[float] = [float(meta[modality]['bands']['std'][b])  for b in self._bands]

        self.encoder = build_clay_v15(
            model_size='large',
            checkpoint_path=checkpoint_path,
            metadata=meta,
        )
        self.patch_size: int = int(self.encoder.model.encoder.patch_size)  # 8
        self.img_size: int = img_size
        self.out_indices: list[int] = list(out_indices)

        # Forward-hook plumbing. Each `transformer.layers[i]` is a `ModuleList([attn, ff])`
        # — the container has no forward(); hook the FF submodule and reconstruct the
        # post-residual output as `ff(x_pre_ff) + x_pre_ff` (see Clay Transformer.forward).
        self._intermediates: dict[int, torch.Tensor] = {}
        for i in self.out_indices:
            ff_module = self.encoder.model.encoder.transformer.layers[i][1]
            ff_module.register_forward_hook(self._make_hook(i))

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # --- decoder ---
        embed_dim: int = int(self.encoder.model.encoder.dim)              # 1024 for large
        encoder_out_channels = [embed_dim] * len(self.out_indices)        # [1024, 1024, 1024, 1024]
        encoder_output_strides = [self.patch_size] * len(self.out_indices)  # [8, 8, 8, 8]
        self.decoder = UnetDecoder(
            encoder_out_channels=encoder_out_channels,
            encoder_output_strides=encoder_output_strides,
            decoder_channels=decoder_channels,
            use_norm=use_norm,
        )

        # --- head ---
        if task == 'regression':
            self.head = UnetRegHead(self.decoder.out_channels,
                                               num_outputs=num_classes,
                                               dropout=head_dropout)
        elif task == 'segmentation':
            self.head = UnetSegHead(self.decoder.out_channels,
                                    num_classes=num_classes,
                                    dropout=head_dropout)
        else:
            raise ValueError(
                f"task must be 'regression' or 'segmentation', got {task!r}"
            )

    def _make_hook(self, idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...],
                 output: torch.Tensor) -> None:
            # Clay's Transformer.forward: x = ff(x_pre_ff) + x_pre_ff
            # inputs[0] is x_pre_ff; output is ff(x_pre_ff).
            self._intermediates[idx] = output + inputs[0]
        return hook

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pixelwise prediction from a Clay-encoded image.

        Args:
            x: input chip ``[B, C, H, W]``. ``C`` must equal ``len(self._bands)``;
                ``H`` and ``W`` must be multiples of ``patch_size`` (8).
            time: per-sample temporal encoding ``[B, 4]`` (e.g. sin/cos of week + hour).
                ``None`` => zeros (Clay-documented default; location-/time-agnostic).
            latlon: per-sample geolocation encoding ``[B, 4]`` (e.g. sin/cos of lat + lon).
                ``None`` => zeros.

        Returns:
            ``[B, num_classes, H, W]`` — regression values (when ``task='regression'``,
            typically ``num_classes=1``) or segmentation logits.
        """
        B, C, H, W = x.shape
        if C != self._in_channels:
            raise ValueError(
                f"expected {self._in_channels} channels for modality {self._platform!r} "
                f"with bands {self._bands}, got {C}"
            )
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(
                f"input H={H}, W={W} not divisible by patch_size={self.patch_size}"
            )

        datacube: dict[str, object] = {
            'pixels':   x,                                                       # [B, C, H, W]
            'platform': self._platform,                                          # str
            'time':     time   if time   is not None else x.new_zeros(B, 4),     # [B, 4]
            'latlon':   latlon if latlon is not None else x.new_zeros(B, 4),     # [B, 4]
            'waves':    self._waves,                                             # [C]
            'gsd':      self._gsd,                                               # scalar
        }

        # Run Clay encoder. Final output unused; forward hooks populate `_intermediates`.
        self._intermediates.clear()
        _ = self.encoder.model.encoder(datacube)                                 # captured: [B, 1+N, D] per hooked block

        side = H // self.patch_size                                              # N = side * side
        features: list[torch.Tensor] = []
        for i in self.out_indices:
            tokens = self._intermediates[i]                                      # [B, 1+N, D]
            spatial = tokens[:, 1:, :]                                           # drop CLS -> [B, N, D]
            feat = spatial.transpose(1, 2).reshape(B, -1, side, side).contiguous()  # [B, D, H/8, W/8]
            features.append(feat)

        fused = self.decoder(features)                                           # [B, decoder_channels[-1], H/2, W/2]
        y = self.head(fused)                                                     # [B, num_classes, H/2, W/2]
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return y                                                                 # [B, num_classes, H, W]

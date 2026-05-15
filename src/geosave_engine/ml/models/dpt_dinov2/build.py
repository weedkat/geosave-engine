from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from .semseg.dpt import DPT
from geosave_engine.utils.weights import cached_weights_path, download_weights

EncoderName = Literal["dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant"]
WeightsName = Literal["imagenet", "none"]

@dataclass(frozen=True)
class EncoderPreset:
    """DPT encoder preset: backbone size plus decoder head channel widths."""

    encoder_size: str
    features: int
    out_channels: tuple[int, int, int, int]


DPT_ENCODER_PRESETS: dict[str, EncoderPreset] = {
    "dinov2_small": EncoderPreset(encoder_size="small", features=64,  out_channels=(48, 96, 192, 384)),
    "dinov2_base":  EncoderPreset(encoder_size="base",  features=128, out_channels=(96, 192, 384, 768)),
    "dinov2_large": EncoderPreset(encoder_size="large", features=256, out_channels=(256, 512, 1024, 1024)),
    "dinov2_giant": EncoderPreset(encoder_size="giant", features=384, out_channels=(1536, 1536, 1536, 1536)),
}

DPT_PRETRAINED_URLS: dict[str, dict[str, str]] = {
    "imagenet": {
        "dinov2_small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "dinov2_base": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "dinov2_large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        "dinov2_giant": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    }
}

def resolve_pretrained_dinov2(
    encoder: EncoderName,
    weights: WeightsName,
    cache_dir: Path,
) -> Path | None:
    """Return a local path to DINOv2 weights, downloading them if needed."""
    if weights == "none":
        return None

    if weights not in DPT_PRETRAINED_URLS:
        valid = [*DPT_PRETRAINED_URLS.keys(), "none"]
        raise ValueError(f"Unknown weights '{weights}'. Valid options: {valid}")

    urls = DPT_PRETRAINED_URLS[weights]
    if encoder not in urls:
        raise ValueError(f"Unknown encoder '{encoder}'. Valid options: {list(urls)}")

    destination = cached_weights_path(cache_dir, encoder)
    return download_weights(urls[encoder], destination)

def dpt_dinov2(encoder, num_classes, in_channels, weights="imagenet", pretrain_dir='./pretrained_weights', **kwargs) -> DPT:
    model = DPT(
        encoder_size=DPT_ENCODER_PRESETS[encoder].encoder_size,
        features=DPT_ENCODER_PRESETS[encoder].features,
        out_channels=list(DPT_ENCODER_PRESETS[encoder].out_channels),
        nclass=num_classes,
        in_chans=in_channels,
        **kwargs,
    )
    pth_path = resolve_pretrained_dinov2(encoder, weights, Path(pretrain_dir))
    if pth_path is not None:
        state_dict = torch.load(pth_path, map_location="cpu")
        model.backbone.load_state_dict(state_dict)

    return model

def dpt_dinov2_small(num_classes, in_channels, **kwargs) -> DPT:
    return dpt_dinov2(encoder="dinov2_small", num_classes=num_classes, in_channels=in_channels, **kwargs)

def dpt_dinov2_base(num_classes, in_channels, **kwargs) -> DPT:
    return dpt_dinov2(encoder="dinov2_base", num_classes=num_classes, in_channels=in_channels, **kwargs)

def dpt_dinov2_large(num_classes, in_channels, **kwargs) -> DPT:
    return dpt_dinov2(encoder="dinov2_large", num_classes=num_classes, in_channels=in_channels, **kwargs)

def dpt_dinov2_giant(num_classes, in_channels, **kwargs) -> DPT:
    return dpt_dinov2(encoder="dinov2_giant", num_classes=num_classes, in_channels=in_channels, **kwargs)


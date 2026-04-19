from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from geosave_engine.core.base import BaseModel
from geosave_engine.utils.pretrained import DEFAULT_CACHE_DIR, cached_weights_path, download_weights

from .semseg.dpt import DPT

EncoderName = Literal["dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant"]
WeightsName = Literal["imagenet", "none"]

DPT_ENCODER_PRESETS: dict[str, dict] = {
    "dinov2_small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
    "dinov2_base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
    "dinov2_large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "dinov2_giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
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


class DensePredictionTransformer(BaseModel):
    """Dense Prediction Transformer model wrapper for semantic segmentation."""

    tasks = {"semantic segmentation": []}
    doc_links = ["https://docs.geosave.dev/models/densepredictiontransformer"]
    model = DPT

    @classmethod
    def build(
        cls,
        in_channels: int,
        nclass: int,
        encoder: EncoderName = "dinov2_base",
        weights: WeightsName = "imagenet",
        pretrain_dir: str | Path = DEFAULT_CACHE_DIR,
        *args,
        **kwargs,
    ) -> DPT:
        if encoder not in DPT_ENCODER_PRESETS:
            raise ValueError(
                f"Unknown encoder '{encoder}'. Valid options: {list(DPT_ENCODER_PRESETS)}"
            )

        preset = DPT_ENCODER_PRESETS[encoder]
        model = cls.model(
            encoder_size=preset["encoder_size"],
            nclass=nclass,
            features=preset["features"],
            out_channels=preset["out_channels"],
            in_chans=in_channels,
            *args,
            **kwargs,
        )

        pth_path = resolve_pretrained_dinov2(encoder, weights, Path(pretrain_dir))
        if pth_path is not None:
            state_dict = torch.load(pth_path, map_location="cpu")
            model.backbone.load_state_dict(state_dict)

        return model

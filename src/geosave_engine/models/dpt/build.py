from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from geosave_engine.core.base import BaseModel

from .semseg.dpt import DPT

PRETRAINED_DIR = Path.home() / ".cache" / "geosave" / "pretrained"
EncoderName = Literal["dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant"]
WeightsName = Literal["imagenet", "none"]

# Defaults for each DINOv2 encoder preset.
dpt_encoder_map = {
    "dinov2_small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
    "dinov2_base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
    "dinov2_large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "dinov2_giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def download_pretrained_dinov2(
    encoder: EncoderName,
    weights: WeightsName = "imagenet",
    pretrain_dir: Path = PRETRAINED_DIR,
) -> Path | None:
    dpt_urls = {
        "imagenet": {
            "dinov2_small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "dinov2_base": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
            "dinov2_large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            "dinov2_giant": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
        }
    }

    if weights == "none":
        print("No pretrained weights, training from scratch")
        return None

    pretrain_dir.mkdir(parents=True, exist_ok=True)

    if weights not in dpt_urls:
        valid_options = [*list(dpt_urls.keys()), "none"]
        raise ValueError(f"{weights} not recognized. Valid options are: {valid_options}")

    urls = dpt_urls[weights]
    if encoder not in urls:
        raise ValueError(f"{encoder} not recognized. Valid options are: {list(urls.keys())}")

    pth_path = pretrain_dir / f"{encoder}.pth"
    if not pth_path.exists():
        print(f"Downloading pretrained DINOv2 {encoder} weights...")
        torch.hub.download_url_to_file(urls[encoder], str(pth_path))
        print("Download complete.")
    else:
        print(f"Pretrained DINOv2 {encoder} weights already exists at {pth_path}. Skipping download.")

    return pth_path


class DensePredictionTransformer(BaseModel):
    """Dense Prediction Transformer model wrapper for semantic segmentation."""
    tasks = {
        "semantic segmentation": []
        }
    doc_links = ["https://docs.geosave.dev/models/densepredictiontransformer"]
    model = DPT

    @classmethod
    def build(
        cls,
        in_channels: int,
        nclass: int,
        encoder: EncoderName = "dinov2_base",
        weights: WeightsName = "imagenet",
        pretrain_dir: str | Path = PRETRAINED_DIR,
        *args,
        **kwargs,
    ) -> DPT:
        if not callable(getattr(cls, "model", None)):
            raise ValueError(f"{cls.__name__}.model must be defined and callable")
            
        if encoder not in dpt_encoder_map:
            raise ValueError(f"Unknown encoder '{encoder}'. Valid options are: {list(dpt_encoder_map.keys())}")

        preset = dpt_encoder_map[encoder]

        model = cls.model(
            encoder_size=preset["encoder_size"],
            nclass=nclass,
            features=preset["features"],
            out_channels=preset["out_channels"],
            in_chans=in_channels,
            *args,
            **kwargs,
        )

        pretrain_path = Path(pretrain_dir)
        pth_path = download_pretrained_dinov2(
            encoder=encoder,
            weights=weights,
            pretrain_dir=pretrain_path,
        )
        if pth_path is not None:
            state_dict = torch.load(pth_path, map_location="cpu")
            model.backbone.load_state_dict(state_dict)
            print(f"Pretrained DINOv2 weights loaded from {pth_path}")

        return model
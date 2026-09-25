from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from terratorch.datasets.utils import HLSBands
from terratorch.tasks import PixelwiseRegressionTask

from geosave_engine.ml.model_chain import chain_step
from geosave_engine.ml.registry import register_model

_REPO_ID = "ibm-granite/granite-geospatial-biomass"
_CKPT_FILENAME = "biomass_model.ckpt"

# HLS S30 (Sentinel-2 derived): B02, B03, B04, B8A, B11, B12
_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]


@register_model("model", "ibm_granite_biomass")
class GraniteGeospatialBiomass(nn.Module):
    """IBM Granite Geospatial Biomass monolith: Prithvi Swin-B + UperNet + regression head.

    Estimates pixelwise above-ground biomass from HLS S30 surface reflectance,
    prepared per the checkpoint's preprocessing spec with the six bands ordered
    BLUE (B02), GREEN (B03), RED (B04), NIR_NARROW (B8A), SWIR_1 (B11), SWIR_2 (B12).

    Args:
        pretrained: Download and load checkpoint from HuggingFace hub.
        map_location: Device for checkpoint loading.

    Raises:
        RuntimeError: If checkpoint keys don't match the built model.
    """

    def __init__(
        self,
        pretrained: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        task = PixelwiseRegressionTask(
            # https://huggingface.co/ibm-granite/granite-geospatial-biomass/blob/main/config.yaml
            model_args={
                "decoder": "UperNetDecoder",
                "pretrained": False,
                "backbone": "prithvi_swin_B",
                "backbone_drop_path_rate": 0.3,
                "decoder_channels": 32,
                "in_channels": 6,
                "bands": _BANDS,
                "num_frames": 1,
                "head_dropout": 0.16194593880230534,
                "head_final_act": "torch.nn.ReLU",
                "head_learned_upscale_layers": 2,
            },
            model_factory="PrithviModelFactory",
            loss="mse",
            ignore_index=-1,
        )
        self.model = task.model

        if pretrained:
            ckpt_path = hf_hub_download(repo_id=_REPO_ID, filename=_CKPT_FILENAME)
            # Lightning checkpoint; weights_only=False required for non-tensor objects in state
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            state = {
                k.removeprefix("model."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.")
            }
            self.model.load_state_dict(state)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run biomass regression on a pre-normalized 6-band HLS S30 image.

        Args:
            image: (B, 6, H, W) tensor prepared for the checkpoint.

        Returns:
            (B, 1, H, W) biomass prediction tensor.
        """
        return self.model(image).output

    @chain_step(head=True)
    def forward_logits(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image)

import timm
import torch
import torch.nn as nn

from typing import cast
from timm.models.eva import Eva

from geosave_engine.ml.registry import register_model
from geosave_engine.ml.model_chain import Published, chain_step

# Sample each model's block depth at even quarters.
MODEL_SPECS: dict[str, dict] = {
    "vit_small_patch16_dinov3.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_small_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_small_plus_patch16_dinov3.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_small_plus_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_base_patch16_dinov3.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_base_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_large_patch16_dinov3.lvd1689m": {"out_indices": (5, 11, 17, 23)},
    "vit_large_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (5, 11, 17, 23)},
    "vit_large_patch16_dinov3.sat493m": {"out_indices": (5, 11, 17, 23)},
    "vit_large_patch16_dinov3_qkvb.sat493m": {"out_indices": (5, 11, 17, 23)},
    "vit_huge_plus_patch16_dinov3.lvd1689m": {"out_indices": (7, 15, 23, 31)},
    "vit_huge_plus_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (7, 15, 23, 31)},
    "vit_7b_patch16_dinov3.lvd1689m": {"out_indices": (9, 19, 29, 39)},
    "vit_7b_patch16_dinov3.sat493m": {"out_indices": (9, 19, 29, 39)},
}


@register_model("encoder", "dinov3")
class DINOv3(nn.Module):
    """A DINOv3 encoder for caller-prepared image tensors."""

    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]
    input_size: Published[int | tuple[int, int]]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        in_channels: int = 3,
        input_size: int | tuple[int, int] = 224,
        out_indices: list[int] | None = None,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_values: float | None = 1e-5,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        """Build a timm DINOv3 backbone.

        Args:
            model_name: timm model name. Must be a key of `MODEL_SPECS`.
            pretrained: load pretrained weights from timm hub.
            in_channels: input image channel count.
            input_size: input spatial size; tuple for non-square.
            drop_path_rate: stochastic depth rate.
            proj_drop_rate: dropout on attention output projection.
            attn_drop_rate: dropout on attention weights.
            init_values: initial layer-scale value; DINOv3 pretraining uses 1e-5.
                Only the init — a real checkpoint's trained values override this at
                load time, so it never affects `pretrained=True` weight compatibility.
            dynamic_img_size: interpolate positional embeddings for variable input sizes.
            dynamic_img_pad: pad input to nearest patch multiple when dynamic sizing.
        """
        super().__init__()
        if model_name not in MODEL_SPECS:
            raise ValueError(
                f"{model_name!r} not in MODEL_SPECS; must be one of {list(MODEL_SPECS)}"
            )

        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            img_size=input_size,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_values=init_values,
            # Shapes the register-token embedding, so an off-spec count breaks loading.
            num_reg_tokens=4,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.model = cast(Eva, model)

        self.out_indices: list[int] = out_indices or list(
            MODEL_SPECS[model_name]["out_indices"]
        )

        feature_info: list = self.model.feature_info
        self.pyramid_channels = [
            int(feature_info[i]["num_chs"]) for i in self.out_indices
        ]
        self.pyramid_strides = [  # [16, 16, 16, 16]
            int(feature_info[i]["reduction"]) for i in self.out_indices
        ]

        self.input_size = input_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Pool the ViT into one embedding per image.

        Args:
            image: (B, C, H, W) prepared pixels.

        Returns:
            (B, embed_dim) pooled embeddings.
        """
        return self.model(image)

    @chain_step()
    def forward_pyramid(self, image: torch.Tensor) -> tuple[list, list]:
        """Extract multi-scale intermediate features from the ViT.

        Args:
            image: (B, C, H, W) input tensor.

        Returns:
            (pyramid, prefix_tokens) — list of per-level feature maps, list
            of per-level prefix tokens.
        """
        pairs = self.model.forward_intermediates(
            image,
            indices=self.out_indices,
            intermediates_only=True,
            return_prefix_tokens=True,
            output_fmt="NCHW",
        )
        pyramid, prefix_tokens = map(list, zip(*pairs))
        return pyramid, prefix_tokens


import timm
import torch
import torch.nn as nn

from typing import cast, Literal
from timm.models.eva import Eva

from geosave_engine.ml.registry import register_model
from geosave_engine.ml.models.contract import chain_step

# out_indices are even quarters of each model's block depth:
#   depth=12 (S, S+, B) -> [2, 5, 8, 11]
#   depth=24 (L)        -> [5, 11, 17, 23]
#   depth=32 (H+)       -> [7, 15, 23, 31]
#   depth=40 (7B)       -> [9, 19, 29, 39]
TIMM_MODELS: dict[str, dict] = {
    'vit_small_patch16_dinov3.lvd1689m':            {'out_indices': (2, 5, 8, 11)},
    'vit_small_patch16_dinov3_qkvb.lvd1689m':       {'out_indices': (2, 5, 8, 11)},
    'vit_small_plus_patch16_dinov3.lvd1689m':       {'out_indices': (2, 5, 8, 11)},
    'vit_small_plus_patch16_dinov3_qkvb.lvd1689m':  {'out_indices': (2, 5, 8, 11)},
    'vit_base_patch16_dinov3.lvd1689m':             {'out_indices': (2, 5, 8, 11)},
    'vit_base_patch16_dinov3_qkvb.lvd1689m':        {'out_indices': (2, 5, 8, 11)},
    'vit_large_patch16_dinov3.lvd1689m':            {'out_indices': (5, 11, 17, 23)},
    'vit_large_patch16_dinov3_qkvb.lvd1689m':       {'out_indices': (5, 11, 17, 23)},
    'vit_large_patch16_dinov3.sat493m':             {'out_indices': (5, 11, 17, 23)},
    'vit_large_patch16_dinov3_qkvb.sat493m':        {'out_indices': (5, 11, 17, 23)},
    'vit_huge_plus_patch16_dinov3.lvd1689m':        {'out_indices': (7, 15, 23, 31)},
    'vit_huge_plus_patch16_dinov3_qkvb.lvd1689m':   {'out_indices': (7, 15, 23, 31)},
    'vit_7b_patch16_dinov3.lvd1689m':               {'out_indices': (9, 19, 29, 39)},
    'vit_7b_patch16_dinov3.sat493m':                {'out_indices': (9, 19, 29, 39)},
}

TimmModelName = Literal[
    'vit_small_patch16_dinov3.lvd1689m',
    'vit_small_patch16_dinov3_qkvb.lvd1689m',
    'vit_small_plus_patch16_dinov3.lvd1689m',
    'vit_small_plus_patch16_dinov3_qkvb.lvd1689m',
    'vit_base_patch16_dinov3.lvd1689m',
    'vit_base_patch16_dinov3_qkvb.lvd1689m',
    'vit_large_patch16_dinov3.lvd1689m',
    'vit_large_patch16_dinov3_qkvb.lvd1689m',
    'vit_large_patch16_dinov3.sat493m',
    'vit_large_patch16_dinov3_qkvb.sat493m',
    'vit_huge_plus_patch16_dinov3.lvd1689m',
    'vit_huge_plus_patch16_dinov3_qkvb.lvd1689m',
    'vit_7b_patch16_dinov3.lvd1689m',
    'vit_7b_patch16_dinov3.sat493m',
]


@register_model('encoder', 'dinov3')
class DINOv3(nn.Module):
    """A DINOv3 model with ImageNet normalization stats attached."""
    MODEL_NAMES = TIMM_MODELS

    def __init__(
        self,
        model_name: TimmModelName = 'vit_base_patch16_dinov3.lvd1689m',
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
        """Build a timm DINOv3 backbone with ImageNet normalization stats attached.

        `num_reg_tokens` not exposed on purpose — it's a real shape-affecting param
        (register-token embeddings are `(1, num_reg_tokens, embed_dim)`), and every
        released DINOv3 checkpoint uses 4; overriding it with `pretrained=True` would
        silently break weight loading, same reasoning as not exposing `embed_dim`/
        `depth` on `prithvi.py`'s `Prithvi`.

        Args:
            model_name: timm model name. Must be a key of `TIMM_MODELS`.
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

        Returns:
            timm Eva model with `img_mean` and `img_std` attributes set.
        """
        super().__init__()
        if model_name not in TIMM_MODELS:
            raise ValueError(
                f"{model_name!r} not in TIMM_MODELS; must be one of {list(TIMM_MODELS)}"
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
            num_reg_tokens=4,  # every released DINOv3 checkpoint uses 4, not user-overridable -- see docstring
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.model = cast(Eva, model)

        self.out_indices: list[int] = out_indices or list(TIMM_MODELS[model_name]['out_indices'])

        # Extract channel dimensions and spatial strides dynamically from the timm model
        feature_info: list = self.model.feature_info
        self.out_channels: list[int] = [
            int(feature_info[i]['num_chs']) for i in self.out_indices
        ]
        self.output_strides: list[int] = [ # [16, 16, 16, 16]
            int(feature_info[i]['reduction']) for i in self.out_indices
        ]

        self.input_size = input_size
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

    def forward(
            self,
            x,
            rope: torch.Tensor | None = None,
            attn_mask: torch.Tensor | None = None,
            is_causal: bool = False,
    ):
        """Forward pass for the attention module.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            rope: Rotary position embeddings tensor for position-aware attention
            attn_mask: Optional attention mask to apply during attention computation
            is_causal: If True, use causal (autoregressive) masking

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.model(x, rope=rope, attn_mask=attn_mask, is_causal=is_causal)

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
            output_fmt='NCHW',
        )
        pyramid, prefix_tokens = map(list, zip(*pairs))
        return pyramid, prefix_tokens


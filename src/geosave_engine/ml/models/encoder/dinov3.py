
import timm
import torch
import torch.nn as nn

from typing import cast
from timm.models.eva import Eva

from geosave_engine.ml.models.contract import ModelContext, model_context

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


class DINOv3(nn.Module):
    """A DINOv3 model with ImageNet normalization stats attached."""

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_dinov3.lvd1689m',
        pretrained: bool = True,
        in_channels: int = 3,
        img_size: int | tuple[int, int] = 224,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_values: float | None = 1e-5,
        num_reg_tokens: int = 4,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        """Build a timm DINOv3 backbone with ImageNet normalization stats attached.

        Args:
            model_name: timm model name. Must be a key of `TIMM_MODELS`.
            pretrained: load pretrained weights from timm hub.
            in_channels: input image channel count.
            img_size: input spatial size; tuple for non-square.
            drop_path_rate: stochastic depth rate.
            proj_drop_rate: dropout on attention output projection.
            attn_drop_rate: dropout on attention weights.
            init_values: initial layer-scale value; DINOv3 pretraining uses 1e-5.
            num_reg_tokens: register tokens; DINOv3 pretraining uses 4.
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
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_values=init_values,
            num_reg_tokens=num_reg_tokens,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.model = cast(Eva, model)

        self.out_indices: list[int] = list(TIMM_MODELS[model_name]['out_indices'])

        # Extract channel dimensions and spatial strides dynamically from the timm model
        feature_info: list = self.model.feature_info
        self.out_channels: list[int] = [
            int(feature_info[i]['num_chs']) for i in self.out_indices
        ]
        self.output_strides: list[int] = [ # [16, 16, 16, 16]
            int(feature_info[i]['reduction']) for i in self.out_indices
        ]

        self.input_size = img_size
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

    @model_context(inputs=(['image'], ['pyramid', 'prefix_tokens']))
    def forward_pyramid(self, ctx: ModelContext) -> ModelContext:
        """Extract multi-scale intermediate features from the ViT.

        Reads ctx.inputs['image'] (B, C, H, W). Writes 'pyramid' (list of
        per-level feature maps) and 'prefix_tokens' (list of per-level prefix
        tokens) to ctx.inputs.

        Args:
            ctx: ModelContext with ctx.inputs['image'] as (B, C, H, W).

        Returns:
            ModelContext with ctx.inputs = {'pyramid': [...], 'prefix_tokens': [...]}.
        """
        pairs = self.model.forward_intermediates(
            ctx.inputs['image'],
            indices=self.out_indices,
            intermediates_only=True,
            return_prefix_tokens=True,
            output_fmt='NCHW',
        )
        features, prefix_tokens = map(list, zip(*pairs))
        return ModelContext(
            inputs={'pyramid': features, 'prefix_tokens': prefix_tokens},
            sample_meta=ctx.sample_meta,
            metadata=ctx.metadata,
        )


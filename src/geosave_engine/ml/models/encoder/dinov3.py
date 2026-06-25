from typing import cast

import timm
from timm.models.eva import Eva


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


def build_dinov3_timm(
    model_name: str,
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
) -> Eva:
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
    model = cast(Eva, model)
    setattr(model, 'img_mean', [0.485, 0.456, 0.406])
    setattr(model, 'img_std', [0.229, 0.224, 0.225])
    return model

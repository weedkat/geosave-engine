from __future__ import annotations

import torch
import torch.nn as nn

from typing import cast
from terratorch.models.backbones.prithvi_mae import PrithviViT
from terratorch.registry import BACKBONE_REGISTRY

from geosave_engine.ml.registry import register_model
from geosave_engine.ml.models.contract import model_context

# From terratorch.models.backbones.prithvi_vit — HLS S30 DN scale (reflectance x 10000).
# terratorch never attaches these to the model itself (PrithviViT.__init__ takes no
# mean/std param, swallows and discards them via **kwargs), so they're copied here.
_V1_MEAN = [775.0, 1081.0, 1229.0, 2497.0, 2204.0, 1611.0]
_V1_STD = [1282.0, 1270.0, 1399.0, 1368.0, 1292.0, 1155.0]
_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
_V1_VARIANTS = frozenset({'prithvi_eo_v1_100'})


@register_model('encoder', 'prithvi')
class Prithvi(nn.Module):
    """A Prithvi-EO ViT model with band normalization stats attached.

    Non-temporal-location variants only — `forward_pyramid` takes just `image`.
    For a `_tl` variant (real time/location conditioning), use
    `PrithviTemporalLocation` instead. Kept as two separate classes rather than
    one class with two `@model_context` methods on `forward_pyramid`: both
    methods would only ever depend on externally-supplied keys, so
    `ContextChain` can't tell them apart by DAG depth — they'd always land in
    the same generation and hit the "ambiguous" error. One method per class
    sidesteps that: which `forward_pyramid` runs is decided by Python's normal
    class/MRO lookup at `model_name` selection time, not by graph resolution.
    """

    # out_indices are even quarters of each model's block depth (same convention
    # as dinov3.py): depth=12 -> [2,5,8,11], depth=24 -> [5,11,17,23], depth=32 -> [7,15,23,31].
    # prithvi_eo_tiny excluded: no public checkpoint, architecture-only debug variant.
    MODEL_NAMES: dict[str, dict] = {
        'prithvi_eo_v1_100': {'out_indices': (2, 5, 8, 11)},
        'prithvi_eo_v2_300': {'out_indices': (5, 11, 17, 23)},
        'prithvi_eo_v2_600': {'out_indices': (7, 15, 23, 31)},
    }

    def __init__(
        self,
        model_name: str = 'prithvi_eo_v2_300',
        pretrained: bool = False,
        in_channels: int = 6,
        input_size: int | tuple[int, int] = 224,
        num_frames: int = 1,
        drop_path_rate: float = 0.0,
        out_indices: list[int] | None = None,
        ckpt_path: str | None = None,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
    ):
        """Build a terratorch Prithvi-EO backbone with band normalization stats attached.

        No architecture-identity params (`embed_dim`/`depth`/`num_heads`/`mlp_ratio`/
        `patch_size`/`norm_layer`/`coords_encoding`) exposed here on purpose — those
        are exactly what `model_name` already picks, and overriding one independently
        of `model_name` silently breaks pretrained-weight shape compatibility. Same
        principle as `dinov3.py` not exposing `embed_dim`/`depth`: if a genuinely
        different architecture is wanted, that's a different `model_name` (or a real
        from-scratch build), not a kwarg on this class.

        Args:
            model_name: terratorch backbone registry name. Must be a key of `MODEL_NAMES`.
            pretrained: load pretrained weights from HuggingFace hub (or `ckpt_path`).
            in_channels: input channel count. `6` (default) matches the checkpoint's
                pretrained HLS S30 bands (BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2)
                — real transferred weights. Any other count gets a patch-embed conv of
                that width with randomly initialized weights (terratorch can only
                weight-transfer by matching band identity, and a plain channel count
                carries none — same as handing timm's dinov3 an off-spec in_channels).
            input_size: input spatial size in pixels; tuple for non-square. Only
                sizes `model_name`'s patch size evenly divides use every pixel —
                anything else gets silently border-cropped by terratorch's PatchEmbed.
                Doesn't have to match the actual tensor size passed to `forward`
                later: the positional embedding is re-interpolated to whatever shape
                shows up at call time, so this is just the size used to size that
                buffer (and, at `pretrained=True`, the size the checkpoint itself used).
            num_frames: number of timesteps in the input (temporal stacking); `1` for
                single-timestep imagery.
            drop_path_rate: stochastic depth rate.
            out_indices: which of the model's blocks to return features from. `None`
                uses this `model_name`'s default (see `MODEL_NAMES`).
            ckpt_path: local checkpoint path; `None` fetches the public one from HF Hub.
            vpt: use Visual Prompt Tuning (freeze backbone, learn small prompt tokens
                prepended per block) instead of full fine-tuning.
            vpt_n_tokens: prompt tokens per block. Required if `vpt` is True.
            vpt_dropout: dropout on VPT prompt tokens.

        Returns:
            terratorch PrithviViT model with `img_mean` and `img_std` attributes set.

        Raises:
            ValueError: `model_name` not in `MODEL_NAMES`.
        """
        super().__init__()
        model_names = type(self).MODEL_NAMES
        if model_name not in model_names:
            raise ValueError(
                f"{model_name!r} not in {type(self).__name__}.MODEL_NAMES; must be one of {list(model_names)}"
            )

        self.out_indices = out_indices if out_indices is not None else list(model_names[model_name]['out_indices'])
        # bands=None -> terratorch assumes the pretrained 6-band HLS S30 order itself
        # (prithvi_vit.py: "model_bands is None -> model_bands = pretrained_bands").
        # Any other in_channels needs an explicit same-length list; plain ints carry
        # no band identity, so terratorch can't weight-transfer them — random-init.
        bands = None if in_channels == 6 else list(range(in_channels))

        model = BACKBONE_REGISTRY.build(
            model_name,
            pretrained=pretrained,
            bands=bands,
            img_size=input_size,
            num_frames=num_frames,
            drop_path=drop_path_rate,
            out_indices=self.out_indices,
            ckpt_path=ckpt_path,
            vpt=vpt,
            vpt_n_tokens=vpt_n_tokens,
            vpt_dropout=vpt_dropout,
        )
        self.model = cast(PrithviViT, model)

        # ViT: same embed_dim/spatial stride at every block, unlike a CNN's per-stage
        # channel growth + downsampling — still indexed per out_index for robustness.
        self.out_channels: list[int] = [self.model.out_channels[i] for i in self.out_indices]
        model_patch_size = self.model.patch_embed.patch_size  # (t, h, w)
        self.output_strides: list[int] = [model_patch_size[-1]] * len(self.out_indices)

        self.input_size = input_size
        mean, std = (_V1_MEAN, _V1_STD) if model_name in _V1_VARIANTS else (_V2_MEAN, _V2_STD)
        self.img_mean = mean
        self.img_std = std

    def forward(
        self,
        x: torch.Tensor,
        temporal_coords: torch.Tensor | None = None,
        location_coords: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Forward pass for the backbone — raw passthrough to the wrapped model.

        Args:
            x: Input tensor of shape (batch_size, in_chans, height, width).
            temporal_coords: (B, num_frames, 2) float32 — (year, day-of-year) per
                frame, day-of-year 0-indexed (Jan 1st = 0), real calendar values,
                not normalized. Only has an effect on a `coords_encoding=["time", ...]`
                model (`PrithviTL`'s variants); ignored otherwise.
            location_coords: (B, 2) float32 — (lat, lon) in degrees, real values.
                Only has an effect on a `coords_encoding=[..., "location"]` model
                (`PrithviTL`'s variants); ignored otherwise.

        Returns:
            List of per-`out_indices` token tensors, each (B, 1 + N_patches, embed_dim)
            — CLS token at index 0.

        Examples:
            >>> temporal_coords = torch.tensor([[[2024.0, 45.0]]])  # Feb 15 2024, single frame
            >>> location_coords = torch.tensor([[52.5, 13.4]])  # Berlin
            >>> features = enc.forward(image, temporal_coords, location_coords)
        """
        # (B, 1+N_patches, embed_dim) per block, full depth -- see forward_pyramid for the
        # out_indices-sliced, spatially-reshaped version used by the actual pipeline.
        return self.model(x, temporal_coords=temporal_coords, location_coords=location_coords)

    @model_context()
    def forward_pyramid(self, image: torch.Tensor) -> tuple[list, list]:
        """Extract multi-scale intermediate features from the ViT.

        Calls `forward_features` directly (not `self.model(image)`/`forward` — that's
        the MAE-pretraining pass, masks most patches, wrong output shape entirely).
        `forward_features` always returns every block's output (can't skip blocks,
        each depends on the last), so the `out_indices` slice happens here, on our
        side, not left to terratorch's own monkeypatched `model.forward`.

        Args:
            image: (B, C, H, W) input tensor.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, C, H, W) feature
            maps, list of per-level (B, 1, C) CLS tokens.
        """
        features = self.model.forward_features(image)  # list[depth] of (B, 1+N_patches, embed_dim), CLS at idx 0
        features = [features[i] for i in self.out_indices]  # list[len(out_indices)] of (B, 1+N_patches, embed_dim)
        prefix_tokens = [f[:, :1, :] for f in features]  # list[len(out_indices)] of (B, 1, embed_dim) -- CLS only
        pyramid = self.model.prepare_features_for_image_model(features)  # list of (B, embed_dim, H/patch, W/patch)
        return pyramid, prefix_tokens


@register_model('encoder', 'prithvi_tl')
class PrithviTL(Prithvi):
    """A Prithvi-EO ViT model conditioned on real time/location, HLS band stats attached.

    The `_tl` variants only — `forward_pyramid` here takes `temporal_coords`/
    `location_coords` too, and actually uses them (`coords_encoding=["time",
    "location"]` is baked into every one of `MODEL_NAMES` below). Everything else
    (`__init__`, `forward`) is shared with `Prithvi` — see its docstring for why
    this is a separate class rather than a second method on the same one.
    """

    MODEL_NAMES: dict[str, dict] = {
        'prithvi_eo_v2_tiny_tl': {'out_indices': (2, 5, 8, 11)},
        'prithvi_eo_v2_100_tl':  {'out_indices': (2, 5, 8, 11)},
        'prithvi_eo_v2_300_tl':  {'out_indices': (5, 11, 17, 23)},
        'prithvi_eo_v2_600_tl':  {'out_indices': (7, 15, 23, 31)},
    }

    def __init__(
        self,
        model_name: str = 'prithvi_eo_v2_300_tl',
        pretrained: bool = False,
        in_channels: int = 6,
        input_size: int | tuple[int, int] = 224,
        num_frames: int = 1,
        drop_path_rate: float = 0.0,
        out_indices: list[int] | None = None,
        ckpt_path: str | None = None,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
    ):
        """Same as `Prithvi.__init__` (see there for full Args), just defaulting
        `model_name` to a `_tl` variant."""
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            input_size=input_size,
            num_frames=num_frames,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            ckpt_path=ckpt_path,
            vpt=vpt,
            vpt_n_tokens=vpt_n_tokens,
            vpt_dropout=vpt_dropout,
        )

    @model_context()
    def forward_pyramid(
        self,
        image: torch.Tensor,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
    ) -> tuple[list, list]:
        """Extract multi-scale intermediate features, conditioned on time/location.

        Args:
            image: (B, C, H, W) input tensor.
            temporal_coords: (B, num_frames, 2) float32 — (year, day-of-year) per
                frame, day-of-year 0-indexed (Jan 1st = 0), real calendar values,
                not normalized.
            location_coords: (B, 2) float32 — (lat, lon) in degrees, real values.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, C, H, W) feature
            maps, list of per-level (B, 1, C) CLS tokens.

        Examples:
            >>> ctx = {
            ...     'image': image,  # (B, 6, H, W)
            ...     'temporal_coords': torch.tensor([[[2024.0, 45.0]]]),  # Feb 15 2024, single frame
            ...     'location_coords': torch.tensor([[52.5, 13.4]]),  # Berlin
            ... }
            >>> out = PrithviTL().forward_pyramid(ctx)
        """
        features = self.model.forward_features(  # list[depth] of (B, 1+N_patches, embed_dim), CLS at idx 0
            image, temporal_coords=temporal_coords, location_coords=location_coords
        )
        features = [features[i] for i in self.out_indices]  # list[len(out_indices)] of (B, 1+N_patches, embed_dim)
        prefix_tokens = [f[:, :1, :] for f in features]  # list[len(out_indices)] of (B, 1, embed_dim) -- CLS only
        pyramid = self.model.prepare_features_for_image_model(features)  # list of (B, embed_dim, H/patch, W/patch)
        return pyramid, prefix_tokens


if __name__ == "__main__":
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
    
    print("Models registered in terratorch:", list(TERRATORCH_BACKBONE_REGISTRY._registry))
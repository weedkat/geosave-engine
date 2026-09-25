from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import xarray as xr
from terratorch.models.backbones.prithvi_mae import PrithviViT
from terratorch.registry import BACKBONE_REGISTRY

from geosave_engine.ml.registry import register_model
from geosave_engine.ml.models.contract import Published, chain_step

from ..context.time import time_labels


class _Prithvi(nn.Module):
    """Shared Prithvi construction and geodata tensor layout."""

    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]
    input_size: Published[int | tuple[int, int]]

    # Sample each transformer depth at even quarters.
    MODEL_SPECS: dict[str, dict] = {
        "prithvi_eo_v1_100": {"out_indices": (2, 5, 8, 11)},
        "prithvi_eo_v2_300": {"out_indices": (5, 11, 17, 23)},
        "prithvi_eo_v2_600": {"out_indices": (7, 15, 23, 31)},
    }

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300",
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
        """Build a terratorch Prithvi-EO backbone.

        Args:
            model_name: terratorch backbone registry name. Must be a key of `MODEL_SPECS`.
            pretrained: load pretrained weights from HuggingFace hub (or `ckpt_path`).
            in_channels: input channel count. `6` matches the checkpoint's pretrained
                HLS S30 bands (BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2) and
                transfers their weights; any other count gets a randomly initialized
                patch-embed conv of that width.
            input_size: size the positional embedding buffer is built for, in pixels;
                tuple for non-square. Sizes `model_name`'s patch size does not divide
                are border-cropped. Call-time tensors may differ — the embedding is
                re-interpolated to whatever shape arrives.
            num_frames: number of timesteps in the input (temporal stacking); `1` for
                single-timestep imagery.
            drop_path_rate: stochastic depth rate.
            out_indices: which of the model's blocks to return features from. `None`
                uses this `model_name`'s default (see `MODEL_SPECS`).
            ckpt_path: local checkpoint path; `None` fetches the public one from HF Hub.
            vpt: use Visual Prompt Tuning (freeze backbone, learn small prompt tokens
                prepended per block) instead of full fine-tuning.
            vpt_n_tokens: prompt tokens per block. Required if `vpt` is True.
            vpt_dropout: dropout on VPT prompt tokens.

        Raises:
            ValueError: `model_name` not in `MODEL_SPECS`.
        """
        super().__init__()
        model_specs = type(self).MODEL_SPECS
        if model_name not in model_specs:
            raise ValueError(
                f"{model_name!r} not in {type(self).__name__}.MODEL_SPECS; must be one of {list(model_specs)}"
            )

        self.out_indices: list[int] = (
            out_indices
            if out_indices is not None
            else list(model_specs[model_name]["out_indices"])
        )
        # TerraTorch uses its pretrained band order when bands is None.
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
        self.model: PrithviViT = cast(PrithviViT, model)

        self.pyramid_channels = [self.model.out_channels[i] for i in self.out_indices]
        model_patch_size = self.model.patch_embed.patch_size  # (t, h, w)
        self.pyramid_strides = [model_patch_size[-1]] * len(self.out_indices)

        self.input_size = input_size

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Run the backbone on geodata-ordered image tensors.

        Args:
            image: (B, C, H, W) or (B, T, C, H, W) prepared pixels.

        Returns:
            Per-block token tensors from the wrapped backbone.
        """
        if image.ndim == 5:
            image = image.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
        return self.model(image, temporal_coords=None, location_coords=None)


@register_model("encoder", "prithvi")
class Prithvi(_Prithvi):
    """Prithvi encoder accepting prepared images in geodata axis order.

    Inputs have shape (B, C, H, W) or (B, T, C, H, W). Constructor options
    configure the shared Prithvi backbone.
    """

    @chain_step()
    def forward_pyramid(self, image: torch.Tensor) -> tuple[list, list]:
        """Extract multi-scale intermediate features from the ViT.

        Args:
            image: (B, C, H, W) or (B, T, C, H, W) prepared pixels.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, C, H, W) feature
            maps, list of per-level (B, 1, C) CLS tokens.
        """
        if image.ndim == 5:
            image = image.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
        features = self.model.forward_features(
            image
        )  # list[depth] of (B, 1+N_patches, embed_dim), CLS at idx 0
        features = [
            features[i] for i in self.out_indices
        ]  # list[len(out_indices)] of (B, 1+N_patches, embed_dim)
        prefix_tokens = [
            f[:, :1, :] for f in features
        ]  # list[len(out_indices)] of (B, 1, embed_dim) -- CLS only
        pyramid = self.model.prepare_features_for_image_model(
            features
        )  # list of (B, embed_dim, H/patch, W/patch)
        return pyramid, prefix_tokens


@register_model("encoder", "prithvi_tl")
class PrithviTL(_Prithvi):
    """Prithvi encoder conditioned on acquisition time and geographic location.

    `model_context` extracts raw coordinates from one geodata sample.
    `forward_pyramid` accepts the tensors after DataLoader collation.
    """

    MODEL_SPECS: dict[str, dict] = {
        "prithvi_eo_v2_tiny_tl": {"out_indices": (2, 5, 8, 11)},
        "prithvi_eo_v2_100_tl": {"out_indices": (2, 5, 8, 11)},
        "prithvi_eo_v2_300_tl": {"out_indices": (5, 11, 17, 23)},
        "prithvi_eo_v2_600_tl": {"out_indices": (7, 15, 23, 31)},
    }

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300_tl",
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
        """Same backbone options as `Prithvi`, defaulting
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

    @staticmethod
    def model_context(data: xr.Dataset | xr.DataArray) -> dict[str, torch.Tensor]:
        """Read this raster's time labels and geographic centre for Prithvi.

        Args:
            data: One sample with datetime time labels and a regular grid.
                Labels are encoded directly, including labels of temporal buckets.

        Returns:
            Unbatched float32 tensors: `temporal_coords` shaped (T, 2), holding
            year and zero-indexed day-of-year; `location_coords` shaped (2,),
            holding latitude and longitude in degrees.

        Raises:
            TypeError: Input is not a Dataset or DataArray.
            ValueError: Time labels or a regular georeferenced grid are missing
                or invalid.
        """
        times = time_labels(data)
        longitude, latitude = data.gs.anchor.geographic_centroid
        return {
            "temporal_coords": torch.tensor(
                [(when.year, when.timetuple().tm_yday - 1) for when in times],
                dtype=torch.float32,
            ),
            "location_coords": torch.tensor([latitude, longitude], dtype=torch.float32),
        }

    @chain_step()
    def forward_pyramid(
        self,
        image: torch.Tensor,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
    ) -> tuple[list, list]:
        """Extract multi-scale intermediate features, conditioned on time/location.

        Args:
            image: (B, C, H, W) or (B, T, C, H, W) prepared pixels.
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
            >>> out = PrithviTL().forward_pyramid(**ctx)
        """
        if image.ndim == 5:
            image = image.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
        features = self.model.forward_features(  # list[depth] of (B, 1+N_patches, embed_dim), CLS at idx 0 # type: ignore
            image, temporal_coords=temporal_coords, location_coords=location_coords
        )
        features = [features[i] for i in self.out_indices]  # pyright: ignore[reportGeneralTypeIssues] # list[len(out_indices)] of (B, 1+N_patches, embed_dim)
        prefix_tokens = [
            f[:, :1, :] for f in features
        ]  # list[len(out_indices)] of (B, 1, embed_dim) -- CLS only
        pyramid = self.model.prepare_features_for_image_model(features)  # type: ignore # list of (B, embed_dim, H/patch, W/patch)
        return pyramid, prefix_tokens

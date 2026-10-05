from __future__ import annotations

import torch
import pandas as pd
import torch.nn as nn
from huggingface_hub import hf_hub_download
from terratorch.models.backbones.clay_v15.model import Encoder

from geosave_engine.model.registry import register_model
from geosave_engine.model.chain import Published, chain_step

from geosave_engine.model.encoder.time import time_labels
from geosave_engine.model.encoder.context import geographic_center

# Only the large variant has a published checkpoint.
MODEL_SOURCE: dict[str, dict[str, str]] = {
    "clay_v15_large": {
        "repo_id": "made-with-clay/Clay",
        "filename": "v1.5/clay-v1.5.ckpt",
    },
}

# Extract only encoder weights from the Lightning checkpoint.
_STATE_DICT_ENCODER_PREFIX: str = "model.encoder."

# Build only the encoder; TerraTorch MAE factories also construct a teacher.
MODEL_SPECS: dict[str, dict] = {
    "clay_v15_tiny": {
        "dim": 192,
        "depth": 6,
        "heads": 4,
        "dim_head": 48,
        "mlp_ratio": 2,
        "patch_size": 8,
    },
    "clay_v15_small": {
        "dim": 384,
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
        "patch_size": 8,
    },
    "clay_v15_base": {
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "patch_size": 8,
    },
    "clay_v15_large": {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "patch_size": 8,
    },
}


_WEEK_PERIOD = 52.0
_HOUR_PERIOD = 24.0


def _normalize_time(time: torch.Tensor) -> torch.Tensor:
    """Raw (iso_week, hour) -> Clay's own sin/cos week+hour position encoding.

    Args:
        time: (B, 2) float, (iso_week [1-52], hour [0-23]).

    Returns:
        (B, 4) float — (sin(week), cos(week), sin(hour), cos(hour)).
    """
    week_angle = time[:, 0] * (2 * torch.pi / _WEEK_PERIOD)
    hour_angle = time[:, 1] * (2 * torch.pi / _HOUR_PERIOD)
    return torch.stack(
        [
            torch.sin(week_angle),
            torch.cos(week_angle),
            torch.sin(hour_angle),
            torch.cos(hour_angle),
        ],
        dim=-1,
    )


def _normalize_latlon(latlon: torch.Tensor) -> torch.Tensor:
    """Raw (lat, lon) degrees -> Clay's own sin/cos location encoding.

    Args:
        latlon: (B, 2) float, (lat, lon) in degrees.

    Returns:
        (B, 4) float — (sin(lat), cos(lat), sin(lon), cos(lon)), radians internally.
    """
    radians = latlon * (torch.pi / 180.0)
    lat, lon = radians[:, 0], radians[:, 1]
    return torch.stack(
        [torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon)], dim=-1
    )


def default_out_indices(depth: int) -> list[int]:
    """Even quarters of `depth` (same convention as dinov3.py/prithvi.py).

    Args:
        depth: transformer block count.

    Returns:
        Up to 4 block indices, e.g. depth=24 -> [5, 11, 17, 23].
    """
    return sorted({max(0, (depth * quarter) // 4 - 1) for quarter in (1, 2, 3, 4)})


def build_clay(
    model_name: str,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> Encoder:
    """Build a Clay v1.5 `Encoder` for one architecture size, optionally with real weights.

    Args:
        model_name: key of `MODEL_SPECS`.
        pretrained: load real Clay v1.5 weights onto the built `Encoder`, from
            `checkpoint_path` if given, else HF Hub via `MODEL_SOURCE`. Only
            `model_name`s in `MODEL_SOURCE` have a published checkpoint.
        checkpoint_path: local ckpt path (same format as the HF one — a
            PyTorch Lightning checkpoint with a `state_dict` key). `None`
            fetches from HF Hub. Ignored if `pretrained` is `False`.

    Returns:
        `encoder` -- built `Encoder`, real weights loaded if `pretrained`.
        No `out_indices` here -- that's a `forward_pyramid` concern (which
        blocks to hook), not part of building the architecture itself.

    Raises:
        ValueError: `model_name` not in `MODEL_SPECS`, or `pretrained` is
            `True` without a local checkpoint and `model_name` has no entry
            in `MODEL_SOURCE`.
    """
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"{model_name!r} not in MODEL_SPECS; must be one of {list(MODEL_SPECS)}"
        )
    if pretrained and checkpoint_path is None and model_name not in MODEL_SOURCE:
        raise ValueError(
            f"{model_name!r} has no published checkpoint; pretrained=True only "
            f"works for {list(MODEL_SOURCE)}"
        )

    spec = MODEL_SPECS[model_name]
    encoder = Encoder(
        mask_ratio=0.0,
        patch_size=spec["patch_size"],
        shuffle=False,
        dim=spec["dim"],
        depth=spec["depth"],
        heads=spec["heads"],
        dim_head=spec["dim_head"],
        mlp_ratio=spec["mlp_ratio"],
    )

    if pretrained:
        path = checkpoint_path
        if path is None:
            source = MODEL_SOURCE[model_name]
            path = hf_hub_download(
                repo_id=source["repo_id"], filename=source["filename"]
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = {
            name.removeprefix(_STATE_DICT_ENCODER_PREFIX): param
            for name, param in ckpt["state_dict"].items()
            if name.startswith(_STATE_DICT_ENCODER_PREFIX)
        }
        encoder.load_state_dict(state_dict)

    return encoder


def time(row: pd.Series, *, raster: str = "image") -> torch.Tensor:
    """Encode one acquisition timestamp as ISO week and hour.

    Args:
        row: Sample reference row with acquisition metadata.
        raster: Prepared raster supplying the frame.

    Returns:
        Float32 tensor shaped (2,).

    Raises:
        ValueError: More than one frame was selected.
    """
    times = time_labels(row, raster=raster)
    if len(times) != 1:
        raise ValueError("Clay requires one time label; select a single frame first")
    when = times[0]
    return torch.tensor([when.isocalendar().week, when.hour], dtype=torch.float32)


def latlon(row: pd.Series) -> torch.Tensor:
    """Return the sample grid centre as a float32 latitude/longitude tensor."""
    return torch.tensor(geographic_center(row), dtype=torch.float32)


def model_context(row: pd.Series, *, raster: str = "image") -> dict[str, torch.Tensor]:
    """Encode sample metadata for Clay's time/location inputs.

    Args:
        row: Reference row describing the actual input window.
        raster: Prepared raster supplying the frame.

    Returns:
        ISO week/hour and latitude/longitude float32 tensors shaped (2,).
        Wavelengths and GSD retain the encoder's constructor defaults.
    """
    return {"time": time(row, raster=raster), "latlon": latlon(row)}


@register_model("encoder", "clay")
class Clay(nn.Module):
    """Clay v1.5 encoder conditioned on band wavelengths and geographic context.

    Supply prepared image tensors, ordered wavelengths in micrometres, and GSD
    in metres. `model_context` reads time and location from one sample reference row.
    """

    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]

    waves: torch.Tensor
    gsd: torch.Tensor

    def __init__(
        self,
        *,
        model_name: str = "clay_v15_large",
        in_channels: int,
        input_size: int | tuple[int, int] = 224,
        waves: list[float],
        gsd: float,
        pretrained: bool = False,
        checkpoint_path: str | None = None,
        out_indices: list[int] | None = None,
    ):
        """Build a Clay v1.5 encoder, wavelength-conditioned on caller-supplied band stats.

        Args:
            model_name: must be a key of module-level `MODEL_SPECS`. Only
                `'clay_v15_large'` has a published checkpoint.
            in_channels: Number of input bands, matching `waves`.
            input_size: input spatial size in pixels; `int` or `(h, w)`. Must
                be square (`h == w`) — Clay's patch grid assumes it — and
                evenly divisible by `model_name`'s patch size.
            waves: per-band wavelength in µm, length `in_channels`, ordered
                like the input tensor's channels.
            gsd: ground sample distance in meters, this instance's default
                (per-call override via `forward`'s own `gsd` param still works).
            pretrained: load real Clay v1.5 weights (from `checkpoint_path`, or
                HF Hub if `None`) onto the built encoder. Only `model_name`s in
                `MODEL_SOURCE` have a published checkpoint.
            checkpoint_path: optional local ckpt path; ``None`` fetches from HF Hub.
                Ignored if `pretrained` is ``False``.
            out_indices: which transformer blocks to return features from. `None`
                picks even quarters of `model_name`'s depth (see `build_clay`).

        Raises:
            ValueError: `model_name` not in `MODEL_SPECS`; `waves` doesn't have
                exactly `in_channels` values; `input_size` isn't square or
                isn't evenly divisible by `model_name`'s patch size; or
                `pretrained` is `True` with no published checkpoint for
                `model_name` (see `MODEL_SOURCE`).
        """
        super().__init__()
        if len(waves) != in_channels:
            raise ValueError(
                f"waves must have {in_channels} values (in_channels), got {len(waves)}"
            )

        height, width = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        if height != width:
            raise ValueError(f"input_size must be square, got {height}x{width}")

        self.encoder = build_clay(
            model_name, pretrained=pretrained, checkpoint_path=checkpoint_path
        )

        dim: int = self.encoder.dim
        patch_size: int = self.encoder.patch_size
        depth: int = len(self.encoder.transformer.layers)

        if height % patch_size != 0:
            raise ValueError(
                f"input_size {height} not evenly divisible by {model_name}'s patch_size {patch_size}"
            )
        self.grid = height // patch_size

        self.out_indices = (
            list(out_indices) if out_indices is not None else default_out_indices(depth)
        )

        self.register_buffer("waves", torch.tensor(waves, dtype=torch.float32))
        self.register_buffer("gsd", torch.tensor(float(gsd)))

        self.pyramid_channels = [dim] * len(self.out_indices)
        self.pyramid_strides = [patch_size] * len(self.out_indices)

    def forward(
        self,
        image: torch.Tensor,
        time: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
        gsd: torch.Tensor | None = None,
        waves: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass for the backbone.

        Args:
            image: (B, C, H, W) input tensor, C == this instance's `in_channels`.
            time: (B, 2) float32 — raw `(iso_week, hour)`, normalized internally
                (`_normalize_time`) into Clay's own sin/cos position encoding.
                `None` fills zeros (no time signal fed to the model).
            latlon: (B, 2) float32 — raw `(lat, lon)` in degrees, normalized
                internally (`_normalize_latlon`) into Clay's own sin/cos
                position encoding. `None` fills zeros (no location signal fed
                to the model).
            gsd: scalar float32 — ground sample distance in meters. `None` uses
                `self.gsd` (this instance's constructor-time `gsd`).
            waves: (C,) float32 — per-band wavelength in µm, ordered like
                `image`'s channels. `None` uses `self.waves` (this instance's
                constructor-time `waves`).

        Returns:
            Whatever `Encoder.forward` returns natively: (encoded_unmasked_patches,
            unmasked_indices, masked_indices, masked_matrix). With `mask_ratio=0.0`
            (fixed at construction), nothing is actually masked, so
            `encoded_unmasked_patches` is (B, 1+L, D) covering every patch, CLS at
            index 0 — same shape convention `forward_pyramid` builds its pyramid from.

        Examples:
            >>> clay = Clay(in_channels=4, waves=[0.49, 0.56, 0.66, 0.84], gsd=10.0)
            >>> image = torch.randn(1, 4, 224, 224)
            >>> time = torch.tensor([[7.0, 14.0]])  # ISO week 7, 2pm
            >>> latlon = torch.tensor([[52.5, 13.4]])  # Berlin
            >>> encoded, *_ = clay.forward(image, time=time, latlon=latlon)
        """
        batch_size = image.shape[0]
        time = (
            _normalize_time(time)
            if time is not None
            else torch.zeros(batch_size, 4, device=image.device, dtype=image.dtype)
        )
        latlon = (
            _normalize_latlon(latlon)
            if latlon is not None
            else torch.zeros(batch_size, 4, device=image.device, dtype=image.dtype)
        )
        if gsd is None:
            gsd = self.gsd
        if waves is None:
            waves = self.waves

        datacube = {
            "pixels": image,  # (B, C, H, W)
            "time": time,  # (B, 4)
            "latlon": latlon,  # (B, 4)
            "gsd": gsd,  # scalar
            "waves": waves,  # (C,)
        }
        return self.encoder(datacube)

    def _tokens_to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) patch tokens (no CLS) -> (B, D, H, W), `self.grid` precomputed at construction.

        Args:
            x: (B, L, D) patch tokens, CLS already stripped.

        Returns:
            (B, D, H, W) spatial feature map.
        """
        b, _, dim = x.shape  # (B, L, D)
        return x.transpose(1, 2).reshape(
            b, dim, self.grid, self.grid
        )  # (B, L, D) -> (B, D, L) -> (B, D, H, W)

    @chain_step()
    def forward_pyramid(
        self,
        image: torch.Tensor,
        time: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Extract multi-scale intermediate features from the ViT.

        Args:
            image: (B, C, H, W) input tensor, C == this instance's `in_channels`.
            time: (B, 2) float32 raw `(iso_week, hour)`, or None for no time signal.
            latlon: (B, 2) float32 raw `(lat, lon)` degrees, or None for no location signal.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, D, H, W) feature
            maps, list of per-level (B, 1, D) CLS tokens.
        """
        target_modules = {
            self.encoder.transformer.get_submodule(f"layers.{i}.1"): i
            for i in self.out_indices
        }
        captured: dict[int, torch.Tensor] = {}

        def hook(
            module: nn.Module, hook_input: tuple, hook_output: torch.Tensor
        ) -> None:
            captured[target_modules[module]] = (
                hook_input[0] + hook_output
            )  # ff(x) + x, true post-residual value

        hooks = [module.register_forward_hook(hook) for module in target_modules]
        try:
            self.forward(
                image, time=time, latlon=latlon
            )  # (B, 1+L, D) per hooked block, discarded -- hooks captured what we need
        finally:
            for h in hooks:
                h.remove()

        features = [
            captured[i] for i in self.out_indices
        ]  # list[len(out_indices)] of (B, 1+L, D), CLS at idx 0
        prefix_tokens: list[torch.Tensor | None] = [f[:, :1, :] for f in features]  # list of (B, 1, D) -- CLS only
        pyramid = [
            self._tokens_to_spatial(f[:, 1:, :]) for f in features
        ]  # list of (B, D, H, W)
        return pyramid, prefix_tokens

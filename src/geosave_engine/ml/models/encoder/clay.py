from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from terratorch.models.backbones.clay_v15.model import Encoder

from geosave_engine.geodata.spatial import GeoAnchor
from geosave_engine.ml.registry import register_model
from geosave_engine.ml.models.geo_context import lat_lon, week_hour
from geosave_engine.ml.models.contract import chain_step

# Only 'clay_v15_large' has a published checkpoint (verified: made-with-clay/Clay
# HF repo has exactly one file, v1.5/clay-v1.5.ckpt; rslearn's own Clay port
# skips the smaller sizes for the same reason) -- tiny/small/base stay in
# MODEL_NAMES as real architectures to train from scratch, just absent here.
MODEL_SOURCE: dict[str, dict[str, str]] = {
    'clay_v15_large': {'repo_id': 'made-with-clay/Clay', 'filename': 'v1.5/clay-v1.5.ckpt'},
}

# Real ckpt (downloaded + inspected once, then deleted -- see build_clay) is a
# standard PyTorch Lightning checkpoint: ckpt['state_dict'] holds every
# ClayMAE submodule (encoder/decoder/proj/teacher) flattened under
# "model.<submodule>." prefixes. Stripping "model.encoder." gives exactly
# Encoder.named_parameters()'s own names -- confirmed 1:1 (265/265) against a
# real clay_v15_large Encoder, so build_clay loads with strict=True.
_STATE_DICT_ENCODER_PREFIX: str = 'model.encoder.'

# Encoder-only subset of terratorch's clay_mae_{tiny,small,base,large} kwargs
# (clay_v15/model.py) -- copied by hand, not read off those factories: calling
# them builds the full ClayMAE (Decoder + a timm SAM "teacher" downloaded with
# pretrained=True unconditionally, no opt-out -- see Clay's class docstring),
# and we only ever want the encoder dims.
#
# Real published checkpoint (made-with-clay/Clay HF repo) only covers the large
# size -- tiny/small/base are architecture definitions with no pretrained
# weights, kept here anyway since they're legitimate skeletons to train from
# scratch. `pretrained=True` is only meaningful for 'clay_v15_large'.
MODEL_NAMES: dict[str, dict] = {
    'clay_v15_tiny':  {'dim': 192,  'depth': 6,  'heads': 4,  'dim_head': 48, 'mlp_ratio': 2, 'patch_size': 8},
    'clay_v15_small': {'dim': 384,  'depth': 6,  'heads': 6,  'dim_head': 64, 'mlp_ratio': 2, 'patch_size': 8},
    'clay_v15_base':  {'dim': 768,  'depth': 12, 'heads': 12, 'dim_head': 64, 'mlp_ratio': 4, 'patch_size': 8},
    'clay_v15_large': {'dim': 1024, 'depth': 24, 'heads': 16, 'dim_head': 64, 'mlp_ratio': 4, 'patch_size': 8},
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
        [torch.sin(week_angle), torch.cos(week_angle), torch.sin(hour_angle), torch.cos(hour_angle)], dim=-1
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
    return torch.stack([torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon)], dim=-1)


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
        model_name: key of `MODEL_NAMES`.
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
        ValueError: `model_name` not in `MODEL_NAMES`, or `pretrained` is
            `True` and `model_name` has no entry in `MODEL_SOURCE`.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"{model_name!r} not in MODEL_NAMES; must be one of {list(MODEL_NAMES)}"
        )
    if pretrained and checkpoint_path is None and model_name not in MODEL_SOURCE:
        raise ValueError(
            f"{model_name!r} has no published checkpoint; pretrained=True only "
            f"works for {list(MODEL_SOURCE)}"
        )

    spec = MODEL_NAMES[model_name]
    encoder = Encoder(
        mask_ratio=0.0,
        patch_size=spec['patch_size'],
        shuffle=False,
        dim=spec['dim'],
        depth=spec['depth'],
        heads=spec['heads'],
        dim_head=spec['dim_head'],
        mlp_ratio=spec['mlp_ratio'],
    )

    if pretrained:
        source = MODEL_SOURCE[model_name]
        path = checkpoint_path or hf_hub_download(repo_id=source['repo_id'], filename=source['filename'])
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        state_dict = {
            name.removeprefix(_STATE_DICT_ENCODER_PREFIX): param
            for name, param in ckpt['state_dict'].items()
            if name.startswith(_STATE_DICT_ENCODER_PREFIX)
        }
        encoder.load_state_dict(state_dict)

    return encoder


@register_model('encoder', 'clay')
class Clay(nn.Module):
    """Clay v1.5 ViT encoder, wavelength-conditioned per band.

    Backbone is a plain ViT (`build_clay` builds terratorch's `Encoder` class,
    not the official `ClayMAEModule` wrapper — that one drags in an unrelated
    SAM "teacher" model, see `build_clay`'s docstring). Patch embedding is
    `DynamicEmbedding`, conditioned on each band's real wavelength instead of a
    fixed `in_channels` — so switching bands/sensors doesn't cost pretrained-
    weight compatibility the way it does elsewhere in this package.

    Sensor-agnostic on purpose: no `modality`/sensor-catalog lookup lives
    here — `in_channels`/`waves`/`gsd` are the caller's own resolved numbers
    (e.g. from `geodata.sensors`), same shape as `Prithvi`/`DINOv3` take
    plain `in_channels`. Keeps sensor identity a geodata concern, not an ml one.

    `forward_pyramid` reads intermediate blocks via `register_forward_hook` —
    terratorch's `Transformer.forward` has no `out_indices` support of its own.

    No `img_mean`/`img_std` (unlike `Prithvi`/`DINOv3`) — doesn't implement
    `Normalization`, on purpose: `ImageProcessor` already has a real,
    explicit path for this (`SemanticSegmentationTask`'s own `mean_norm`/
    `std_norm` config), so a second, model-attribute fallback here would
    just be a second place the same two numbers could come from. Set
    `mean_norm`/`std_norm` explicitly in config (e.g. from
    `geodata.sensors.band_mean`/`band_std`).
    """

    waves: torch.Tensor
    gsd: torch.Tensor

    def __init__(
        self,
        *,
        model_name: str = 'clay_v15_large',
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
            model_name: must be a key of module-level `MODEL_NAMES`. Only
                `'clay_v15_large'` has a published checkpoint — see the class
                docstring for why.
            in_channels: input channel count. No default — Clay has no one
                pretrained band spec to fall back to (see class docstring),
                unlike `Prithvi`'s `in_channels=6`.
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
            ValueError: `model_name` not in `MODEL_NAMES`; `waves` doesn't have
                exactly `in_channels` values; `input_size` isn't square or
                isn't evenly divisible by `model_name`'s patch size; or
                `pretrained` is `True` with no published checkpoint for
                `model_name` (see `MODEL_SOURCE`).
        """
        super().__init__()
        if len(waves) != in_channels:
            raise ValueError(f"waves must have {in_channels} values (in_channels), got {len(waves)}")

        height, width = (input_size, input_size) if isinstance(input_size, int) else input_size
        if height != width:
            raise ValueError(f"input_size must be square, got {height}x{width}")

        self.encoder = build_clay(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        dim: int = self.encoder.dim
        patch_size: int = self.encoder.patch_size
        depth: int = len(self.encoder.transformer.layers)

        if height % patch_size != 0:
            raise ValueError(f"input_size {height} not evenly divisible by {model_name}'s patch_size {patch_size}")
        self.grid = height // patch_size

        self.out_indices = list(out_indices) if out_indices is not None else default_out_indices(depth)

        self.register_buffer('waves', torch.tensor(waves, dtype=torch.float32))
        self.register_buffer('gsd', torch.tensor(float(gsd)))

        self.out_channels: list[int] = [dim] * len(self.out_indices)
        self.output_strides: list[int] = [patch_size] * len(self.out_indices)

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
        time = _normalize_time(time) if time is not None else torch.zeros(batch_size, 4, device=image.device, dtype=image.dtype)
        latlon = _normalize_latlon(latlon) if latlon is not None else torch.zeros(batch_size, 4, device=image.device, dtype=image.dtype)
        if gsd is None:
            gsd = self.gsd
        if waves is None:
            waves = self.waves

        datacube = {
            'pixels': image,  # (B, C, H, W)
            'time': time,  # (B, 4)
            'latlon': latlon,  # (B, 4)
            'gsd': gsd,  # scalar
            'waves': waves,  # (C,)
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
        return x.transpose(1, 2).reshape(b, dim, self.grid, self.grid)  # (B, L, D) -> (B, D, L) -> (B, D, H, W)

    @staticmethod
    def model_context(anchor: GeoAnchor) -> dict[str, np.ndarray]:
        """This window's own time/latlon, Clay's own forward_pyramid input shape.

        Pure geometry/time math — no `Clay` instance needed. Values are raw, not
        yet sin/cos-encoded (`forward`'s `_normalize_time`/`_normalize_latlon` do that),
        and numpy so the context serializes — `to_sample` tensorizes at this dtype.

        Args:
            anchor: Window to derive from. One row of `time` per step its
                header records.

        Returns:
            {
                "time": (steps, 2) float32 — raw (iso_week, hour) per step,
                "latlon": (2,) float32 — raw (lat, lon) degrees, one per window,
            }
        """
        lat, lon = lat_lon(anchor)
        return {
            "time": week_hour(anchor),
            "latlon": np.array([lat, lon], dtype="float32"),
        }

    @chain_step()
    def forward_pyramid(
        self,
        image: torch.Tensor,
        anchor: list[GeoAnchor] | None = None,
        time: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
    ) -> tuple[list, list]:
        """Extract multi-scale intermediate features from the ViT.

        latlon/time left unset with anchor given derive from it — see
        `model_context`. Calls `self.forward(image, ...)` under the hooks
        below instead of rebuilding the datacube by hand.

        `Transformer.forward` (clay_v15/backbone.py) is a plain block loop —
        only the final block's output leaves the module, nothing intermediate
        is collected. `register_forward_hook` on each target block's
        `FeedForward` submodule captures its output as PyTorch calls it, no
        need to reimplement the loop ourselves. A hook only sees that
        submodule's own output (`ff(x)`, pre-residual) — the hook reconstructs
        the true post-residual value as ``hook_input[0] + hook_output``,
        verified against a manual copy of the real loop before relying on it.

        Args:
            image: (B, C, H, W) input tensor, C == this instance's `in_channels`.
            anchor: This batch's own `"anchor"` list (`batch["anchor"]`), one
                GeoAnchor per sample. Only used to derive time/latlon when
                either isn't given directly.
            time: (B, 2) float32 — raw `(iso_week, hour)`. `None` derives
                it from anchor if given, else keeps `forward()`'s own "no
                time signal" default.
            latlon: (B, 2) float32 — raw `(lat, lon)` in degrees. `None`
                derives it from anchor if given, else keeps `forward()`'s
                own "no location signal" default.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, D, H, W) feature
            maps, list of per-level (B, 1, D) CLS tokens.
        """
        if anchor is not None and (time is None or latlon is None):
            if latlon is None:
                stacked = np.stack([np.array(lat_lon(a), dtype="float32") for a in anchor])
                latlon = torch.as_tensor(stacked).to(image.device)
            if time is None:
                # one row per anchor: a bare anchor carries a span, not the steps a tile would
                stacked = np.stack([week_hour(a)[0] for a in anchor])
                time = torch.as_tensor(stacked).to(image.device)

        target_modules = {self.encoder.transformer.layers[i][1]: i for i in self.out_indices} #type: ignore
        captured: dict[int, torch.Tensor] = {}

        def hook(module: nn.Module, hook_input: tuple, hook_output: torch.Tensor) -> None:
            captured[target_modules[module]] = hook_input[0] + hook_output  # ff(x) + x, true post-residual value

        hooks = [module.register_forward_hook(hook) for module in target_modules]
        try:
            self.forward(image, time=time, latlon=latlon)  # (B, 1+L, D) per hooked block, discarded -- hooks captured what we need
        finally:
            for h in hooks:
                h.remove()

        features = [captured[i] for i in self.out_indices]  # list[len(out_indices)] of (B, 1+L, D), CLS at idx 0
        prefix_tokens = [f[:, :1, :] for f in features]  # list of (B, 1, D) -- CLS only
        pyramid = [self._tokens_to_spatial(f[:, 1:, :]) for f in features]  # list of (B, D, H, W)
        return pyramid, prefix_tokens


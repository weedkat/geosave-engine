from __future__ import annotations

from importlib.resources import files
from typing import Literal

import yaml
from box import Box
from huggingface_hub import hf_hub_download
from terratorch.models.backbones.clay_v15.module import ClayMAEModule


CLAY_MODALITIES: tuple[str, ...] = (
    'sentinel-2-l2a',
    'sentinel-1-rtc',
    'landsat-c2l1',
    'landsat-c2l2-sr',
    'planetscope-sr',
    'naip',
    'linz',
    'modis',
    'satellogic-MSI-L1D',
)

_METADATA_PACKAGE: str = 'geosave_engine.ml.models.encoder.configs'
_METADATA_FILENAME: str = 'metadata.yaml'
_HF_REPO_ID: str = 'made-with-clay/Clay'
_HF_FILENAME: str = 'v1.5/clay-v1.5.ckpt'

CLAY_DOLLS: dict[str, list[int]] = {
    'large': [16, 32, 64, 128, 256, 768, 1024],
    'base':  [16, 32, 64, 128, 256, 768],
}
CLAY_DOLL_WEIGHTS: dict[str, list[int]] = {
    'large': [1, 1, 1, 1, 1, 1, 1],
    'base':  [1, 1, 1, 1, 1, 1],
}


def get_clay_checkpoint_path(
    filename: str = _HF_FILENAME,
    repo_id: str = _HF_REPO_ID,
) -> str:
    """Download (or get cached) Clay v1.5 checkpoint from HuggingFace.

    Args:
        filename: file inside the HF repo.
        repo_id: HF repo id.

    Returns:
        Local filesystem path to the cached checkpoint.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_clay_metadata() -> Box:
    """Load Clay per-modality metadata (band order, mean/std, wavelengths, gsd).

    Returns:
        Box (attrdict) keyed by modality name; e.g. ``meta['sentinel-2-l2a']['band_order']``.
    """
    path = files(_METADATA_PACKAGE).joinpath(_METADATA_FILENAME)
    with path.open('r') as f:
        return Box(yaml.safe_load(f))


def resolve_clay_bands(
    metadata: Box,
    modality: str,
    bands: list[str] | None,
) -> list[str]:
    """Pick bands to feed Clay for a modality.

    Args:
        metadata: result of :func:`load_clay_metadata`.
        modality: one of :data:`CLAY_MODALITIES`; must exist in ``metadata``.
        bands: optional subset; each name must appear in ``metadata[modality]['band_order']``.
            When ``None``, the full pretraining ``band_order`` is used (recommended).

    Returns:
        Ordered list of band names that will index per-band stats and wavelengths.
    """
    if modality not in metadata:
        raise ValueError(
            f"modality {modality!r} not in metadata; available: {list(metadata)}"
        )
    available: list[str] = list(metadata[modality]['band_order'])
    if bands is None:
        return available
    unknown = [b for b in bands if b not in available]
    if unknown:
        raise ValueError(
            f"bands {unknown} not in modality {modality!r}; available: {available}"
        )
    return list(bands)


def build_clay_v15(
    model_size: Literal['large'] = 'large',
    checkpoint_path: str | None = None,
    metadata: Box | None = None,
) -> ClayMAEModule:
    """Load Clay v1.5 weights into a ``ClayMAEModule`` (encoder at ``.model.encoder``).

    Only the encoder is used downstream; the MAE decoder + SAM teacher stay instantiated
    but are not invoked. ``ClayMAEModule`` is a ``LightningModule`` used purely as a
    checkpoint loader.

    Args:
        model_size: Clay v1.5 size; only ``'large'`` supported (no public ``'base'`` ckpt).
        checkpoint_path: optional local ckpt path; ``None`` fetches from HF Hub.
        metadata: optional pre-loaded metadata Box; ``None`` loads fresh.

    Returns:
        Loaded ``ClayMAEModule``. Encoder at ``.model.encoder``; transformer blocks at
        ``.model.encoder.transformer.layers`` (each a ``ModuleList[(Attention, FeedForward)]``).
    """
    if model_size != 'large':  # type: ignore[comparison-overlap]
        raise ValueError(
            f"Clay v1.5 only supports 'large' (no public 'base' ckpt); got {model_size!r}"
        )
    ckpt = checkpoint_path or get_clay_checkpoint_path()
    meta_path = str(files(_METADATA_PACKAGE).joinpath(_METADATA_FILENAME))
    module = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=ckpt,
        model_size=model_size,
        metadata_path=meta_path,
        dolls=CLAY_DOLLS[model_size],
        doll_weights=CLAY_DOLL_WEIGHTS[model_size],
        mask_ratio=0.0,
        shuffle=False,
    )
    if metadata is not None:
        module.metadata = metadata
    return module

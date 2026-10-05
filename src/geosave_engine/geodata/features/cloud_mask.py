"""Cloud and validity masks derived from Sentinel-2 rasters."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter
from s2cloudless.cloud_detector import S2PixelCloudDetector

from geosave_engine.geodata.utils.xarray import map_spatial_overlap

from ._raster import feature_raster, prepared_reflectance


S2C_BAND_ORDER = (
    "b01",
    "b02",
    "b04",
    "b05",
    "b08",
    "b8a",
    "b09",
    "b10",
    "b11",
    "b12",
)
SCL_VALID_CLASSES = (
    2,  # dark area
    4,  # vegetation
    5,  # bare soil
    6,  # water
    7,  # unclassified
    11,  # snow/ice
)


def s2cloudless_mask(
    raster: xr.Dataset,
    *,
    name: str,
    b01: str,
    b02: str,
    b04: str,
    b05: str,
    b08: str,
    b8a: str,
    b09: str,
    b10: str,
    b11: str,
    b12: str,
    probability_threshold: float = 0.4,
) -> xr.Dataset:
    """Derive a named s2cloudless mask raster, one chunk at a time.

    Needs Sentinel-2 L1C TOA reflectance in [0, 1]. Prepare stored bands
    explicitly with `.gs.to_nan().gs.unpack()` first. Bands are ordered into
    `S2C_BAND_ORDER` here.

    Args:
        raster: Prepared Sentinel-2 reflectance raster.
        name: Output variable name.
        b01: Coastal aerosol variable name.
        b02: Blue variable name.
        b04: Red variable name.
        b05: Red-edge 1 variable name.
        b08: Near-infrared variable name.
        b8a: Narrow near-infrared variable name.
        b09: Water vapour variable name.
        b10: Cirrus variable name.
        b11: Shortwave-infrared 1 variable name.
        b12: Shortwave-infrared 2 variable name.
        probability_threshold: Cloud probability above which a pixel is flagged.

    Returns:
        One-variable bool raster, True where cloud, lazy when the input is.
    """
    bands = prepared_reflectance(
        raster, b01, b02, b04, b05, b08, b8a, b09, b10, b11, b12
    )
    field = map_spatial_overlap(
        _s2cloudless_block,
        *bands,
        depth=3,
        dtype="bool",
        probability_threshold=probability_threshold,
    )
    return feature_raster(raster, field, name=name, reference=bands[0])


def cdi_cloud_mask(
    raster: xr.Dataset,
    *,
    name: str,
    b07: str,
    b08: str,
    b8a: str,
    cdi_threshold: float = -0.5,
    eps: float = 1e-6,
) -> xr.Dataset:
    """Derive a named Cloud Displacement Index mask raster.

    CDI = (V(B07/B8A) - V(B08/B8A)) / (V(B07/B8A) + V(B08/B8A)), where V is
    local variance. B08 is pre-smoothed to match B07/B8A's coarser native
    resolution. Missing observations are excluded from neighborhood estimates;
    an unavailable center pixel is not flagged as cloud.

    Args:
        raster: Prepared reflectance raster.
        name: Output variable name.
        b07: Band 7 variable name.
        b08: Band 8 variable name.
        b8a: Band 8A variable name.
        cdi_threshold: CDI below this is flagged as cloud.
        eps: Guards division by zero.

    Returns:
        One-variable bool raster, True where cloud, lazy when the input is.
    """
    b07_band, b08_band, b8a_band = prepared_reflectance(raster, b07, b08, b8a)
    field = map_spatial_overlap(
        _cdi_block,
        b07_band,
        b08_band,
        b8a_band,
        depth=8,
        dtype="bool",
        cdi_threshold=cdi_threshold,
        eps=eps,
    )
    return feature_raster(raster, field, name=name, reference=b07_band)


def cirrus_cloud_mask(
    raster: xr.Dataset,
    *,
    name: str,
    b10: str,
    reflectance_threshold: float = 0.01,
) -> xr.Dataset:
    """Derive a named cirrus mask from Sentinel-2 Band B10 reflectance.

    Args:
        raster: Prepared reflectance raster.
        name: Output variable name.
        b10: Band 10 variable name.
        reflectance_threshold: Reflectance above which cirrus is flagged.

    Returns:
        One-variable bool raster, True where cirrus, lazy when the input is.
    """
    (band,) = prepared_reflectance(raster, b10)
    field = band > reflectance_threshold
    return feature_raster(raster, field, name=name, reference=band)


def scl_valid_mask(
    raster: xr.Dataset,
    *,
    name: str,
    scl: str,
    valid_classes: Sequence[int] = SCL_VALID_CLASSES,
) -> xr.Dataset:
    """Derive pixels valid under Sentinel-2 L2A's Scene Classification Layer.

    Sen2Cor's own per-pixel classification: 0=no data, 1=saturated, 2=dark,
    3=cloud shadow, 4=vegetation, 5=bare soil, 6=water, 7=unclassified,
    8/9=cloud med/high prob, 10=cirrus, 11=snow/ice.

    Args:
        raster: Raster carrying a Scene Classification Layer.
        name: Output variable name.
        scl: Scene Classification Layer variable name.
        valid_classes: SCL values retained as valid. The default includes dark
            areas and snow; pass a narrower set such as 4, 5, 6, and 7 when
            those should be excluded.

    Returns:
        One-variable bool raster, True where pixels are valid, lazy when the
        input is.
    """
    field = raster[scl].isin(valid_classes)
    return feature_raster(raster, field, name=name, reference=raster[scl])


@lru_cache(maxsize=4)
def _s2cloudless_detector(
    probability_threshold: float,
) -> S2PixelCloudDetector:
    """Load one detector per probability threshold and process."""
    return S2PixelCloudDetector(
        threshold=probability_threshold,
        all_bands=False,
    )


def _s2cloudless_block(
    *bands: np.ndarray,
    probability_threshold: float,
) -> np.ndarray:
    """Run s2cloudless over one block in `S2C_BAND_ORDER`."""
    stacked = np.stack(bands, axis=-1).astype(np.float32)
    batch = stacked[np.newaxis] if stacked.ndim == 3 else stacked
    masks = _s2cloudless_detector(probability_threshold).get_cloud_masks(batch)
    return masks[0] if stacked.ndim == 3 else masks


def _local_variance(
    values: np.ndarray,
    size: int | tuple[int, int, int] = 7,
) -> np.ndarray:
    """Compute finite-weighted local spatial variance."""
    valid = np.isfinite(values)
    observed = np.where(valid, values, 0)
    # SciPy accepts per-axis widths; its untyped default is inferred as int.
    weight = uniform_filter(valid.astype(np.float32), size=size)  # pyright: ignore[reportArgumentType]
    mean = np.divide(
        uniform_filter(observed, size=size),  # pyright: ignore[reportArgumentType]
        weight,
        out=np.full_like(values, np.nan),
        where=weight > 0,
    )
    mean_sq = np.divide(
        uniform_filter(observed**2, size=size),  # pyright: ignore[reportArgumentType]
        weight,
        out=np.full_like(values, np.nan),
        where=weight > 0,
    )
    return mean_sq - mean**2


def _cdi_block(
    b07: np.ndarray,
    b08: np.ndarray,
    b8a: np.ndarray,
    *,
    cdi_threshold: float,
    eps: float,
) -> np.ndarray:
    """Compute a CDI cloud mask over one block."""
    b07 = b07.astype(np.float32)
    b8a = b8a.astype(np.float32)
    sigma = (0.0, 1.0, 1.0) if b08.ndim == 3 else 1.0
    window = (1, 7, 7) if b08.ndim == 3 else 7
    valid = np.isfinite(b07) & np.isfinite(b08) & np.isfinite(b8a)
    observed = np.isfinite(b08)
    weight = gaussian_filter(observed.astype(np.float32), sigma=sigma)
    b08 = np.divide(
        gaussian_filter(np.where(observed, b08, 0).astype(np.float32), sigma=sigma),
        weight,
        out=np.full_like(b08, np.nan),
        where=weight > 0,
    )
    b08 = np.where(observed, b08, np.nan)

    variance_b07 = _local_variance(b07 / (b8a + eps), size=window)
    variance_b08 = _local_variance(b08 / (b8a + eps), size=window)
    cdi = (variance_b07 - variance_b08) / (variance_b07 + variance_b08 + eps)
    return valid & (cdi < cdi_threshold)

"""Named spectral-index rasters derived from prepared reflectance."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._raster import feature_raster, prepared_reflectance


def _bands(raster: xr.Dataset, **selectors: str) -> tuple[xr.DataArray, ...]:
    """Select prepared reflectance variables in semantic argument order."""
    return prepared_reflectance(raster, *selectors.values())


def _ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    """Divide, answering NaN where the denominator vanishes."""
    return numerator / denominator.where(denominator != 0)


def _normalized_difference(
    raster: xr.Dataset,
    *,
    name: str,
    a: str,
    b: str,
    eps: float,
) -> xr.Dataset:
    """Return one named normalized-difference raster."""
    first, second = _bands(raster, a=a, b=b)
    field = _ratio(first - second, first + second + eps)
    return feature_raster(raster, field, name=name, reference=first)


def ndvi(
    raster: xr.Dataset, *, name: str, nir: str, red: str, eps: float = 0.0
) -> xr.Dataset:
    """Derive a normalized difference vegetation index raster.

    Args:
        raster: Prepared reflectance raster.
        name: Output variable name.
        nir: Near-infrared variable name.
        red: Red variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        One-variable NDVI raster on the input grid.
    """
    return _normalized_difference(raster, name=name, a=nir, b=red, eps=eps)


def evi(
    raster: xr.Dataset,
    *,
    name: str,
    nir: str,
    red: str,
    blue: str,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    L: float = 1.0,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive an enhanced vegetation index raster."""
    nir_band, red_band, blue_band = _bands(raster, nir=nir, red=red, blue=blue)
    field = g * _ratio(
        nir_band - red_band,
        nir_band + c1 * red_band - c2 * blue_band + L + eps,
    )
    return feature_raster(raster, field, name=name, reference=nir_band)


def evi2(
    raster: xr.Dataset,
    *,
    name: str,
    nir: str,
    red: str,
    g: float = 2.5,
    c: float = 2.4,
    L: float = 1.0,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive the two-band enhanced vegetation index raster."""
    nir_band, red_band = _bands(raster, nir=nir, red=red)
    field = g * _ratio(nir_band - red_band, nir_band + c * red_band + L + eps)
    return feature_raster(raster, field, name=name, reference=nir_band)


def savi(
    raster: xr.Dataset,
    *,
    name: str,
    nir: str,
    red: str,
    L: float = 0.5,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive a soil-adjusted vegetation index raster."""
    nir_band, red_band = _bands(raster, nir=nir, red=red)
    field = _ratio(nir_band - red_band, nir_band + red_band + L + eps) * (1 + L)
    return feature_raster(raster, field, name=name, reference=nir_band)


def msavi2(raster: xr.Dataset, *, name: str, nir: str, red: str) -> xr.Dataset:
    """Derive a modified soil-adjusted vegetation index raster."""
    nir_band, red_band = _bands(raster, nir=nir, red=red)
    term = np.clip((2 * nir_band + 1) ** 2 - 8 * (nir_band - red_band), 0.0, None)
    field = (2 * nir_band + 1 - np.sqrt(term)) / 2
    return feature_raster(raster, field, name=name, reference=nir_band)


def ndre(
    raster: xr.Dataset,
    *,
    name: str,
    nir: str,
    red_edge: str,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive a normalized difference red-edge index raster."""
    return _normalized_difference(raster, name=name, a=nir, b=red_edge, eps=eps)


def bsi(
    raster: xr.Dataset,
    *,
    name: str,
    swir1: str,
    red: str,
    nir: str,
    blue: str,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive a bare soil index raster."""
    swir, red_band, nir_band, blue_band = _bands(
        raster, swir1=swir1, red=red, nir=nir, blue=blue
    )
    field = _ratio(
        (swir + red_band) - (nir_band + blue_band),
        (swir + red_band) + (nir_band + blue_band) + eps,
    )
    return feature_raster(raster, field, name=name, reference=swir)


def ndbi(
    raster: xr.Dataset, *, name: str, swir1: str, nir: str, eps: float = 0.0
) -> xr.Dataset:
    """Derive a normalized difference built-up index raster."""
    return _normalized_difference(raster, name=name, a=swir1, b=nir, eps=eps)


def ndwi(
    raster: xr.Dataset, *, name: str, green: str, nir: str, eps: float = 0.0
) -> xr.Dataset:
    """Derive a normalized difference water index raster."""
    return _normalized_difference(raster, name=name, a=green, b=nir, eps=eps)


def mndwi(
    raster: xr.Dataset,
    *,
    name: str,
    green: str,
    swir1: str,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive a modified normalized difference water index raster."""
    return _normalized_difference(raster, name=name, a=green, b=swir1, eps=eps)


def ndci(
    raster: xr.Dataset,
    *,
    name: str,
    red_edge: str,
    red: str,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive a normalized difference chlorophyll index raster."""
    return _normalized_difference(raster, name=name, a=red_edge, b=red, eps=eps)


def ndmi(
    raster: xr.Dataset, *, name: str, nir: str, swir1: str, eps: float = 0.0
) -> xr.Dataset:
    """Derive a normalized difference moisture index raster."""
    return _normalized_difference(raster, name=name, a=nir, b=swir1, eps=eps)


def nbr(
    raster: xr.Dataset, *, name: str, nir: str, swir2: str, eps: float = 0.0
) -> xr.Dataset:
    """Derive a normalized burn ratio raster."""
    return _normalized_difference(raster, name=name, a=nir, b=swir2, eps=eps)


def ndsi(
    raster: xr.Dataset,
    *,
    name: str,
    green: str,
    swir1: str,
    eps: float = 0.0,
) -> xr.Dataset:
    """Derive a normalized difference snow index raster."""
    return _normalized_difference(raster, name=name, a=green, b=swir1, eps=eps)

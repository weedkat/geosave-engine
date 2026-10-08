"""Spectral indices derived from reflectance bands.

Each index decodes packing and nodata metadata on its selected variables
with `.gs.mask_and_scale()`, then calculates an unnamed float32 band.
Already decoded values pass through; variables without storage metadata
are treated as reflectance. Execution stays lazy where the inputs are.

    scene = scene.assign(
        ndvi=ndvi(scene, nir="B08", red="B04"),
        evi2=evi2(scene, nir="B08", red="B04"),
    )
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    """Divide, answering NaN where the denominator vanishes."""
    return numerator / denominator.where(denominator != 0)


def _normalized_difference(
    scene: xr.Dataset, a: str, b: str, eps: float
) -> xr.DataArray:
    """Return `(a - b) / (a + b + eps)` over prepared bands."""
    bands = scene[[a, b]].gs.mask_and_scale().astype(np.float32)
    a_band, b_band = bands[a], bands[b]
    result = _ratio(a_band - b_band, a_band + b_band + eps)
    return result.drop_attrs(deep=False).rename(None)


def ndvi(scene: xr.Dataset, *, nir: str, red: str, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference vegetation index.

    Formula:
        `(nir - red) / (nir + red + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        red: Red variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the inputs' grid.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.

    Examples:
        >>> scene.assign(ndvi=ndvi(scene, nir="B08", red="B04")).ndvi.dims
        ('time', 'y', 'x')
    """
    return _normalized_difference(scene, nir, red, eps)


def evi(
    scene: xr.Dataset,
    *,
    nir: str,
    red: str,
    blue: str,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    L: float = 1.0,
    eps: float = 0.0,
) -> xr.DataArray:
    """Derive the enhanced vegetation index.

    Formula:
        `g * (nir - red) / (nir + c1 * red - c2 * blue + L + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        red: Red variable name.
        blue: Blue variable name.
        g: Gain factor.
        c1: Red atmospheric correction coefficient.
        c2: Blue atmospheric correction coefficient.
        L: Soil adjustment term.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    bands = scene[[nir, red, blue]].gs.mask_and_scale().astype(np.float32)
    nir_band, red_band, blue_band = bands[nir], bands[red], bands[blue]
    result = g * _ratio(
        nir_band - red_band, nir_band + c1 * red_band - c2 * blue_band + L + eps
    )
    return result.drop_attrs(deep=False).rename(None)


def evi2(
    scene: xr.Dataset,
    *,
    nir: str,
    red: str,
    g: float = 2.5,
    c: float = 2.4,
    L: float = 1.0,
    eps: float = 0.0,
) -> xr.DataArray:
    """Derive the two-band enhanced vegetation index.

    Formula:
        `g * (nir - red) / (nir + c * red + L + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        red: Red variable name.
        g: Gain factor.
        c: Red correction coefficient.
        L: Soil adjustment term.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    bands = scene[[nir, red]].gs.mask_and_scale().astype(np.float32)
    nir_band, red_band = bands[nir], bands[red]
    result = g * _ratio(nir_band - red_band, nir_band + c * red_band + L + eps)
    return result.drop_attrs(deep=False).rename(None)


def savi(
    scene: xr.Dataset, *, nir: str, red: str, L: float = 0.5, eps: float = 0.0
) -> xr.DataArray:
    """Derive the soil-adjusted vegetation index.

    Formula:
        `(1 + L) * (nir - red) / (nir + red + L + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        red: Red variable name.
        L: Soil adjustment term.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    bands = scene[[nir, red]].gs.mask_and_scale().astype(np.float32)
    nir_band, red_band = bands[nir], bands[red]
    result = _ratio(nir_band - red_band, nir_band + red_band + L + eps) * (1 + L)
    return result.drop_attrs(deep=False).rename(None)


def msavi2(scene: xr.Dataset, *, nir: str, red: str) -> xr.DataArray:
    """Derive the modified soil-adjusted vegetation index.

    Formula:
        `(2 * nir + 1 - sqrt(D)) / 2`, where
        `D = max((2 * nir + 1)**2 - 8 * (nir - red), 0)` elementwise.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        red: Red variable name.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    bands = scene[[nir, red]].gs.mask_and_scale().astype(np.float32)
    nir_band, red_band = bands[nir], bands[red]
    term = np.clip((2 * nir_band + 1) ** 2 - 8 * (nir_band - red_band), 0.0, None)
    result = (2 * nir_band + 1 - np.sqrt(term)) / 2
    return result.drop_attrs(deep=False).rename(None)


def ndre(
    scene: xr.Dataset, *, nir: str, red_edge: str, eps: float = 0.0
) -> xr.DataArray:
    """Derive the normalized difference red-edge index.

    Formula:
        `(nir - red_edge) / (nir + red_edge + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        red_edge: Red-edge variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, nir, red_edge, eps)


def bsi(
    scene: xr.Dataset,
    *,
    swir1: str,
    red: str,
    nir: str,
    blue: str,
    eps: float = 0.0,
) -> xr.DataArray:
    """Derive the bare soil index.

    Formula:
        `((swir1 + red) - (nir + blue)) /
        ((swir1 + red) + (nir + blue) + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        swir1: Shortwave-infrared 1 variable name.
        red: Red variable name.
        nir: Near-infrared variable name.
        blue: Blue variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    bands = scene[[swir1, red, nir, blue]].gs.mask_and_scale().astype(np.float32)
    swir1_band, red_band = bands[swir1], bands[red]
    nir_band, blue_band = bands[nir], bands[blue]
    result = _ratio(
        (swir1_band + red_band) - (nir_band + blue_band),
        (swir1_band + red_band) + (nir_band + blue_band) + eps,
    )
    return result.drop_attrs(deep=False).rename(None)


def ndbi(scene: xr.Dataset, *, swir1: str, nir: str, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference built-up index.

    Formula:
        `(swir1 - nir) / (swir1 + nir + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        swir1: Shortwave-infrared 1 variable name.
        nir: Near-infrared variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, swir1, nir, eps)


def ndwi(scene: xr.Dataset, *, green: str, nir: str, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference water index.

    Formula:
        `(green - nir) / (green + nir + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        green: Green variable name.
        nir: Near-infrared variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, green, nir, eps)


def mndwi(
    scene: xr.Dataset, *, green: str, swir1: str, eps: float = 0.0
) -> xr.DataArray:
    """Derive the modified normalized difference water index.

    Formula:
        `(green - swir1) / (green + swir1 + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        green: Green variable name.
        swir1: Shortwave-infrared 1 variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, green, swir1, eps)


def ndci(
    scene: xr.Dataset, *, red_edge: str, red: str, eps: float = 0.0
) -> xr.DataArray:
    """Derive the normalized difference chlorophyll index.

    Formula:
        `(red_edge - red) / (red_edge + red + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        red_edge: Red-edge variable name.
        red: Red variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, red_edge, red, eps)


def ndmi(scene: xr.Dataset, *, nir: str, swir1: str, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference moisture index.

    Formula:
        `(nir - swir1) / (nir + swir1 + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        swir1: Shortwave-infrared 1 variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, nir, swir1, eps)


def nbr(scene: xr.Dataset, *, nir: str, swir2: str, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized burn ratio.

    Formula:
        `(nir - swir2) / (nir + swir2 + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        nir: Near-infrared variable name.
        swir2: Shortwave-infrared 2 variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, nir, swir2, eps)


def ndsi(
    scene: xr.Dataset, *, green: str, swir1: str, eps: float = 0.0
) -> xr.DataArray:
    """Derive the normalized difference snow index.

    Formula:
        `(green - swir1) / (green + swir1 + eps)`.

    Args:
        scene: Reflectance or stored bands with packing and nodata metadata.
        green: Green variable name.
        swir1: Shortwave-infrared 1 variable name.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the scene's grid, lazy when its bands are.

    Raises:
        KeyError: A selected variable is absent.
        ValueError: Selected band metadata is incompatible with decoding.
    """
    return _normalized_difference(scene, green, swir1, eps)

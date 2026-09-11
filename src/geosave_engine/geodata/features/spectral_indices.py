"""Spectral indices over named reflectance bands."""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_ndvi(
    *, nir: xr.DataArray, red: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NDVI.

    Args:
        nir: Near-infrared reflectance.
        red: Red reflectance.
        eps: Denominator guard.

    Returns:
        NDVI over the inputs' own grid.

    Examples:
        >>> compute_ndvi(nir=ds.gs["B08"], red=ds.gs["B04"])
    """
    nir, red = nir.astype(np.float32), red.astype(np.float32)
    return (nir - red) / (nir + red + eps)


def compute_evi(
    *,
    nir: xr.DataArray,
    red: xr.DataArray,
    blue: xr.DataArray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    L: float = 1.0,
    eps: float = 1e-6,
) -> xr.DataArray:
    """Compute EVI.

    Args:
        nir: Near-infrared reflectance.
        red: Red reflectance.
        blue: Blue reflectance.
        g: Gain factor.
        c1: Red aerosol-resistance coefficient.
        c2: Blue aerosol-resistance coefficient.
        L: Canopy-background adjustment.
        eps: Denominator guard.

    Returns:
        EVI over the inputs' own grid.
    """
    nir, red, blue = (
        nir.astype(np.float32),
        red.astype(np.float32),
        blue.astype(np.float32),
    )
    return g * (nir - red) / (nir + c1 * red - c2 * blue + L + eps)


def compute_evi2(
    *,
    nir: xr.DataArray,
    red: xr.DataArray,
    g: float = 2.5,
    c: float = 2.4,
    L: float = 1.0,
    eps: float = 1e-6,
) -> xr.DataArray:
    """Compute EVI2, the two-band EVI needing no blue band.

    Args:
        nir: Near-infrared reflectance.
        red: Red reflectance.
        g: Gain factor.
        c: Red coefficient.
        L: Canopy-background adjustment.
        eps: Denominator guard.

    Returns:
        EVI2 over the inputs' own grid.
    """
    nir, red = nir.astype(np.float32), red.astype(np.float32)
    return g * (nir - red) / (nir + c * red + L + eps)


def compute_savi(
    *, nir: xr.DataArray, red: xr.DataArray, L: float = 0.5, eps: float = 1e-6
) -> xr.DataArray:
    """Compute SAVI.

    Args:
        nir: Near-infrared reflectance.
        red: Red reflectance.
        L: Soil-brightness adjustment.
        eps: Denominator guard.

    Returns:
        SAVI over the inputs' own grid.
    """
    nir, red = nir.astype(np.float32), red.astype(np.float32)
    return ((nir - red) / (nir + red + L + eps)) * (1.0 + L)


def compute_msavi2(*, nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
    """Compute MSAVI2, which needs no soil-brightness constant.

    Args:
        nir: Near-infrared reflectance.
        red: Red reflectance.

    Returns:
        MSAVI2 over the inputs' own grid.
    """
    nir, red = nir.astype(np.float32), red.astype(np.float32)
    term = np.clip((2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red), 0.0, None)
    return (2.0 * nir + 1.0 - np.sqrt(term)) / 2.0


def compute_ndre(
    *, nir: xr.DataArray, red_edge: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NDRE.

    Args:
        nir: Near-infrared reflectance.
        red_edge: Red-edge reflectance.
        eps: Denominator guard.

    Returns:
        NDRE over the inputs' own grid.
    """
    nir, red_edge = nir.astype(np.float32), red_edge.astype(np.float32)
    return (nir - red_edge) / (nir + red_edge + eps)


def compute_bsi(
    *,
    swir1: xr.DataArray,
    red: xr.DataArray,
    nir: xr.DataArray,
    blue: xr.DataArray,
    eps: float = 1e-6,
) -> xr.DataArray:
    """Compute BSI.

    Args:
        swir1: Shortwave-infrared 1 reflectance.
        red: Red reflectance.
        nir: Near-infrared reflectance.
        blue: Blue reflectance.
        eps: Denominator guard.

    Returns:
        BSI over the inputs' own grid.
    """
    swir1, red = swir1.astype(np.float32), red.astype(np.float32)
    nir, blue = nir.astype(np.float32), blue.astype(np.float32)
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)


def compute_ndbi(
    *, swir1: xr.DataArray, nir: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NDBI.

    Args:
        swir1: Shortwave-infrared 1 reflectance.
        nir: Near-infrared reflectance.
        eps: Denominator guard.

    Returns:
        NDBI over the inputs' own grid.
    """
    swir1, nir = swir1.astype(np.float32), nir.astype(np.float32)
    return (swir1 - nir) / (swir1 + nir + eps)


def compute_ndwi(
    *, green: xr.DataArray, nir: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NDWI.

    Args:
        green: Green reflectance.
        nir: Near-infrared reflectance.
        eps: Denominator guard.

    Returns:
        NDWI over the inputs' own grid.
    """
    green, nir = green.astype(np.float32), nir.astype(np.float32)
    return (green - nir) / (green + nir + eps)


def compute_mndwi(
    *, green: xr.DataArray, swir1: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute MNDWI.

    Args:
        green: Green reflectance.
        swir1: Shortwave-infrared 1 reflectance.
        eps: Denominator guard.

    Returns:
        MNDWI over the inputs' own grid.
    """
    green, swir1 = green.astype(np.float32), swir1.astype(np.float32)
    return (green - swir1) / (green + swir1 + eps)


def compute_ndci(
    *, red_edge: xr.DataArray, red: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NDCI.

    Args:
        red_edge: Red-edge reflectance.
        red: Red reflectance.
        eps: Denominator guard.

    Returns:
        NDCI over the inputs' own grid.
    """
    red_edge, red = red_edge.astype(np.float32), red.astype(np.float32)
    return (red_edge - red) / (red_edge + red + eps)


def compute_ndmi(
    *, nir: xr.DataArray, swir1: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NDMI.

    Args:
        nir: Near-infrared reflectance.
        swir1: Shortwave-infrared 1 reflectance.
        eps: Denominator guard.

    Returns:
        NDMI over the inputs' own grid.
    """
    nir, swir1 = nir.astype(np.float32), swir1.astype(np.float32)
    return (nir - swir1) / (nir + swir1 + eps)


def compute_nbr(
    *, nir: xr.DataArray, swir2: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NBR.

    Args:
        nir: Near-infrared reflectance.
        swir2: Shortwave-infrared 2 reflectance.
        eps: Denominator guard.

    Returns:
        NBR over the inputs' own grid.
    """
    nir, swir2 = nir.astype(np.float32), swir2.astype(np.float32)
    return (nir - swir2) / (nir + swir2 + eps)


def compute_ndsi(
    *, green: xr.DataArray, swir1: xr.DataArray, eps: float = 1e-6
) -> xr.DataArray:
    """Compute NDSI.

    Args:
        green: Green reflectance.
        swir1: Shortwave-infrared 1 reflectance.
        eps: Denominator guard.

    Returns:
        NDSI over the inputs' own grid.
    """
    green, swir1 = green.astype(np.float32), swir1.astype(np.float32)
    return (green - swir1) / (green + swir1 + eps)

import numpy as np
import xarray as xr

# =============================================================================
# 1. VEGETATION & CANOPY INDICES
# =============================================================================

def compute_ndvi(nir: xr.DataArray, red: xr.DataArray, eps: float = 1e-6) -> xr.DataArray:
    """Normalized Difference Vegetation Index.

    Sentinel-2: nir=B08, red=B04. Output range [-1, 1].

    Args:
        nir: (y, x) near-infrared reflectance.
        red: (y, x) red reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NDVI values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + eps)


def compute_evi(
    nir: xr.DataArray,
    red: xr.DataArray,
    blue: xr.DataArray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    L: float = 1.0,
    eps: float = 1e-6
) -> xr.DataArray:
    """Enhanced Vegetation Index.

    Sentinel-2: nir=B08, red=B04, blue=B02.

    Args:
        nir: (y, x) near-infrared reflectance.
        red: (y, x) red reflectance.
        blue: (y, x) blue reflectance.
        g: Gain factor (default 2.5).
        c1: Aerosol resistance red coefficient (default 6.0).
        c2: Aerosol resistance blue coefficient (default 7.5).
        L: Canopy background adjustment factor (default 1.0).
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) EVI values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    blue = blue.astype(np.float32)
    
    denominator = nir + (c1 * red) - (c2 * blue) + L + eps
    return g * ((nir - red) / denominator)


def compute_evi2(
    nir: xr.DataArray,
    red: xr.DataArray,
    g: float = 2.5,
    c: float = 2.4,
    L: float = 1.0,
    eps: float = 1e-6
) -> xr.DataArray:
    """Two-Band Enhanced Vegetation Index (Does not require Blue band).

    Sentinel-2: nir=B08, red=B04. High performance over atmospheric noise.

    Args:
        nir: (y, x) near-infrared reflectance.
        red: (y, x) red reflectance.
        g: Gain factor (default 2.5).
        c: Red coefficient (default 2.4).
        L: Canopy background adjustment factor (default 1.0).
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) EVI2 values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return g * ((nir - red) / (nir + (c * red) + L + eps))


def compute_savi(
    nir: xr.DataArray,
    red: xr.DataArray,
    L: float = 0.5,
    eps: float = 1e-6
) -> xr.DataArray:
    """Soil-Adjusted Vegetation Index.

    Sentinel-2: nir=B08, red=B04.

    Args:
        nir: (y, x) near-infrared reflectance.
        red: (y, x) red reflectance.
        L: Soil brightness correction factor (default 0.5).
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) SAVI values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return ((nir - red) / (nir + red + L + eps)) * (1.0 + L)


def compute_msavi2(nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
    """Modified Soil-Adjusted Vegetation Index 2 (Self-adjusting soil factor).

    Sentinel-2: nir=B08, red=B04.

    Args:
        nir: (y, x) near-infrared reflectance.
        red: (y, x) red reflectance.

    Returns:
        (y, x) MSAVI2 values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    
    term = (2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red)
    # Clip negative values inside square root to 0 to prevent NaN
    term = np.clip(term, 0.0, None)
    
    return (2.0 * nir + 1.0 - np.sqrt(term)) / 2.0


def compute_ndre(
    nir: xr.DataArray,
    red_edge: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Normalized Difference Red Edge Index (Chlorophyll & nitrogen sensing).

    Sentinel-2: nir=B08, red_edge=B05 (or B06/B07).

    Args:
        nir: (y, x) near-infrared reflectance.
        red_edge: (y, x) red-edge reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NDRE values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    red_edge = red_edge.astype(np.float32)
    return (nir - red_edge) / (nir + red_edge + eps)


# =============================================================================
# 2. SOIL, URBAN & BUILT-UP INDICES
# =============================================================================

def compute_bsi(
    swir1: xr.DataArray,
    red: xr.DataArray,
    nir: xr.DataArray,
    blue: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Bare Soil Index (Differentiates bare soil from vegetation/urban).

    Sentinel-2: swir1=B11, red=B04, nir=B08, blue=B02. Output range [-1, 1].

    Args:
        swir1: (y, x) short-wave infrared 1 reflectance.
        red: (y, x) red reflectance.
        nir: (y, x) near-infrared reflectance.
        blue: (y, x) blue reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) BSI values in [-1, 1].
    """
    swir1 = swir1.astype(np.float32)
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    blue = blue.astype(np.float32)
    
    numerator = (swir1 + red) - (nir + blue)
    denominator = (swir1 + red) + (nir + blue) + eps
    return numerator / denominator


def compute_ndbi(
    swir1: xr.DataArray,
    nir: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Normalized Difference Built-Up Index.

    Sentinel-2: swir1=B11, nir=B08. Output range [-1, 1].

    Args:
        swir1: (y, x) short-wave infrared 1 reflectance.
        nir: (y, x) near-infrared reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NDBI values in [-1, 1].
    """
    swir1 = swir1.astype(np.float32)
    nir = nir.astype(np.float32)
    return (swir1 - nir) / (swir1 + nir + eps)


# =============================================================================
# 3. WATER & AQUATIC INDICES
# =============================================================================

def compute_ndwi(
    green: xr.DataArray,
    nir: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Normalized Difference Water Index (McFeeters).

    Sentinel-2: green=B03, nir=B08.

    Args:
        green: (y, x) green reflectance.
        nir: (y, x) near-infrared reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NDWI values in [-1, 1].
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    return (green - nir) / (green + nir + eps)


def compute_mndwi(
    green: xr.DataArray,
    swir1: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Modified Normalized Difference Water Index (Xu - suppresses urban noise).

    Sentinel-2: green=B03, swir1=B11.

    Args:
        green: (y, x) green reflectance.
        swir1: (y, x) short-wave infrared 1 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) MNDWI values in [-1, 1].
    """
    green = green.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    return (green - swir1) / (green + swir1 + eps)


def compute_ndci(
    red_edge: xr.DataArray,
    red: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Normalized Difference Chlorophyll Index (Water quality & algae blooms).

    Sentinel-2: red_edge=B05, red=B04. Designed specifically for water bodies.

    Args:
        red_edge: (y, x) red-edge 1 reflectance.
        red: (y, x) red reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NDCI values in [-1, 1].
    """
    red_edge = red_edge.astype(np.float32)
    red = red.astype(np.float32)
    return (red_edge - red) / (red_edge + red + eps)


def compute_ndmi(
    nir: xr.DataArray,
    swir1: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Normalized Difference Moisture Index (Canopy water content).

    Sentinel-2: nir=B08, swir1=B11.

    Args:
        nir: (y, x) near-infrared reflectance.
        swir1: (y, x) short-wave infrared 1 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NDMI values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    return (nir - swir1) / (nir + swir1 + eps)


# =============================================================================
# 4. FIRE & SNOW INDICES
# =============================================================================

def compute_nbr(
    nir: xr.DataArray,
    swir2: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Normalized Burn Ratio.

    Sentinel-2: nir=B08, swir2=B12.

    Args:
        nir: (y, x) near-infrared reflectance.
        swir2: (y, x) short-wave infrared 2 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NBR values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    swir2 = swir2.astype(np.float32)
    return (nir - swir2) / (nir + swir2 + eps)


def compute_ndsi(
    green: xr.DataArray,
    swir1: xr.DataArray,
    eps: float = 1e-6
) -> xr.DataArray:
    """Normalized Difference Snow Index.

    Sentinel-2: green=B03, swir1=B11.

    Args:
        green: (y, x) green reflectance.
        swir1: (y, x) short-wave infrared 1 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (y, x) NDSI values in [-1, 1].
    """
    green = green.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    return (green - swir1) / (green + swir1 + eps)
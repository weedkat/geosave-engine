import numpy as np

# =============================================================================
# 1. VEGETATION & CANOPY INDICES
# =============================================================================

def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalized Difference Vegetation Index.

    Sentinel-2: nir=B08, red=B04. Output range [-1, 1].

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        red: (H, W) float32 red reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDVI values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + eps)


def compute_evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    L: float = 1.0,
    eps: float = 1e-6
) -> np.ndarray:
    """Enhanced Vegetation Index.

    Sentinel-2: nir=B08, red=B04, blue=B02.

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        red: (H, W) float32 red reflectance.
        blue: (H, W) float32 blue reflectance.
        g: Gain factor (default 2.5).
        c1: Aerosol resistance red coefficient (default 6.0).
        c2: Aerosol resistance blue coefficient (default 7.5).
        L: Canopy background adjustment factor (default 1.0).
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 EVI values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    blue = blue.astype(np.float32)
    
    denominator = nir + (c1 * red) - (c2 * blue) + L + eps
    return g * ((nir - red) / denominator)


def compute_evi2(
    nir: np.ndarray,
    red: np.ndarray,
    g: float = 2.5,
    c: float = 2.4,
    L: float = 1.0,
    eps: float = 1e-6
) -> np.ndarray:
    """Two-Band Enhanced Vegetation Index (Does not require Blue band).

    Sentinel-2: nir=B08, red=B04. High performance over atmospheric noise.

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        red: (H, W) float32 red reflectance.
        g: Gain factor (default 2.5).
        c: Red coefficient (default 2.4).
        L: Canopy background adjustment factor (default 1.0).
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 EVI2 values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return g * ((nir - red) / (nir + (c * red) + L + eps))


def compute_savi(
    nir: np.ndarray,
    red: np.ndarray,
    L: float = 0.5,
    eps: float = 1e-6
) -> np.ndarray:
    """Soil-Adjusted Vegetation Index.

    Sentinel-2: nir=B08, red=B04.

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        red: (H, W) float32 red reflectance.
        L: Soil brightness correction factor (default 0.5).
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 SAVI values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return ((nir - red) / (nir + red + L + eps)) * (1.0 + L)


def compute_msavi2(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Modified Soil-Adjusted Vegetation Index 2 (Self-adjusting soil factor).

    Sentinel-2: nir=B08, red=B04.

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        red: (H, W) float32 red reflectance.

    Returns:
        (H, W) float32 MSAVI2 values.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    
    term = (2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red)
    # Clip negative values inside square root to 0 to prevent NaN
    term = np.clip(term, 0.0, None)
    
    return (2.0 * nir + 1.0 - np.sqrt(term)) / 2.0


def compute_ndre(
    nir: np.ndarray,
    red_edge: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Normalized Difference Red Edge Index (Chlorophyll & nitrogen sensing).

    Sentinel-2: nir=B08, red_edge=B05 (or B06/B07).

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        red_edge: (H, W) float32 red-edge reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDRE values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    red_edge = red_edge.astype(np.float32)
    return (nir - red_edge) / (nir + red_edge + eps)


# =============================================================================
# 2. SOIL, URBAN & BUILT-UP INDICES
# =============================================================================

def compute_bsi(
    swir1: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    blue: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Bare Soil Index (Differentiates bare soil from vegetation/urban).

    Sentinel-2: swir1=B11, red=B04, nir=B08, blue=B02. Output range [-1, 1].

    Args:
        swir1: (H, W) float32 short-wave infrared 1 reflectance.
        red: (H, W) float32 red reflectance.
        nir: (H, W) float32 near-infrared reflectance.
        blue: (H, W) float32 blue reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 BSI values in [-1, 1].
    """
    swir1 = swir1.astype(np.float32)
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    blue = blue.astype(np.float32)
    
    numerator = (swir1 + red) - (nir + blue)
    denominator = (swir1 + red) + (nir + blue) + eps
    return numerator / denominator


def compute_ndbi(
    swir1: np.ndarray,
    nir: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Normalized Difference Built-Up Index.

    Sentinel-2: swir1=B11, nir=B08. Output range [-1, 1].

    Args:
        swir1: (H, W) float32 short-wave infrared 1 reflectance.
        nir: (H, W) float32 near-infrared reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDBI values in [-1, 1].
    """
    swir1 = swir1.astype(np.float32)
    nir = nir.astype(np.float32)
    return (swir1 - nir) / (swir1 + nir + eps)


# =============================================================================
# 3. WATER & AQUATIC INDICES
# =============================================================================

def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters).

    Sentinel-2: green=B03, nir=B08.

    Args:
        green: (H, W) float32 green reflectance.
        nir: (H, W) float32 near-infrared reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDWI values in [-1, 1].
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    return (green - nir) / (green + nir + eps)


def compute_mndwi(
    green: np.ndarray,
    swir1: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Modified Normalized Difference Water Index (Xu - suppresses urban noise).

    Sentinel-2: green=B03, swir1=B11.

    Args:
        green: (H, W) float32 green reflectance.
        swir1: (H, W) float32 short-wave infrared 1 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 MNDWI values in [-1, 1].
    """
    green = green.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    return (green - swir1) / (green + swir1 + eps)


def compute_ndci(
    red_edge: np.ndarray,
    red: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Normalized Difference Chlorophyll Index (Water quality & algae blooms).

    Sentinel-2: red_edge=B05, red=B04. Designed specifically for water bodies.

    Args:
        red_edge: (H, W) float32 red-edge 1 reflectance.
        red: (H, W) float32 red reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDCI values in [-1, 1].
    """
    red_edge = red_edge.astype(np.float32)
    red = red.astype(np.float32)
    return (red_edge - red) / (red_edge + red + eps)


def compute_ndmi(
    nir: np.ndarray,
    swir1: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Normalized Difference Moisture Index (Canopy water content).

    Sentinel-2: nir=B08, swir1=B11.

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        swir1: (H, W) float32 short-wave infrared 1 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDMI values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    return (nir - swir1) / (nir + swir1 + eps)


# =============================================================================
# 4. FIRE & SNOW INDICES
# =============================================================================

def compute_nbr(
    nir: np.ndarray,
    swir2: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Normalized Burn Ratio.

    Sentinel-2: nir=B08, swir2=B12.

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        swir2: (H, W) float32 short-wave infrared 2 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NBR values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    swir2 = swir2.astype(np.float32)
    return (nir - swir2) / (nir + swir2 + eps)


def compute_ndsi(
    green: np.ndarray,
    swir1: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Normalized Difference Snow Index.

    Sentinel-2: green=B03, swir1=B11.

    Args:
        green: (H, W) float32 green reflectance.
        swir1: (H, W) float32 short-wave infrared 1 reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDSI values in [-1, 1].
    """
    green = green.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    return (green - swir1) / (green + swir1 + eps)
import numpy as np
import pystac

from geosave_engine.geodata.core import derive_step


@derive_step("intersect")
def intersect(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
) -> np.ndarray:
    """Logical AND of every mask.

    Reduce: yaml `bands:` order doesn't matter, just needs 2+.

    Args:
        *bands: Two or more boolean/uint8 masks.
        stac_item: unused, injected for signature consistency.
        resolution: unused, injected for signature consistency.

    Returns:
        (H, W) uint8 mask — 1 where every mask agrees.

    Raises:
        ValueError: If fewer than 2 masks are given.
    """
    if len(bands) < 2:
        raise ValueError(f"intersect needs at least 2 masks, got {len(bands)}")
    result = bands[0].astype(bool)
    for arr in bands[1:]:
        result &= arr.astype(bool)
    return result.astype(np.uint8)


@derive_step("union")
def union(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
) -> np.ndarray:
    """Logical OR of every mask.

    Reduce: yaml `bands:` order doesn't matter, just needs 2+.

    Args:
        *bands: Two or more boolean/uint8 masks.
        stac_item: unused, injected for signature consistency.
        resolution: unused, injected for signature consistency.

    Returns:
        (H, W) uint8 mask — 1 where any mask agrees.

    Raises:
        ValueError: If fewer than 2 masks are given.
    """
    if len(bands) < 2:
        raise ValueError(f"union needs at least 2 masks, got {len(bands)}")
    result = bands[0].astype(bool)
    for arr in bands[1:]:
        result |= arr.astype(bool)
    return result.astype(np.uint8)

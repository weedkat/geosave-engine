import numpy as np
import pystac
from s2cloudless import S2PixelCloudDetector

from geosave_engine.geodata.core import derive_step


@derive_step("s2c_mask")
def compute_s2c_mask(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
    prob_threshold: float = 0.4,
) -> np.ndarray:
    """Cloud probability mask via s2cloudless.

    Reduce: 10 bands in, 1 mask out. yaml `bands:` order, exactly:
    [B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12] — same order s2cloudless
    itself expects, each (H, W) float32 TOA reflectance in [0, 1].

    Args:
        *bands: The 10 Sentinel-2 L1C bands above, in that exact order.
        stac_item: unused, injected for signature consistency.
        resolution: unused, injected for signature consistency.
        prob_threshold: Probability threshold above which a pixel is flagged as cloud.

    Returns:
        (H, W) uint8 mask — 1 where cloud detected.

    Raises:
        ValueError: If not given exactly 10 bands.
    """
    if len(bands) != 10:
        raise ValueError(f"s2c_mask needs exactly 10 bands, got {len(bands)}")
    stacked = np.stack(bands, axis=-1).astype(np.float32)
    detector = S2PixelCloudDetector(threshold=prob_threshold, all_bands=False)
    return detector.get_cloud_masks(stacked[np.newaxis])[0].astype(np.uint8)  # (1, H, W)

from __future__ import annotations

from typing import Literal

import numpy as np
import pystac

from geosave_engine.geodata.core import derive_step
from geosave_engine.utils.geodata import extract_raster_scale_offset


@derive_step("apply_scale")
def apply_scale(
    *,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
    mode: Literal["from_stac", "fixed"] = "from_stac",
    scale: float | None = None,
    offset: float = 0.0,
    **bands: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply radiometric scale/offset to every band uniformly.

    Map: unknown/variable band set (whatever the upstream tile carries) —
    caught by ``**bands`` rather than positional args, since the names must
    survive the round trip into the output.

    Args:
        stac_item: Scene's STAC item — auto-injected from the tile;
            required when ``mode="from_stac"``.
        resolution: unused, injected for signature consistency.
        mode: ``"from_stac"`` reads scale/offset from ``stac_item``'s
            ``raster:bands`` metadata (e.g. Sentinel-2). ``"fixed"`` uses the
            ``scale``/``offset`` args directly (e.g. HLS ``scale=0.0001``).
        scale: Fixed scale factor. Required when ``mode="fixed"``.
        offset: Fixed offset. Only used when ``mode="fixed"``.
        **bands: Every band on the upstream tile, name -> (H, W) array.

    Returns:
        Every input band, rescaled, same names.

    Raises:
        ValueError: If ``mode="fixed"`` and ``scale`` is not given, or
            ``mode="from_stac"`` and no ``stac_item`` is available.
    """
    if mode == "from_stac":
        if stac_item is None:
            raise ValueError("apply_scale: mode='from_stac' requires STAC provenance on the tile")
        scale, offset = extract_raster_scale_offset(stac_item)
    elif scale is None:
        raise ValueError("apply_scale: mode='fixed' requires a 'scale' param")

    return {name: arr * scale + offset for name, arr in bands.items()}

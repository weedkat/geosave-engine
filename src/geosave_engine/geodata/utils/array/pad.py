"""Edge padding that keeps a DataArray's y/x coordinate grid real, not mirrored."""
from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from odc.geo import Resolution


def pad_edge(
    da: xr.DataArray, pad_y: int, pad_x: int, mode: Literal["reflect", "edge"], resolution: Resolution
) -> xr.DataArray:
    """Pad da's trailing y/x edge — pixel values padded, coordinate grid stays real, not mirrored.

    Args:
        da: Array to pad, `y`/`x` not already padded.
        pad_y: Pixels to add after da's own last row. 0 skips y entirely.
        pad_x: Pixels to add after da's own last column. 0 skips x entirely.
        resolution: da's own pixel size, e.g. `anchor.geobox.resolution` (signed — `.y` usually negative, north to south).
        mode: Passed straight to xr.DataArray.pad's own mode kwarg.

    Returns:
        da padded to (height + pad_y, width + pad_x).
    """
    # xr.DataArray.pad mirrors coord labels along with pixel values; overwrite y/x below with a plain linear extension instead.
    padded = da.pad(y=(0, pad_y), x=(0, pad_x), mode=mode)
    if pad_y:
        padded = padded.assign_coords(y=np.concatenate([da.y.values, da.y.values[-1] + resolution.y * np.arange(1, pad_y + 1)]))
    if pad_x:
        padded = padded.assign_coords(x=np.concatenate([da.x.values, da.x.values[-1] + resolution.x * np.arange(1, pad_x + 1)]))
    return padded

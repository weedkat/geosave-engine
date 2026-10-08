"""Read one window off an opened sample."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd
import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.conventions import TIME_COORDINATE
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.transform.chip import crop


def select(sample: xr.DataTree, window: Mapping[str, Any]) -> xr.DataTree:
    """Read the instants and pixels one window names.

    Args:
        sample: The whole sample the window was cut from, opened lazily.
        window: One row of `stacks`, `frames` or `chips`.

    Returns:
        Lazy stack of the same groups. Each dated group keeps only the
        window's instants for it, and every group only the window's pixels,
        filled past the sample's edge as the window states.

    Examples:
        >>> chip = select(read_stack("samples/s0"), windows.iloc[0])
        >>> dict(chip["optical"].sizes)
        {'time': 4, 'y': 224, 'x': 224}
    """
    return select_pixels(select_times(sample, window), window)


def select_times(sample: xr.DataTree, window: Mapping[str, Any]) -> xr.DataTree:
    """Read the instants one window names, every pixel kept.

    Args:
        sample: The whole sample the window was cut from, opened lazily.
        window: One row of `stacks`, `frames` or `chips`.

    Returns:
        Lazy stack whose dated groups keep only the window's instants for
        them; a group the window states no labels for is kept whole.
    """
    taken = {}
    for name, raster in sample.gs.rasters.items():
        labels = window["times"].get(name)
        if labels is not None:
            raster = raster.sel({TIME_COORDINATE: pd.DatetimeIndex(labels)})
        taken[name] = raster
    return attrs.rebase(stack(taken), sample.gs.attrs.root)


def select_pixels(sample: xr.DataTree, window: Mapping[str, Any]) -> xr.DataTree:
    """Read the pixels one window names, every instant kept.

    Args:
        sample: Stack on the grid the window was cut from: the whole sample,
            or what was prepared from one of its frames.
        window: One row of `stacks`, `frames` or `chips`.

    Returns:
        Lazy stack cropped to the window's pixels, filled past the sample's
        edge as the window states.
    """
    halo = window.get("halo")
    # A window that is not a chip states no halo, and reads no pixel past an edge.
    chip = isinstance(halo, (list, tuple)) or hasattr(halo, "tolist")
    return cast(
        "xr.DataTree",
        crop(
            sample,
            (int(window["row_off"]), int(window["col_off"])),
            (int(window["height"]), int(window["width"])),
            padding=[tuple(int(w) for w in widths) for widths in halo]
            if chip
            else ((0, 0), (0, 0)),
            mode=window["mode"] if chip else "constant",
        ),
    )

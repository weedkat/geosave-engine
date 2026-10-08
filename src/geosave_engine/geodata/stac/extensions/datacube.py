"""Datacube extension: the time labels a file or store holds."""

from __future__ import annotations

import pystac
import xarray as xr
from pystac.extensions.datacube import (
    DatacubeExtension,
    DimensionType,
    TemporalDimension,
)
from pystac.utils import datetime_to_str

from geosave_engine.geodata.conventions import TIME_COORDINATE


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write a raster's time labels as a temporal dimension.

    An Item states when its raster starts and ends; this states every label
    in between, so frames can be cut from a table without opening a store.
    A timeless raster writes nothing.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `read_raster` opens it. Only its time labels are
            read.

    Examples:
        >>> write(asset, cube)
        >>> DatacubeExtension.ext(asset).dimensions["time"].values
        ['2025-06-01T00:00:00Z', '2025-06-11T00:00:00Z']
    """
    labels = raster.gs.times
    if labels is None or labels.isna().all():
        return
    values = [datetime_to_str(label.tz_localize("UTC")) for label in labels]
    time = TemporalDimension({})
    time.dim_type = DimensionType.TEMPORAL
    time.extent = [values[0], values[-1]]
    time.values = values
    DatacubeExtension.ext(asset, add_if_missing=True).apply(
        dimensions={TIME_COORDINATE: time}
    )

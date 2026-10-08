"""Raster extension: how each band's stored numbers are typed, filled and packed."""

from __future__ import annotations

from math import isfinite

import pystac
import xarray as xr
from pystac.extensions.raster import (
    DataType,
    NoDataStrings,
    RasterBand,
    RasterExtension,
)

from geosave_engine.geodata.attrs import AttrsModel, CFVariable, Nodata, Packing


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write one Raster band per variable, in file order.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `read_raster` opens it, holding stored values. No
            pixel is read.

    Examples:
        >>> write(asset, scene)
        >>> RasterExtension.ext(asset).bands[0].scale
        0.0001
    """
    header = raster.gs.attrs
    bands = []
    for name in raster.gs.variables:
        variable = header.data_vars[name]
        nodata = variable.get(Nodata)
        packing = variable.get(Packing)
        semantics = variable.get(CFVariable)

        fill = None if nodata is None else nodata.fill_value
        if isinstance(fill, float) and not isfinite(fill):
            fill = NoDataStrings(str(fill))
        dtype = raster[name].dtype.name
        bands.append(
            RasterBand.create(
                nodata=fill,
                data_type=DataType(dtype)
                if dtype in list(DataType)
                else DataType.OTHER,
                scale=None if packing is None else packing.scale_factor,
                offset=None if packing is None else packing.add_offset,
                unit=None if semantics is None else semantics.units,
            )
        )
    RasterExtension.ext(asset, add_if_missing=True).apply(bands)


def read(band: RasterBand) -> list[AttrsModel]:
    """Read the attrs models one Raster band states.

    `nodata` and `data_type` describe the file, and the loader that opens it
    states them, so neither is read.

    Args:
        band: One band of an asset's Raster listing.

    Returns:
        `Packing` where the band states a scale or an offset, then
        `CFVariable` where it states a unit. Empty where it states neither.

    Examples:
        >>> read(RasterBand.create(unit="1", scale=0.0001))
        [Packing(scale_factor=0.0001, add_offset=None), CFVariable(units='1', ...)]
    """
    models: list[AttrsModel] = []
    if band.scale is not None or band.offset is not None:
        models.append(Packing(scale_factor=band.scale, add_offset=band.offset))
    if band.unit is not None:
        models.append(CFVariable(units=band.unit))
    return models

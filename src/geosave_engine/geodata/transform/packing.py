"""Reading physical values out of stored digital numbers.

Absence is `Nodata`, which packing never touches — `nodata.decode` first, then
this, so a stored fill value is never scaled into a bogus physical reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import xarray as xr
from xarray.coding.variables import CFScaleOffsetCoder

import geosave_engine.geodata.attrs as attrs

if TYPE_CHECKING:
    from geosave_engine.geodata import DataArray, Dataset


@overload
def decode(data: xr.DataArray) -> DataArray: ...


@overload
def decode(data: xr.Dataset) -> Dataset: ...


def decode(data: xr.DataArray | xr.Dataset) -> DataArray | Dataset:
    """Read physical values out of each variable's stored digital numbers.

    A variable carrying neither `scale_factor` nor `add_offset` passes through
    unchanged.

    Args:
        data: DataArray or Dataset holding stored values.

    Returns:
        New object of the same kind holding physical values, each decoded
        variable's packing dropped from attrs since its values are no longer
        the stored numbers packing described.

    Examples:
        >>> decode(scene).red.max().item()
        0.09
    """
    if isinstance(data, xr.DataArray):
        return cast("DataArray", _decode_array(data))

    decoded = {
        variable: _decode_array(data[variable]) for variable in data.gs.variables
    }
    return cast("Dataset", xr.Dataset(decoded, attrs=dict(data.attrs)))


def _decode_array(array: xr.DataArray) -> xr.DataArray:
    """Read physical values out of one variable's stored digital numbers.

    Args:
        array: Variable holding stored values.

    Returns:
        New DataArray of physical values, or `array` unchanged where it
        carries neither `scale_factor` nor `add_offset`.
    """
    packing = array.gs.attrs.root.get(attrs.Packing)
    if packing is None or (packing.scale_factor is None and packing.add_offset is None):
        return array

    # The coder pops what it reads, so it gets a copy, not the source.
    decoded = CFScaleOffsetCoder().decode(
        array.variable.copy(deep=False), name=array.name
    )
    return xr.DataArray(
        decoded, coords=array.coords, name=array.name, attrs=decoded.attrs
    )

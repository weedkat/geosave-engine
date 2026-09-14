"""Reading physical values out of stored digital numbers.

Absence is `Nodata`, which packing never touches — `nodata.to_nan` first, then
this, so a stored fill value is never scaled into a bogus physical reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import xarray as xr
from xarray.coding.variables import CFScaleOffsetCoder

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import nodata

if TYPE_CHECKING:
    from geosave_engine.geodata import DataArray, Dataset


@overload
def unpack(data: xr.DataArray) -> DataArray: ...


@overload
def unpack(data: xr.Dataset) -> Dataset: ...


def unpack(data: xr.DataArray | xr.Dataset) -> DataArray | Dataset:
    """Read physical values out of each variable's stored digital numbers.

    A variable carrying neither `scale_factor` nor `add_offset` passes through
    unchanged.

    Args:
        data: DataArray or Dataset holding stored values.

    Returns:
        New object of the same kind holding physical values, each unpacked
        variable's packing dropped from attrs since its values are no longer
        the stored numbers packing described.

    Raises:
        ValueError: A packed variable still marks its nodata pixels with a
            fill value, which scaling would read as an ordinary value.

    Examples:
        >>> unpack(scene).red.max().item()
        0.09
    """
    if isinstance(data, xr.DataArray):
        return cast("DataArray", _unpack_array(data))

    physical = {
        variable: _unpack_array(data[variable]) for variable in data.gs.variables
    }
    return cast("Dataset", xr.Dataset(physical, attrs=dict(data.attrs)))


def _unpack_array(array: xr.DataArray) -> xr.DataArray:
    """Read physical values out of one variable's stored digital numbers.

    Args:
        array: Variable holding stored values.

    Returns:
        New DataArray of physical values, or `array` unchanged where it
        carries neither `scale_factor` nor `add_offset`.

    Raises:
        ValueError: `array` still marks its nodata pixels with a fill value.
    """
    packing = array.gs.attrs.root.get(attrs.Packing)
    if packing is None or (packing.scale_factor is None and packing.add_offset is None):
        return array

    fill = nodata.fill_value(array)
    if fill is not None:
        raise ValueError(
            f"{str(array.name)!r} marks its nodata pixels with {fill!r}, which "
            f"scaling turns into an ordinary reading nothing can tell from data; "
            f"blank them with nodata.to_nan first, then unpack"
        )

    # xarray copies the attrs it reads today; the copy keeps that promise ours.
    physical = CFScaleOffsetCoder().decode(
        array.variable.copy(deep=False), name=array.name
    )
    return xr.DataArray(
        physical, coords=array.coords, name=array.name, attrs=physical.attrs
    )

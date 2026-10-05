"""Reading physical values out of stored digital numbers.

Absence is `Nodata`, which packing never touches — `nodata.to_nan` first, then
this, so a stored fill value is never scaled into a bogus physical reading.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import xarray as xr
from xarray.coding.variables import CFScaleOffsetCoder

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import nodata


def unpack[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Read physical values out of each variable's stored digital numbers.

    A variable carrying neither `scale_factor` nor `add_offset` passes through
    unchanged. A stack is unpacked group by group.

    Args:
        data: DataArray, Dataset, or DataTree holding stored values.

    Returns:
        New object of the same kind holding physical values, float32 where
        the stored integers are 16 bits or fewer and float64 otherwise. Each
        unpacked variable's packing is dropped from attrs, since its values
        are no longer the stored numbers packing described.

    Raises:
        ValueError: A packed variable still marks its nodata pixels with a
            fill value, which scaling would read as an ordinary value.

    Examples:
        >>> unpack(scene).red.max().item()
        0.09
    """
    if isinstance(data, xr.DataTree):
        return cast("T", data.map_over_datasets(lambda dataset: unpack(dataset)))

    if isinstance(data, xr.DataArray):
        return cast("T", _unpack_array(data))

    physical = {
        variable: _unpack_array(data[variable]) for variable in data.gs.variables
    }
    return cast("T", nodata.drop_source(data.assign(physical)))


def _unpack_array(array: xr.DataArray) -> xr.DataArray:
    """Read physical values out of one variable's stored digital numbers.

    Args:
        array: Variable holding stored values.

    Returns:
        New DataArray of physical values, or `array` unchanged where it
        carries neither `scale_factor` nor `add_offset`.

    Raises:
        ValueError: `array` holds class codes, or still marks its nodata pixels
            with a fill value.
    """
    packing = attrs.Packing.from_attrs(array.attrs)
    if packing is None or (packing.scale_factor is None and packing.add_offset is None):
        return array

    # A class code names a class, so scaling it yields a number naming none.
    if attrs.flag_variables(array):
        raise ValueError(
            f"{str(array.name)!r} holds class codes and declares packing, which "
            f"cannot both be true; drop its Legend or its Packing, whichever "
            f"does not describe the pixels"
        )

    fill = nodata.fill_value(array)
    if fill is not None:
        raise ValueError(
            f"{str(array.name)!r} marks its nodata pixels with {fill!r}, which "
            f"scaling turns into an ordinary reading nothing can tell from data; "
            f"blank them with nodata.to_nan first, then unpack"
        )

    # xarray copies the attrs it reads today; the copy keeps that promise ours.
    stored = array.variable.copy(deep=False)
    if array.dtype.kind in "iu" and array.dtype.itemsize <= 2:
        # CF reads a 16-bit integer out as float32 only where its packing is
        # float32 too, and a packing read off disk arrives as a Python float.
        for key in ("scale_factor", "add_offset"):
            if key in stored.attrs:
                stored.attrs[key] = np.float32(stored.attrs[key])
    physical = CFScaleOffsetCoder().decode(stored, name=array.name)
    return xr.DataArray(
        physical, coords=array.coords, name=array.name, attrs=physical.attrs
    )

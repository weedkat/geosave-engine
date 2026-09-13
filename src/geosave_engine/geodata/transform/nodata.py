"""Marking pixels nodata.

A pixel's nodata is `Nodata.fill_value`, never an implicit NaN. Filling
nodata pixels back in is `composite.mosaic`, which is the same operation as
mosaicking.

Examples:
    A scene classification band says which pixels are cloud::

        clear = mask(scene, scene.scl.isin(CLEAR_CLASSES))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr
from xarray.coding.variables import CFMaskCoder

import geosave_engine.geodata.attrs as attrs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from geosave_engine.geodata import DataArray, Dataset


@overload
def decode(data: xr.DataArray) -> DataArray: ...


@overload
def decode(data: xr.Dataset) -> Dataset: ...


def decode(data: xr.DataArray | xr.Dataset) -> DataArray | Dataset:
    """Replace each variable's fill value with NaN.

    A reduction reads a stored fill value as data unless it finds NaN there
    instead. A variable carrying no fill value passes through unchanged.

    Args:
        data: DataArray or Dataset holding stored values.

    Returns:
        New object of the same kind holding NaN where the pixels were nodata.
        The fill value leaves attrs with them, since no pixel holds it now.

    Examples:
        >>> decode(scene).red.dtype
        dtype('float64')
        >>> decode(scene).red.attrs
        {}
    """
    if isinstance(data, xr.DataArray):
        return cast("DataArray", _decode_array(data))

    decoded = {
        variable: _decode_array(data[variable]) for variable in data.gs.variables
    }
    return cast("Dataset", xr.Dataset(decoded, attrs=dict(data.attrs)))


def _decode_array(array: xr.DataArray) -> xr.DataArray:
    """Replace one variable's fill value with NaN.

    Args:
        array: Variable holding stored values.

    Returns:
        New DataArray holding NaN where the pixels were nodata, its fill value
        gone from attrs, or `array` unchanged where it carries none.
    """
    nodata = array.gs.attrs.root.get(attrs.Nodata)
    if nodata is None or nodata.fill_value is None:
        return array

    # The coder pops what it reads, so it gets a copy, not the source.
    decoded = CFMaskCoder().decode(array.variable.copy(deep=False), name=array.name)
    # The coder knows CF's spelling of nodata, not odc's.
    decoded_attrs = {
        key: value for key, value in decoded.attrs.items() if key != "nodata"
    }
    return xr.DataArray(
        decoded, coords=array.coords, name=array.name, attrs=decoded_attrs
    )


def mask[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    valid: xr.DataArray | np.ndarray,
    *,
    fill: float | int | None = None,
) -> T:
    """Make nodata every pixel a boolean mask does not keep.

    Blanked pixels take the fill value their own variable carries, so no dtype
    is promoted and no variable gains a NaN it never carried.

    Args:
        data: DataArray, Dataset, or DataTree to mask.
        valid: Boolean array, True where a pixel is real data. A DataArray
            names its own axes and may span fewer than `data`, broadcasting
            over the rest; a bare numpy array names none, so it is read as the
            grid alone and must match its shape.
        fill: Value the blanked pixels take, also written onto the result for
            variables carrying no fill value yet. None reads what each
            variable already carries.

    Returns:
        New object of the same kind, its unkept pixels holding the fill value
        their variable carries.

    Raises:
        TypeError: `data` is not an xarray object this masks.
        ValueError: `valid` is not boolean, does not span the grid, spans an
            axis `data` does not, `fill` does not fit a variable's dtype, or a
            variable carries no fill value and none is given.

    Examples:
        Sentinel-2 ships a scene classification band naming what each pixel is;
        keep only the classes that are actually ground:

        >>> clear = mask(scene, scene.scl.isin([4, 5, 6, 7]))
        >>> int((clear.red == 0).sum()) > 0
        True

        A variable carrying no fill value is refused rather than promoted, so
        say what nodata looks like and it travels onto the result:

        >>> mask(unpacked, valid, fill=255).red.attrs["_FillValue"]
        255
    """
    if not isinstance(data, xr.DataArray | xr.Dataset | xr.DataTree):
        raise TypeError(
            f"masking takes a DataArray, Dataset, or DataTree, got "
            f"{type(data).__name__}"
        )

    if valid.dtype != bool:
        raise ValueError(
            f"valid is {valid.dtype}, not boolean; pass a comparison such as "
            f"scl.isin([...]) rather than the values themselves"
        )

    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import stack

        return cast(
            "T",
            stack(
                {
                    name: mask(raster, valid, fill=fill)
                    for name, raster in data.gs.rasters.items()
                }
            ),
        )

    spatial = cast("xr.DataArray | xr.Dataset", data)
    grid_dims = spatial.gs.grid_dims
    if isinstance(valid, np.ndarray):
        grid_shape = tuple(spatial.sizes[dim] for dim in grid_dims)
        if valid.shape != grid_shape:
            raise ValueError(
                f"valid is shaped {valid.shape} but the grid {list(grid_dims)} is "
                f"{grid_shape}; a bare array names no axes, so pass a DataArray "
                f"to mask along anything else"
            )
        valid = xr.DataArray(valid, dims=grid_dims)

    if not set(grid_dims) <= set(valid.dims):
        raise ValueError(
            f"valid spans {list(valid.dims)}, which does not cover the grid "
            f"{list(grid_dims)}; a mask names which pixels survive, so it "
            f"spans at least the two spatial axes"
        )
    stray = sorted(str(dim) for dim in valid.dims if dim not in spatial.dims)
    if stray:
        raise ValueError(
            f"valid spans {stray}, which {type(data).__name__} does not carry, so "
            f"those axes name no pixels to mask; select them off valid first"
        )

    if isinstance(data, xr.DataArray):
        band = data if fill is None else data.gs.write_nodata(fill)
        return cast("T", band.where(valid, _written_fill(band)))

    # Writing it first means every variable below reads one spelling of nodata.
    raster = cast("xr.Dataset", data if fill is None else data.gs.write_nodata(fill))
    masked = {
        str(name): array.where(valid, _written_fill(array))
        for name, array in raster.data_vars.items()
    }
    blanked = xr.Dataset(masked, attrs=dict(raster.attrs))
    return cast("T", blanked.assign_coords(raster.coords))


def fill_value(array: xr.DataArray) -> float | int | None:
    """Read the value a variable's nodata pixels hold.

    Args:
        array: Data variable that may carry a fill value.

    Returns:
        The variable's `Nodata.fill_value`, or None where it carries none.
    """
    nodata = array.gs.attrs.root.get(attrs.Nodata)
    if nodata is None or nodata.fill_value is None:
        return None
    return nodata.fill_value


def required_fill_value(array: xr.DataArray) -> float | int:
    """Read the value a variable's nodata pixels hold, refusing one with none.

    Args:
        array: Data variable that must carry a fill value.

    Returns:
        The variable's `Nodata.fill_value`.

    Raises:
        ValueError: The variable carries none, so a pixel no raster covers
            has nothing to hold.
    """
    fill = fill_value(array)
    if fill is None:
        raise ValueError(
            f"{str(array.name)!r} carries no fill value, so a pixel no raster "
            f"covers has nothing to hold; write one with "
            f"raster.gs.write_nodata(...) first"
        )
    return fill


def required_fill_values(
    reference: xr.Dataset, var_names: Iterable[str] | None = None
) -> dict[str, float | int]:
    """Read every named variable's fill value off one raster.

    Args:
        reference: Raster carrying every named variable's fill value.
        var_names: Data variable names to read, each of `reference`'s own.
            None reads every variable `reference` carries.

    Returns:
        Variable name mapped to its `Nodata.fill_value`.

    Raises:
        ValueError: `var_names` names a variable `reference` does not carry,
            or a named variable carries no fill value, so a pixel no raster
            covers has nothing to hold.
    """
    if var_names is None:
        names = reference.gs.variables
    else:
        names = set(var_names)
        missing = names - reference.gs.variables
        if missing:
            raise ValueError(
                f"{sorted(missing)} are not in the raster's variables "
                f"{sorted(reference.gs.variables)}"
            )
    return {name: required_fill_value(reference[name]) for name in names}


def _written_fill(array: xr.DataArray) -> float | int:
    """Read the value blanked pixels take, refusing a variable that has none.

    Args:
        array: Data variable being masked.

    Returns:
        The variable's `Nodata.fill_value`.

    Raises:
        ValueError: The variable carries none, so masking has nothing to
            write into the pixels it blanks.
    """
    fill = fill_value(array)
    if fill is None:
        raise ValueError(
            f"{str(array.name)!r} carries no fill value, so masking has nothing "
            f"to write into the pixels it blanks; pass fill= or write one with "
            f"raster.gs.write_nodata(...) first"
        )
    return fill


def is_fill(array: xr.DataArray) -> xr.DataArray:
    """Mark the pixels holding a variable's fill value.

    `nan == nan` is never true, so a NaN fill is matched through inequality
    with itself instead.

    Args:
        array: Variable whose nodata pixels are being tested.

    Returns:
        Boolean array, True where a pixel is nodata.

    Raises:
        ValueError: The variable carries no fill value, so nodata has no
            spelling to test against.

    Examples:
        >>> int(is_fill(scene.red).sum())
        4096
    """
    fill = fill_value(array)
    if fill is None:
        raise ValueError(
            f"{str(array.name)!r} carries no fill value, so nodata has no "
            f"spelling to test against; write one before mosaicking"
        )
    return array != array if fill != fill else array == fill

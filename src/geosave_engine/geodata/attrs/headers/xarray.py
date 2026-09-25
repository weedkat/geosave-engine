"""Create an attrs header from an xarray object."""

from __future__ import annotations

import xarray as xr

from ..header import AttrsHeader

type XarrayObject = xr.Dataset | xr.DataArray | xr.DataTree


def create_header(obj: XarrayObject) -> AttrsHeader:
    """Create a detached header from an xarray object's attrs.

    A DataTree node is captured on its own, without traversing its children.

    Args:
        obj: Dataset, DataArray, or DataTree node carrying the attrs.

    Returns:
        Header holding the typed models and foreign attrs from every mapping.

    Raises:
        ValidationError: A value does not satisfy the field that owns its key.

    Examples:
        >>> create_header(ds).data_vars["B04"].to_attrs()
        {'units': '1', 'long_name': 'Red', '_FillValue': 0, 'nodata': 0}
    """
    variables = obj.coords.variables if isinstance(obj, xr.DataArray) else obj.variables
    data_vars = () if isinstance(obj, xr.DataArray) else obj.data_vars
    return AttrsHeader.from_attrs(
        root=obj.attrs,
        data_vars={str(name): variables[name].attrs for name in sorted(data_vars)},
        coords={str(name): variables[name].attrs for name in sorted(obj.coords)},
    )

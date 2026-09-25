"""Create an attrs header from an open GDAL raster."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from rasterio.enums import ColorInterp, MaskFlags

from geosave_engine.geodata.errors import UnreadMaskWarning

from ..header import AttrsHeader
from ..models import GDALVariable, Legend
from ..namespace import AttrsNamespace

if TYPE_CHECKING:
    import rasterio


def create_header(src: rasterio.DatasetReader) -> AttrsHeader:
    """Create a header from GDAL tags and native band properties.

    Args:
        src: Open raster.

    Returns:
        Header whose data variables stand in band order, named by their
        `GDALVariable.variable_name` or by rasterio's positional band name.

    Raises:
        ValueError: Two bands name one variable.

    Warns:
        UnreadMaskWarning: A band records absence in a mask that is not read.
    """
    with_mask_band = [
        index
        for index, flags in zip(src.indexes, src.mask_flag_enums, strict=True)
        if MaskFlags.per_dataset in flags
    ]
    if with_mask_band:
        warnings.warn(
            f"bands {with_mask_band} mark their absent pixels in a mask band, "
            f"which this reader does not read, so every pixel reads as present; "
            f"give them a fill value, or read the mask with rasterio",
            UnreadMaskWarning,
            stacklevel=2,
        )

    data_vars: dict[str, dict[str, Any]] = {}
    for index in src.indexes:
        position = index - 1
        carried: dict[str, Any] = {
            key: value
            for key, value in src.tags(index).items()
            if not key.startswith("STATISTICS_")
        }
        scale, offset = src.scales[position], src.offsets[position]
        interpretation = src.colorinterp[position]
        native: dict[str, Any] = {
            "long_name": src.descriptions[position],
            "units": src.units[position],
            "scale_factor": None if scale == 1.0 else scale,
            "add_offset": None if offset == 0.0 else offset,
            "_FillValue": src.nodatavals[position],
            "colorinterp": interpretation,
        }
        carried.update(
            {key: value for key, value in native.items() if value is not None}
        )

        listed = AttrsNamespace.from_attrs(carried).get(Legend)
        if interpretation is ColorInterp.palette and listed and listed.flag_values:
            colours = src.colormap(index)
            carried["color_map"] = {
                value: colours[value][:3]
                for value in listed.flag_values
                if value in colours
            }

        identity = AttrsNamespace.from_attrs(carried).get(GDALVariable)
        name = (
            identity.variable_name
            if identity is not None and identity.variable_name is not None
            else f"band_{index}"
        )
        if name in data_vars:
            raise ValueError(
                f"band {index} names variable {name!r}, which an earlier band "
                f"already names; one file names each variable once"
            )
        data_vars[name] = carried

    return AttrsHeader.from_attrs(root=src.tags(), data_vars=data_vars)

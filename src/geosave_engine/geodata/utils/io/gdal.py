"""Read one GDAL-supported raster file into a Dataset of its bands.

A band is one variable: GDAL writes its name and colour interpretation in the
band's own metadata, which `GDALVariable` reads, and its `long_name` in the band
description, which stays CF's. ``TIFFTAG_DATETIME`` becomes a scalar `time`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr
from rasterio.enums import ColorInterp

from geosave_engine.geodata.attrs import (
    GDALVariable,
    GeoTIFFTags,
    read as read_attrs,
    rebase,
)

from geosave_engine.geodata.core.convention import TIME_COORDINATE

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset


class RasterioOpenOptions(TypedDict, total=False):
    """Optional rioxarray behavior supported when reading one raster."""

    parse_coordinates: bool | None
    cache: bool | None
    lock: Any
    decode_times: bool
    decode_timedelta: bool | None
    overview_level: int


def read(
    source: str | PathLike[str],
    *,
    chunks: Any = None,
    mask_and_scale: bool = False,
    **open_options: Unpack[RasterioOpenOptions],
) -> Dataset:
    """Read any GDAL-readable raster as one variable per band.

    A band naming itself in `GDALVariable.variable_name` becomes a variable of
    that name; the rest keep rasterio's `band_1`, `band_2`. A band's other
    metadata stays on its variable, the description among it as `long_name`.

    Args:
        source: Local path or URI to a raster GDAL can open.
        chunks: Chunk configuration for the opened variables.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        **open_options: Supported rioxarray and rasterio open options.

    Returns:
        Dataset holding one `(y, x)` variable per band.

    Raises:
        ValueError: The file cannot be read, holds subdatasets, or names one
            variable on two bands.

    Examples:
        >>> read("scene.tif").gs.variables
        ('B04', 'B08')
    """
    with rasterio.open(source) as src:
        # rioxarray reads no colour interpretation, so it is taken here in band order.
        band_colorinterp = tuple(band.name for band in src.colorinterp)
        cube = rioxarray.open_rasterio(
            src,
            band_as_variable=True,
            chunks=chunks,
            mask_and_scale=mask_and_scale,
            **open_options,
        )

    if isinstance(cube, list):
        raise ValueError(
            f"{source} holds subdatasets; open one of them by its own URI instead"
        )
    if not isinstance(cube, xr.Dataset):
        raise ValueError(
            f"{source} opened as a banded array rather than one variable per "
            f"band; open it with the reader for its own format"
        )

    for var_name, colorinterp in zip(cube.data_vars, band_colorinterp, strict=True):
        if colorinterp != ColorInterp.undefined.name:
            cube = rebase(
                cube, GDALVariable(colorinterp=colorinterp), target=str(var_name)
            )

    # Renaming leaves the root alone, so one read serves both the bands and tags.
    header = read_attrs(cube)
    rename_map: dict[str, str] = {}
    for var_name, namespace in header.data_vars.items():
        identity = namespace.get(GDALVariable)
        if identity is not None and identity.variable_name is not None:
            rename_map[var_name] = identity.variable_name

    # rename_vars refuses two bands naming one variable.
    cube = cube.rename_vars(rename_map)

    tags = header.root.get(GeoTIFFTags)
    if tags is not None and tags.TIFFTAG_DATETIME is not None:
        cube = cube.assign_coords({TIME_COORDINATE: tags.TIFFTAG_DATETIME})

    return cast("Dataset", cube)

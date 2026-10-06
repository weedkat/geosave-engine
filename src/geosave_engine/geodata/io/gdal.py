"""Read one GDAL-supported raster file into a Dataset of its bands.

A band is one variable, named by the `GDALVariable.variable_name` it carries.
The attrs GDAL header factory reads tags and native band properties alike,
leaving rioxarray the pixels and the grid.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import numpy as np
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr

from geosave_engine.geodata.attrs import (
    GeoTIFFTags,
    rebase,
)
from geosave_engine.geodata.attrs.headers.gdal import create_header

from geosave_engine.geodata.conventions import TIME_COORDINATE
from geosave_engine.geodata.utils.datetime import parse_stem_dates
from .storage import absolute_location, gdal_path

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

    A band naming itself in `GDALVariable.variable_name` becomes that variable,
    spending the tag; the rest keep rasterio's `band_1`, `band_2`. Its metadata
    stays on the variable, the description as `long_name` unless it is the name.

    Args:
        source: Local path or URI to a raster GDAL can open. An
            `hf://buckets/` URL is read through the bucket's S3 gateway.
        chunks: Chunk configuration for the opened variables.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        **open_options: Supported rioxarray and rasterio open options.

    Returns:
        Dataset holding one `(y, x)` variable per band, carrying a scalar
        `time` where the file dates itself. ``GEOSAVE_DATETIME`` names the
        precise instant; without it ``TIFFTAG_DATETIME`` supplies whole
        seconds, then the filename stem supplies a fallback. A stem naming a
        whole period places the raster at its first instant.

    Raises:
        RasterioIOError: The file cannot be opened or read.
        ValueError: The file holds subdatasets or names one variable on two bands.

    Examples:
        >>> read("scene.tif").gs.variables
        ('B04', 'B08')
        >>> read("20190507_red.tif").time.item()
        datetime.datetime(2019, 5, 7, 0, 0)
    """
    with rasterio.open(gdal_path(source)) as src:
        header = create_header(src)
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

    opened = cube
    try:
        # rioxarray names the bands by position; the header names them as they name themselves.
        cube = cube.rename_vars(
            dict(zip(cube.data_vars, header.data_vars, strict=True))
        )
        cube = rebase(cube, header)
        if mask_and_scale:
            # Decoded pixels are physical, so the file's packing lives in encoding alone.
            cube = rebase(cube, target=cube.gs.variables, packing=None, nodata=None)

        tags = header.root.get(GeoTIFFTags)
        if tags is not None and tags.GEOSAVE_DATETIME is not None:
            cube = cube.assign_coords(
                {TIME_COORDINATE: np.datetime64(tags.GEOSAVE_DATETIME)}
            )
        elif tags is not None and tags.TIFFTAG_DATETIME is not None:
            cube = cube.assign_coords({TIME_COORDINATE: tags.TIFFTAG_DATETIME})
        else:
            # A file dating itself only in its name still places itself in time.
            timespan = parse_stem_dates(PurePath(str(source)).stem)
            if timespan is not None:
                cube = cube.assign_coords({TIME_COORDINATE: timespan[0]})
    except BaseException:
        opened.close()
        raise

    # Xarray's derived Datasets do not retain the backend closer they still use.
    if cube is not opened:
        cube.set_close(opened.close)

    cube.encoding["source"] = absolute_location(source)
    return cast("Dataset", cube)

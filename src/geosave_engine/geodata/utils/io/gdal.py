"""Read one GDAL-supported raster file into a banded DataArray."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr

from geosave_engine.geodata.core.array import BAND_DIMENSION

from geosave_engine.geodata.attrs import GeoTIFFTags, read as read_attrs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from geosave_engine.geodata import DataArray


_BAND_DESCRIPTIONS = "long_name"


class RasterioOpenOptions(TypedDict, total=False):
    """Optional rioxarray behavior supported when reading one raster."""

    parse_coordinates: bool | None
    cache: bool | None
    lock: Any
    decode_times: bool
    decode_timedelta: bool | None
    rasterio_open_options: dict[str, Any]


def read(
    source: str | PathLike[str],
    *,
    band_names: Sequence[str] | None = None,
    overview_level: int | None = None,
    chunks: Any = None,
    mask_and_scale: bool = False,
    **open_options: Unpack[RasterioOpenOptions],
) -> DataArray:
    """Read any GDAL-readable raster as one `(band, y, x)` array.

    The `band` coordinate holds the file's GDAL band descriptions where it
    names them, and rasterio's indexes where it does not. Tags stay in the
    array's attrs, ``TIFFTAG_DATETIME`` also becoming `time`.

    Args:
        source: Local path or URI to a raster GDAL can open.
        band_names: Names replacing the file's own descriptions, in band
            order.
        overview_level: Zero-based overview level, or None for full
            resolution.
        chunks: Chunk configuration for the opened array.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        **open_options: Supported rioxarray and rasterio open options.

    Returns:
        Array shaped `(band, y, x)`.

    Raises:
        ValueError: The file cannot be read, holds subdatasets, or
            `band_names` does not match the band count.

    Examples:
        >>> read("scene.tif").band.values
        array(['B04', 'B08'], dtype=object)
    """
    options: dict[str, Any] = dict(open_options)
    if overview_level is not None:
        options["overview_level"] = overview_level

    opened = rioxarray.open_rasterio(
        source, chunks=chunks, mask_and_scale=mask_and_scale, **options
    )
    if isinstance(opened, list):
        raise ValueError(
            f"{source} holds subdatasets; open one of them by its own URI instead"
        )
    if isinstance(opened, xr.Dataset):
        raise ValueError(
            f"{source} opened as a Dataset rather than a banded raster; open it "
            f"with the reader for its own format"
        )

    count = opened.sizes.get(BAND_DIMENSION)
    if band_names is not None:
        if len(band_names) != count:
            raise ValueError(
                f"band_names names {len(band_names)} bands but {source} carries "
                f"{count}; name every band exactly once, in band order"
            )
        opened = opened.assign_coords(
            {BAND_DIMENSION: [str(name) for name in band_names]}
        )
        opened.attrs.pop(_BAND_DESCRIPTIONS, None)
    else:
        # A file naming every band labels them; the rest keep rasterio's indexes.
        described = opened.attrs.get(_BAND_DESCRIPTIONS)
        descriptions = (described,) if isinstance(described, str) else described or ()
        if count is not None and len(descriptions) == count and all(descriptions):
            opened = opened.assign_coords(
                {BAND_DIMENSION: [str(name) for name in descriptions]}
            )
            opened.attrs.pop(_BAND_DESCRIPTIONS, None)

    # The tag model reads the instant GDAL spells its own way.
    tags = read_attrs(opened).root.get(GeoTIFFTags)
    if isinstance(tags, GeoTIFFTags) and tags.TIFFTAG_DATETIME is not None:
        opened = opened.assign_coords(time=tags.TIFFTAG_DATETIME)

    return cast("DataArray", opened)

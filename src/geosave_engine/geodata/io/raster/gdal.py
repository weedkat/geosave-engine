"""Everything GDAL-specific: its environment, its paths, and reading one file.

`configure_gdal` sets the process environment remote reads use. `read` opens
one GDAL-supported raster file as a Dataset of its bands: a band is one
variable, named by the `GDALVariable.variable_name` it carries. The attrs GDAL
header factory reads tags and native band properties alike, leaving rioxarray
the pixels and the grid.
"""

from __future__ import annotations

import os
from enum import Enum, auto
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import numpy as np
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
import xarray as xr
from fsspec.core import split_protocol

from geosave_engine.geodata.attrs import (
    GeoTIFFTags,
    rebase,
)
from geosave_engine.geodata.attrs.headers.gdal import create_header

from geosave_engine.geodata.conventions import TIME_COORDINATE
from geosave_engine.geodata.utils.datetime import parse_stem_dates
from ..storage import absolute_location

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
    with rasterio.open(hf_to_s3(source)) as src:
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


def hf_to_s3(location: str | PathLike[str]) -> str:
    """Spell a Hugging Face bucket URL as the S3 URL GDAL opens.

    GDAL reads local paths and `s3://`, `gs://`, `az://` and `https://` URLs
    itself. A Hugging Face bucket is reached through its S3 gateway, which
    needs S3 keys and `configure_gdal` pointed at `s3.hf.co` beforehand.

    Args:
        location: Local path or URL of a raster file.

    Returns:
        `s3://<namespace>/<bucket>/<key>` for `hf://buckets/<namespace>/<bucket>/<key>`,
        else the location unchanged.

    Raises:
        ValueError: An `hf://` URL names something other than a bucket.

    Examples:
        >>> hf_to_s3("hf://buckets/me/samples/forest/a.tif")
        's3://me/samples/forest/a.tif'
    """
    protocol, path = split_protocol(str(location))
    if protocol != "hf":
        return str(location)
    if not path.startswith("buckets/"):
        raise ValueError(
            f"{location} is not a Hugging Face bucket; only hf://buckets/ URLs "
            f"are served by the S3 gateway GDAL reads through"
        )
    return f"s3://{path.removeprefix('buckets/')}"


class _Unset(Enum):
    """Sentinel type for a keyword argument that was not passed.

    Distinct from an explicit None, and single-member so that `is not _UNSET`
    narrows a `X | None | _Unset` union down to `X | None`.
    """

    TOKEN = auto()


_UNSET = _Unset.TOKEN


def _bool_env(value: bool | _Unset) -> str | _Unset:
    return value if value is _UNSET else ("TRUE" if value else "FALSE")


def configure_gdal(
    *,
    aws_no_sign_request: bool | _Unset = _UNSET,
    aws_access_key_id: str | _Unset = _UNSET,
    aws_secret_access_key: str | _Unset = _UNSET,
    aws_session_token: str | _Unset = _UNSET,
    aws_default_region: str | _Unset = _UNSET,
    aws_s3_endpoint: str | _Unset = _UNSET,
    aws_virtual_hosting: bool | _Unset = _UNSET,
    aws_https: bool | _Unset = _UNSET,
    gdal_disable_readdir_on_open: bool | Literal["EMPTY_DIR"] | _Unset = _UNSET,
    gdal_http_max_retry: int | _Unset = _UNSET,
    gdal_http_retry_delay: float | _Unset = _UNSET,
    gdal_http_merge_consecutive_ranges: bool | _Unset = _UNSET,
    gdal_num_threads: int | Literal["ALL_CPUS"] | _Unset = _UNSET,
    cpl_vsil_curl_allowed_extensions: list[str] | _Unset = _UNSET,
    vsi_cache: bool | _Unset = _UNSET,
    vsi_cache_size: int | _Unset = _UNSET,
    omp_num_threads: int | _Unset = _UNSET,
) -> None:
    """Set the environment used by remote GDAL-backed raster reads.

    This is the single configuration adapter for rasterio and odc-stac
    reads. Settings are process-wide, so call it once during application
    startup before creating lazy reads.

    Args:
        aws_no_sign_request: Skip request signing for public AWS buckets.
        aws_access_key_id: AWS access key.
        aws_secret_access_key: AWS secret key.
        aws_session_token: Temporary AWS session token.
        aws_default_region: Default AWS region.
        aws_s3_endpoint: Host of an S3-compatible service, such as
            ``s3.hf.co`` for Hugging Face buckets.
        aws_virtual_hosting: False addresses buckets as path segments, which
            S3-compatible gateways need.
        aws_https: False reaches an S3-compatible service over plain HTTP,
            as a local one usually is.
        gdal_disable_readdir_on_open: True skips listing a remote folder
            before opening a file in it. `"EMPTY_DIR"` also skips probing
            for sidecar files, which makes remote COG reads far faster but
            hides `.aux.xml` statistics and `.ovr` overviews, local ones too.
        gdal_http_max_retry: Maximum HTTP retry attempts.
        gdal_http_retry_delay: Delay between HTTP retries in seconds.
        gdal_http_merge_consecutive_ranges: Merge adjacent byte-range reads.
        gdal_num_threads: GDAL worker count or ``"ALL_CPUS"``.
        cpl_vsil_curl_allowed_extensions: Remote file extensions GDAL may
            inspect.
        vsi_cache: Enable GDAL's in-memory VSI cache.
        vsi_cache_size: VSI cache size in bytes.
        omp_num_threads: OpenMP worker count for linked codecs.

    Examples:
        >>> configure_gdal(
        ...     aws_no_sign_request=True,
        ...     gdal_disable_readdir_on_open="EMPTY_DIR",
        ...     gdal_http_max_retry=3,
        ... )
    """
    values: dict[str, str | _Unset] = {
        "AWS_NO_SIGN_REQUEST": _bool_env(aws_no_sign_request),
        "AWS_ACCESS_KEY_ID": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        "AWS_SESSION_TOKEN": aws_session_token,
        "AWS_DEFAULT_REGION": aws_default_region,
        "AWS_S3_ENDPOINT": aws_s3_endpoint,
        "AWS_VIRTUAL_HOSTING": _bool_env(aws_virtual_hosting),
        # GDAL spells this one YES or NO.
        "AWS_HTTPS": aws_https
        if aws_https is _UNSET
        else ("YES" if aws_https else "NO"),
        "GDAL_DISABLE_READDIR_ON_OPEN": (
            gdal_disable_readdir_on_open
            if isinstance(gdal_disable_readdir_on_open, str)
            else _bool_env(gdal_disable_readdir_on_open)
        ),
        "GDAL_HTTP_MAX_RETRY": (
            gdal_http_max_retry
            if gdal_http_max_retry is _UNSET
            else str(gdal_http_max_retry)
        ),
        "GDAL_HTTP_RETRY_DELAY": (
            gdal_http_retry_delay
            if gdal_http_retry_delay is _UNSET
            else str(gdal_http_retry_delay)
        ),
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": _bool_env(
            gdal_http_merge_consecutive_ranges
        ),
        "GDAL_NUM_THREADS": (
            gdal_num_threads if gdal_num_threads is _UNSET else str(gdal_num_threads)
        ),
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": (
            cpl_vsil_curl_allowed_extensions
            if cpl_vsil_curl_allowed_extensions is _UNSET
            else ",".join(cpl_vsil_curl_allowed_extensions)
        ),
        "VSI_CACHE": _bool_env(vsi_cache),
        "VSI_CACHE_SIZE": vsi_cache_size
        if vsi_cache_size is _UNSET
        else str(vsi_cache_size),
        "OMP_NUM_THREADS": (
            omp_num_threads if omp_num_threads is _UNSET else str(omp_num_threads)
        ),
    }
    for key, value in values.items():
        if value is not _UNSET:
            os.environ[key] = value

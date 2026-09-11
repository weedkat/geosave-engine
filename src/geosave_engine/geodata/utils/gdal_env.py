"""Process-wide GDAL, AWS, and OpenMP read configuration."""

from __future__ import annotations

import os
from typing import Literal

from geosave_engine.utils.fn import UNSET, Unset


def _bool_env(value: bool | Unset) -> str | Unset:
    return value if value is UNSET else ("TRUE" if value else "FALSE")


def configure_gdal(
    *,
    aws_no_sign_request: bool | Unset = UNSET,
    aws_access_key_id: str | Unset = UNSET,
    aws_secret_access_key: str | Unset = UNSET,
    aws_session_token: str | Unset = UNSET,
    aws_default_region: str | Unset = UNSET,
    gdal_disable_readdir_on_open: bool | Unset = UNSET,
    gdal_http_max_retry: int | Unset = UNSET,
    gdal_http_retry_delay: float | Unset = UNSET,
    gdal_http_merge_consecutive_ranges: bool | Unset = UNSET,
    gdal_num_threads: int | Literal["ALL_CPUS"] | Unset = UNSET,
    cpl_vsil_curl_allowed_extensions: list[str] | Unset = UNSET,
    vsi_cache: bool | Unset = UNSET,
    vsi_cache_size: int | Unset = UNSET,
    omp_num_threads: int | Unset = UNSET,
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
        gdal_disable_readdir_on_open: Skip remote directory listing before
            opening an asset.
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
        ...     gdal_disable_readdir_on_open=True,
        ...     gdal_http_max_retry=3,
        ... )
    """
    values: dict[str, str | Unset] = {
        "AWS_NO_SIGN_REQUEST": _bool_env(aws_no_sign_request),
        "AWS_ACCESS_KEY_ID": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        "AWS_SESSION_TOKEN": aws_session_token,
        "AWS_DEFAULT_REGION": aws_default_region,
        "GDAL_DISABLE_READDIR_ON_OPEN": _bool_env(gdal_disable_readdir_on_open),
        "GDAL_HTTP_MAX_RETRY": (
            gdal_http_max_retry
            if gdal_http_max_retry is UNSET
            else str(gdal_http_max_retry)
        ),
        "GDAL_HTTP_RETRY_DELAY": (
            gdal_http_retry_delay
            if gdal_http_retry_delay is UNSET
            else str(gdal_http_retry_delay)
        ),
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": _bool_env(
            gdal_http_merge_consecutive_ranges
        ),
        "GDAL_NUM_THREADS": (
            gdal_num_threads if gdal_num_threads is UNSET else str(gdal_num_threads)
        ),
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": (
            cpl_vsil_curl_allowed_extensions
            if cpl_vsil_curl_allowed_extensions is UNSET
            else ",".join(cpl_vsil_curl_allowed_extensions)
        ),
        "VSI_CACHE": _bool_env(vsi_cache),
        "VSI_CACHE_SIZE": vsi_cache_size
        if vsi_cache_size is UNSET
        else str(vsi_cache_size),
        "OMP_NUM_THREADS": (
            omp_num_threads if omp_num_threads is UNSET else str(omp_num_threads)
        ),
    }
    for key, value in values.items():
        if value is not UNSET:
            os.environ[key] = value

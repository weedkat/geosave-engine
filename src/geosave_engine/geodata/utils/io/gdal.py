"""configure_gdal: process-wide GDAL/AWS/OpenMP read tuning. See configure_gdal."""
from __future__ import annotations

import os
from typing import Literal

from geosave_engine.utils.fn import UNSET, Unset


def _bool_env(value: bool | Unset) -> str | Unset:
    """True/False to GDAL's own "TRUE"/"FALSE" config-option spelling. UNSET passes through.

    Args:
        value: Flag to spell, or UNSET.

    Returns:
        `"TRUE"`, `"FALSE"`, or UNSET unchanged.
    """
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
    """Set the GDAL/AWS/OpenMP environment variables remote raster reads use.

    Every argument is a real OS environment variable, process-global once
    set. Call once at start-up, before anything opens a remote asset.

    Args:
        aws_no_sign_request: True skips AWS request signing — public buckets, no credentials needed.
        aws_access_key_id: S3 access key, paired with aws_secret_access_key.
        aws_secret_access_key: S3 secret key, paired with aws_access_key_id.
        aws_session_token: Temporary S3 session token (STS-issued credentials).
        aws_default_region: S3 bucket region, when not inferable from the endpoint.
        gdal_disable_readdir_on_open: True skips GDAL's directory listing
            before opening one remote file — faster on stores where that
            listing is slow or blocked.
        gdal_http_max_retry: HTTP retry attempts on a transient read
            failure. GDAL's own default is 0 (no retry).
        gdal_http_retry_delay: Seconds between HTTP retry attempts. GDAL's own default is 30.
        gdal_http_merge_consecutive_ranges: True merges adjacent byte-range
            reads into one HTTP call. GDAL's own default is True.
        gdal_num_threads: Worker threads for GDAL's own multithreaded ops
            (e.g. warp/resample) — an integer, or `"ALL_CPUS"`.
        cpl_vsil_curl_allowed_extensions: Restrict remote directory
            listing/sniffing to these file extensions (e.g. `[".tif",
            ".jp2"]`) — faster on stores that can't list efficiently.
        vsi_cache: True enables GDAL's in-memory block cache for remote reads.
        vsi_cache_size: `vsi_cache`'s own cache size in bytes. GDAL's own default is 25MB.
        omp_num_threads: Worker threads for OpenMP-linked codecs (e.g. some
            JP2 decoders). Only takes effect if set before whatever first
            initializes that codec's own thread pool — setting it later in
            a long-running process may do nothing.

    Examples:
        >>> configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)
    """
    values: dict[str, str | Unset] = {
        "AWS_NO_SIGN_REQUEST": _bool_env(aws_no_sign_request),
        "AWS_ACCESS_KEY_ID": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        "AWS_SESSION_TOKEN": aws_session_token,
        "AWS_DEFAULT_REGION": aws_default_region,
        "GDAL_DISABLE_READDIR_ON_OPEN": _bool_env(gdal_disable_readdir_on_open),
        "GDAL_HTTP_MAX_RETRY": gdal_http_max_retry if gdal_http_max_retry is UNSET else str(gdal_http_max_retry),
        "GDAL_HTTP_RETRY_DELAY": gdal_http_retry_delay if gdal_http_retry_delay is UNSET else str(gdal_http_retry_delay),
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": _bool_env(gdal_http_merge_consecutive_ranges),
        "GDAL_NUM_THREADS": gdal_num_threads if gdal_num_threads is UNSET else str(gdal_num_threads),
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": (
            cpl_vsil_curl_allowed_extensions
            if cpl_vsil_curl_allowed_extensions is UNSET
            else ",".join(cpl_vsil_curl_allowed_extensions)
        ),
        "VSI_CACHE": _bool_env(vsi_cache),
        "VSI_CACHE_SIZE": vsi_cache_size if vsi_cache_size is UNSET else str(vsi_cache_size),
        "OMP_NUM_THREADS": omp_num_threads if omp_num_threads is UNSET else str(omp_num_threads),
    }
    for key, value in values.items():
        if value is not UNSET:
            os.environ[key] = value

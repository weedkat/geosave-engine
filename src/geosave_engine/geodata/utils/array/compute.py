"""Force a lazy DataArray to compute, retrying past a transient read failure."""
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Generator

import xarray as xr
from dask.diagnostics.progress import ProgressBar
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from geosave_engine.geodata.errors import TileDecodeError


@contextmanager
def progress_bar(enabled: bool = True) -> Generator[None]:
    """Dask progress bar for the wrapped block's compute, or a no-op if disabled.

    Wraps any dask-triggering call (`.values`, `.compute()`, `.rio.to_raster()`,
    `.to_zarr()`, `.to_netcdf()`) — shows nothing for an already-computed
    (in-memory) array, since no graph runs.

    Args:
        enabled: False skips the progress bar.
    """
    if enabled:
        with ProgressBar():
            yield
    else:
        yield

# GDAL warns (doesn't raise) on a truncated JP2 tile read, straight to the OS stderr fd — see _StderrCapture.
_GDAL_DECODE_FAILURE_MARKERS = ("opj_get_decoded_tile", "Stream too short")


class _StderrCapture:
    """Redirect the process's real stderr (fd 2) into a buffer for the block's duration.

    Replays the captured bytes to the real stderr afterward (as one lump,
    not live). Fd 2 is process-wide — concurrent use from another thread
    of the same process clobbers this one's redirect target.
    """

    def __init__(self) -> None:
        self.text = ""

    def __enter__(self) -> "_StderrCapture":
        sys.stderr.flush()
        self._saved_fd = os.dup(2)
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, *exc_info: object) -> None:
        sys.stderr.flush()
        os.dup2(self._saved_fd, 2)
        os.close(self._saved_fd)
        self._tmp.seek(0)
        captured = self._tmp.read()
        self._tmp.close()
        os.write(2, captured)
        self.text = captured.decode("utf-8", errors="replace")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(OSError),  # TileDecodeError + genuine transient GDAL/network OSError
    reraise=True,  # exhausted retries re-raise the original exception, not tenacity's own wrapper
)
def safe_compute(da: xr.DataArray) -> xr.DataArray:
    """Compute a lazy DataArray, retrying (3x) past a transient read failure.

    Not safe to call from more than one thread of the same process at
    once — see _StderrCapture. Separate processes (e.g. a DataLoader's own
    workers) are unaffected.

    Args:
        da: Dask-backed DataArray, any source (STAC/GDAL read, local file, in-memory graph).

    Returns:
        Same DataArray, computed (in-memory) values.

    Raises:
        TileDecodeError: GDAL logged a tile decode failure — after 3 retries.
    """
    stderr_capture = _StderrCapture()
    with stderr_capture:
        computed = da.compute()
    matched = next((m for m in _GDAL_DECODE_FAILURE_MARKERS if m in stderr_capture.text), None)
    if matched is not None:
        raise TileDecodeError(f"GDAL tile decode failed — stderr matched {matched!r}")
    return computed

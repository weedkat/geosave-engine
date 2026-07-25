from __future__ import annotations

import copy
import logging
import os
import sys
import tempfile
import warnings
from calendar import monthrange
from dataclasses import replace
from datetime import datetime as dt
from datetime import timedelta
from typing import Any, Iterator, Literal, Protocol, Self, TypedDict

import numpy as np
import pystac
import xarray as xr
from dask.diagnostics import ProgressBar
from odc.stac import load as odc_load
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.tile import GeoAnchor, GeoTile
from geosave_engine.geodata.utils.datetime import DateRange, TemporalGranularity, TemporalReduce

from .query import StacQuery

logger = logging.getLogger(__name__)

Bands = list[str] | tuple[str, ...]

# rasterio only raises Python exceptions for GDAL CE_Failure. JP2OpenJPEG
# truncated/corrupt tile reads (usually a flaky network read) are emitted as
# CE_Warning instead — never raise, GDAL just fills the tile with
# partial/zero data and moves on. Two message variants seen in practice for
# the same underlying truncated-stream failure: "opj_get_decoded_tile()
# failed" and "Stream too short". Neither reaches Python's "rasterio._err"
# logger — rasterio only bridges GDAL errors to that logger for a narrow set
# of operations it explicitly wraps (dataset open, etc); this particular
# warning comes from deep inside GDAL's block-cache/driver machinery during a
# windowed read, a path rasterio never wraps, so it falls through to GDAL's
# own default handler — a raw write to the process's real stderr file
# descriptor, never through Python logging at all (confirmed empirically:
# a logging.Handler on "rasterio._err" sees nothing for this exact failure,
# even single-threaded). Only reliable way to see it is the OS-level stderr
# stream itself.
_GDAL_DECODE_FAILURE_MARKERS = ("opj_get_decoded_tile", "Stream too short")


class _StderrCapture:
    """Redirect the process's real stderr (fd 2) into a buffer for the block's duration.

    Replays the captured bytes to the real stderr afterward, so anything
    that wrote there (e.g. dask's own `ProgressBar`) still ends up visible —
    just as one lump when the block ends, instead of scrolling live. Needed
    because GDAL's default error handler writes some driver-level warnings
    (see `_GDAL_DECODE_FAILURE_MARKERS` above) straight to the OS stderr file
    descriptor, bypassing Python's `logging` entirely — no `logging.Handler`
    can see them, regardless of thread or logger name.
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


class SearchClient(Protocol):
    """Structural protocol for any client that can search a STAC catalog."""

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...


class StacSourceArgs(TypedDict, total=False):
    """Constructor kwargs for StacSource, used with Unpack in client.source() overloads."""

    bands: Bands
    filter: str | None
    max_nodata_fraction: float
    resampling: str
    groupby: Literal["solar_day", "id", "time"]
    chunks: dict[str, int | Any] | None
    dtype: str
    temporal_granularity: TemporalGranularity
    temporal_reduce: TemporalReduce
    temporal_slots: int
    temporal_strides: int | None
    temporal_fallback: bool


class StacSource:
    """Generic satellite data source. Owns its own temporal bucketing end to end.

    `load()` yields one already-shaped, still-lazy GeoTile per final sample
    this source's own anchor window produces — search happens once, then
    the tile is split into `temporal_granularity` windows, each reduced to
    one time step (`temporal_reduce`, `temporal_fallback`), then grouped
    `temporal_slots` at a time, `temporal_strides` apart. Caller downloads
    (`download`, module-level below) when it actually needs pixels. No
    cross-source coordination: two sources in the same `GeoPipeline` each
    run this independently, off their own real data and their own config.

    Args:
        client: STAC client used to search for items.
        collection: STAC collection identifier (as returned by `client.get_collections()`).
        bands: Band names to load for every anchor.
        filter: CQL2 text filter applied to every anchor's search (e.g.
            `"eo:cloud_cover <= 10"`) — anchor-independent, so it survives
            unchanged through every per-anchor `load()` search.
        max_nodata_fraction: Reject a bucket where nodata fraction exceeds this value.
        resampling: Resampling method passed to odc-stac.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for spatial dimensions.
        dtype: Output dtype passed to odc-stac.
        temporal_granularity: What one yielded sample's time axis is
            bucketed by. `"scene"`: one bucket per real matched
            acquisition — this source's own timestamps, no calendar math.
            `"day"`/`"month"`/`"year"`: calendar buckets instead, which can
            span more than one real scene each.
        temporal_reduce: How to collapse a bucket with more than one real
            scene down to exactly one time step.
        temporal_slots: How many consecutive buckets stack into one
            yielded sample's time dimension. `1` (default): each bucket is
            its own sample.
        temporal_strides: How many buckets apart consecutive yielded samples
            start — the window's stride. `None` (default): equals
            `temporal_slots`, i.e. non-overlapping samples. Set lower than
            `temporal_slots` for overlapping/sliding-window samples (e.g.
            `temporal_slots=4, temporal_strides=1` for a dense sliding
            4-step sequence). A trailing window shorter than
            `temporal_slots` is dropped, not padded.
        temporal_fallback: Allow substituting the nearest real scene (by
            absolute time distance, ignoring the bucket window entirely)
            when a bucket has none of its own. Default False — an empty
            bucket is skipped rather than silently using stale data.
    """

    def __init__(
        self,
        client: SearchClient,
        *,
        collection: str,
        bands: Bands | None = None,
        filter: str | None = None,
        max_nodata_fraction: float = 0.0,
        resampling: str = "bilinear",
        groupby: Literal["solar_day", "id", "time"] = "solar_day",
        chunks: dict[str, int | Any] | None = None,
        dtype: str = "uint16",
        temporal_granularity: TemporalGranularity = "scene",
        temporal_reduce: TemporalReduce = "median",
        temporal_slots: int = 1,
        temporal_strides: int | None = None,
        temporal_fallback: bool = False,
    ) -> None:
        self.client = client
        self.collection = collection
        self.bands = bands
        self.max_nodata_fraction = max_nodata_fraction
        self.resampling = resampling
        self.groupby = groupby
        self.chunks: dict[str, int | Any] = chunks if chunks is not None else {"x": 1024, "y": 1024}
        self.dtype = dtype
        self.temporal_granularity = temporal_granularity
        self.temporal_reduce = temporal_reduce
        self.temporal_slots = temporal_slots
        self.temporal_strides = temporal_strides if temporal_strides is not None else temporal_slots
        self.temporal_fallback = temporal_fallback
        self.query: StacQuery = StacQuery(collections=[collection])
        if filter is not None:
            self.query = self.query.with_filter(filter, inplace=True)

    def get_bands_metadata(self) -> dict[str, Any]:
        """Fetch asset metadata for this source's collection.

        Returns:
            `{band_name: asset.extra_fields}` for one example item. Empty
            dict if the collection has no items yet.
        """
        query = StacQuery(collections=[self.collection], max_items=1)
        items = self.client.search(query)
        if not items:
            warnings.warn(f"No items found when fetching bands metadata for {self.collection!r}")
            return {}
        return {name: asset.extra_fields for name, asset in items[0].assets.items()}
    
    def set_bands(self, bands: Bands, inplace: bool = False) -> Self:
        """Set the bands to load for this source.

        Args:
            bands: List of band names to load.
            inplace: Mutate self and return it. Default leaves self untouched,
                returns a copy with the new bands instead.

        Returns:
            Self (mutated) if `inplace`, otherwise a new `StacSource`.
        """
        target = self if inplace else copy.copy(self)
        target.bands = bands
        return target

    def set_filter(self, filter: str, inplace: bool = False) -> Self:
        """Merge a CQL2 text filter into this source's own query.

        Applies to every anchor this source loads from here on — see
        `StacQuery.with_filter` for how it merges with any filter already set.

        Args:
            filter: CQL2 text filter expression, e.g. `"eo:cloud_cover <= 10"`.
            inplace: Mutate self and return it. Default leaves self untouched,
                returns a copy with the merged filter instead.

        Returns:
            Self (mutated) if `inplace`, otherwise a new `StacSource`.
        """
        target = self if inplace else copy.copy(self)
        target.query = target.query.with_filter(filter, inplace=inplace)
        return target

    def load(self, anchor: GeoAnchor, lazy_load: bool = False) -> Iterator[GeoTile]:
        """Every final sample this source's own anchor window produces.

        Searches once, builds one lazy multi-scene GeoTile, then patches it
        (`_patch_time_window`), reduces each patch (`_reduce_tile`), and
        groups `temporal_slots` reduced patches — `temporal_strides` apart
        — into each final sample (`_stack_steps`). All three stay lazy, no
        compute.

        Args:
            anchor: Reference bbox, geobox, and datetime window.
            lazy_load: Default False — downloads (`download`, module-level
                below) each sample before yielding it. True: yields still-
                lazy tiles instead, caller downloads itself once a sample
                is actually going to be used (e.g. `GeoPipeline.fetch`,
                which discards samples that fail to align across sources —
                lazy avoids paying for a download that gets thrown away).

        Yields:
            GeoTile, `(time=temporal_slots, band, y, x)` — computed, or
            still lazy if `lazy_load`.

        Raises:
            AnchorFetchError: Nothing matched anchor's bbox/datetime window,
                or fewer usable time buckets turned up than `temporal_slots`
                needs — raised rather than silently yielding zero samples.
        """
        start, end = anchor.start, anchor.end
        query = replace(
            self.query,
            bbox=anchor.wgs84_bbox,
            datetime=(start, end),
        )
        items = self.client.search(query)
        if not items:
            raise AnchorFetchError(
                f"{self.collection!r}: no scenes matched anchor bbox/datetime window "
                f"(anchor at {anchor.centroid}, window {start}–{end})"
            )

        ds: xr.Dataset = odc_load(
            items,
            geobox=anchor.geobox,
            bands=self.bands,
            resampling=self.resampling,
            chunks=self.chunks,
            dtype=self.dtype,
            groupby=self.groupby,
        )
        da = ds.to_array(dim="band").transpose("time", "band", "y", "x")
        tile = GeoTile(
            geobox=anchor.geobox, datetime=anchor.datetime, data=da, polygon=anchor.polygon
        ).with_stac(items)

        buckets: list[GeoTile] = []
        for window_start, window_end in self._patch_time_window(tile):
            try:
                buckets.append(self._reduce_tile(tile, window_start, window_end))
            except AnchorFetchError as e:
                logger.debug("No usable scene for a time patch, skipping: %s", e)
                continue

        slots = self.temporal_slots
        if len(buckets) < slots:
            raise AnchorFetchError(
                f"{self.collection!r}: only {len(buckets)} usable time bucket(s) in anchor "
                f"window {start}–{end}, need temporal_slots={slots} "
                f"(anchor at {anchor.centroid})"
            )
        for i in range(0, len(buckets) - slots + 1, self.temporal_strides):
            stacked = self._stack_steps(buckets[i : i + slots])
            if lazy_load:
                yield stacked
            else:
                yield download(stacked, max_nodata_fraction=self.max_nodata_fraction)

    def _patch_time_window(self, tile: GeoTile) -> list[DateRange]:
        """Split tile's own datetime window into temporal_granularity patches.

        `"scene"`: one patch per tile's own real timestamp, no calendar
        math. `"day"`/`"month"`/`"year"`: consecutive calendar patches
        spanning tile's own start..end.

        Args:
            tile: This source's own lazily fetched tile.

        Returns:
            One `(start, end)` per patch, chronological.
        """
        if self.temporal_granularity == "scene":
            # Not tile.times — that truncates to whole seconds, and a point
            # window needs to exact-match tile.data.time.values (full
            # precision) in _reduce_tile's mask, or real sub-second
            # acquisition timestamps never match their own window.
            times = [v.astype("datetime64[us]").item() for v in tile.data.time.values]
            return [(t, t) for t in times]

        unit = self.temporal_granularity
        patches: list[DateRange] = []
        patch_start = tile.start
        while patch_start <= tile.end:
            if unit == "day":
                patch_end = patch_start + timedelta(days=1) - timedelta(microseconds=1)
            elif unit == "month":
                last_day = monthrange(patch_start.year, patch_start.month)[1]
                patch_end = patch_start.replace(
                    day=last_day, hour=23, minute=59, second=59, microsecond=999999
                )
            else:  # "year"
                patch_end = patch_start.replace(
                    month=12, day=31, hour=23, minute=59, second=59, microsecond=999999
                )
            patches.append((patch_start, min(patch_end, tile.end)))
            patch_start = patch_end + timedelta(microseconds=1)
        return patches

    def _reduce_tile(self, tile: GeoTile, start: dt, end: dt) -> GeoTile:
        """Collapse tile's data to exactly one time step within [start, end]. Still lazy.

        Pure — no compute, no I/O. Result stays lazy through `load()` too —
        whoever calls `load()` downloads it, once it's actually needed.

        Args:
            tile: This source's own lazily fetched tile, still lazy.
            start: Patch window start (inclusive).
            end: Patch window end (inclusive).

        Returns:
            New GeoTile rebased onto `(start, end)`, data always `(time=1,
            band, y, x)` — standardized shape, never squeezed away. Still
            dask-backed. The one remaining time coordinate is the matched
            scene's real acquisition timestamp when exactly one scene
            survives reduction (`temporal_reduce="first"`/`"last"`, or only
            one scene fell in the window to begin with) — `median`/`mean`
            fold several real scenes into one, so those get labeled with
            the bucket's own `start` instead, there being no single real
            timestamp left to attribute the pixels to.

        Raises:
            AnchorFetchError: No scene in window, and `temporal_fallback` is off.
        """
        time_values = tile.data.time.values
        mask = (time_values >= np.datetime64(start)) & (time_values <= np.datetime64(end))

        if mask.any():
            matched = tile.data.isel(time=mask)
        elif self.temporal_fallback:
            window_mid = np.datetime64(start) + (np.datetime64(end) - np.datetime64(start)) / 2
            nearest_idx = np.abs(time_values - window_mid).argmin()
            matched = tile.data.isel(time=[nearest_idx])
        else:
            raise AnchorFetchError(f"{self.collection!r}: no scene in window {start}–{end}")

        if matched.sizes["time"] == 1:
            reduced = matched
        elif self.temporal_reduce == "first":
            reduced = matched.isel(time=[0])
        elif self.temporal_reduce == "last":
            reduced = matched.isel(time=[-1])
        elif self.temporal_reduce == "median":
            # No single real scene left to attribute a timestamp to — label
            # with the bucket's own start, same synthetic marker "mean" uses.
            reduced = matched.median(dim="time").expand_dims("time").assign_coords(time=[np.datetime64(start)])
        else:  # "mean"
            reduced = matched.mean(dim="time").expand_dims("time").assign_coords(time=[np.datetime64(start)])

        return tile.with_datetime((start, end)).with_data(reduced)

    def _stack_steps(self, chunk: list[GeoTile]) -> GeoTile:
        """Concatenate temporal_slots resolved tiles into one final sample.

        Args:
            chunk: `temporal_slots` resolved tiles (each already `(time=1,
                band, y, x)`), consecutive in time.

        Returns:
            GeoTile, `(time=len(chunk), band, y, x)`.
        """
        span_start, span_end = chunk[0].start, chunk[-1].end
        stacked_data = xr.concat([bucket.data for bucket in chunk], dim="time")
        return chunk[0].with_datetime((span_start, span_end)).with_data(stacked_data)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((IOError, OSError, ValueError)),
)
def download(tile: GeoTile, *, max_nodata_fraction: float = 0.0) -> GeoTile:
    """Compute a lazy GeoTile's data. Retries on GDAL decode failure or excess nodata.

    Doesn't care how many time steps tile carries — pure compute step, no
    STAC/collection knowledge. A GDAL tile-decode warning and an
    over-threshold nodata fraction are both treated as possibly-transient
    bad reads, so both retried up to 3 times before giving up as genuinely
    unusable.

    Args:
        tile: Lazy (dask-backed) GeoTile, any time length.
        max_nodata_fraction: Give up (after retries) if computed nodata
            fraction exceeds this.

    Returns:
        New GeoTile, same shape, computed (in-memory) data.

    Raises:
        IOError: GDAL logged a tile decode failure, or nodata fraction
            exceeded max_nodata_fraction — after 3 retries.
    """
    logger.info("Downloading tile")
    stderr_capture = _StderrCapture()
    with stderr_capture:
        with ProgressBar():
            computed = tile.data.compute()
    matched = next((m for m in _GDAL_DECODE_FAILURE_MARKERS if m in stderr_capture.text), None)
    if matched is not None:
        raise IOError(f"GDAL tile decode failed — stderr matched {matched!r}")

    nodata_fraction = computed.isnull().mean().item()
    if nodata_fraction > max_nodata_fraction:
        raise IOError(f"nodata fraction {nodata_fraction:.3f} exceeds max {max_nodata_fraction}")

    return tile.with_data(computed)

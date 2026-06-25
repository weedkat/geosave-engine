from __future__ import annotations

import calendar
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime as dt, timedelta
from typing import Callable, Literal

import logging
import numpy as np
import pystac
import xarray as xr
from dateutil.relativedelta import relativedelta
from odc.stac import load as odc_load
from dask.diagnostics import ProgressBar
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from geosave_engine.geodata.core import GeoTile
from geosave_engine.geodata.stac import StacClient, StacQuery

logger = logging.getLogger(__name__)

ReduceMode = Literal["latest", "nearest", "mean", "median"]
CompositeMode = Literal["mean", "median"]
SlotMode = Literal["monthly", "weekly", "yearly", "per_scene"]
Bands = list[str] | tuple[str, ...]

_FAR_PAST = dt(2000, 1, 1)


@dataclass
class _SourceBase(ABC):
    """Shared infrastructure for all satellite sources.

    Owns the STAC client, odc-stac loading parameters, preprocessing callback,
    dask compute, and band metadata. Does not define ``load()`` — subclasses
    ``BaseSource`` and ``BaseSourceSeries`` define their own contracts.

    Args:
        client: STAC client used to search for items.
        preprocess: Optional transform applied lazily after loading, before compute.
        max_nodata_fraction: Reject tiles where the nodata fraction exceeds this value.
        resampling: Resampling method passed to odc-stac.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for spatial dimensions.
        dtype: Output dtype passed to odc-stac.
        preserve_original_order: Preserve STAC item order in the time axis.
    """

    client: StacClient
    preprocess: Callable[[xr.Dataset, list[pystac.Item]], xr.Dataset] | None = None
    max_nodata_fraction: float = 0.

    resampling: str = "bilinear"
    groupby: Literal["time", "solar_day", "id"] = "solar_day"
    chunks: dict[str, int | Literal["auto"]] = field(default_factory=lambda: {"x": 1024, "y": 1024})
    dtype: str = "float32"
    preserve_original_order: bool | None = None
    query: StacQuery = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.query = self._default_query()

    @abstractmethod
    def _default_query(self) -> StacQuery:
        """Return the source-specific base StacQuery (collection + default filters)."""
        ...

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((IOError, OSError, ValueError)),
    )
    def _compute(self, data: xr.Dataset) -> xr.Dataset:
        """Trigger dask compute and reject tiles exceeding nodata threshold."""
        collection = self.query.collections[0] if self.query.collections else "unknown"
        logger.info("Downloading '%s'", collection)

        with ProgressBar():
            result = data.compute()

        # nodata fraction across all band variables (Dataset has no scalar .item())
        nodata_fraction = result.to_array().isnull().mean().item()
        if nodata_fraction > self.max_nodata_fraction:
            raise ValueError(
                f"Nodata fraction {nodata_fraction:.2f} exceeds threshold {self.max_nodata_fraction}"
            )
        return result

    def get_bands_metadata(self) -> dict:
        """Fetch asset metadata for this source's collection."""
        query = StacQuery(collections=self.query.collections, max_items=1)
        items = self.client.search(query)
        if not items:
            warnings.warn(f"No items found when fetching bands metadata for '{type(self).__name__}'")
            return {}
        return {name: asset.extra_fields for name, asset in items[0].assets.items()}


@dataclass
class BaseSource(_SourceBase):
    """Abstract base for single-scene satellite sources.

    Searches a time window around ``anchor.datetime``, loads all matching scenes,
    and collapses them to a single ``GeoTile`` via ``reduce``.

    Args:
        time_range: Window around anchor datetime to search. A single timedelta
            is applied symmetrically; a tuple applies ``(before, after)`` offsets.
    """

    time_range: timedelta | tuple[timedelta, timedelta] = field(default_factory=lambda: timedelta(days=1))

    def _build_query(self, anchor: GeoTile) -> StacQuery:
        """Inject anchor bbox and datetime into the base query."""
        if isinstance(self.time_range, tuple):
            start = anchor.datetime - self.time_range[0]
            end = anchor.datetime + self.time_range[1]
        else:
            start = anchor.datetime - self.time_range
            end = anchor.datetime + self.time_range
        return replace(self.query, bbox=anchor.wgs84_bbox, datetime=(start, end))

    def load(self, anchor: GeoTile, reduce: ReduceMode = "nearest", bands: Bands | None = None) -> GeoTile:
        """Search STAC, load pixels aligned to anchor, preprocess, reduce, and compute.

        Preprocessing is applied lazily before compute so dask fuses the load
        and transform into a single pass. Reduction (except ``'median'``) is
        also lazy.

        Args:
            anchor: Reference tile providing bbox, geobox, and datetime.
            reduce: How to collapse multiple scenes.
                ``'latest'``/``'nearest'`` select one scene lazily before compute.
                ``'mean'`` aggregates all scenes lazily.
                ``'median'`` aggregates all scenes (may not be fully lazy).

        Raises:
            ValueError: If no STAC items are found or nodata exceeds threshold.
        """
        items = self.client.search(self._build_query(anchor))
        if not items:
            raise ValueError(
                f"No STAC items found for anchor at {anchor.centroid} on {anchor.datetime.date()}"
            )

        ds: xr.Dataset = odc_load(
            items,
            geobox=anchor.geobox,
            bands=bands,
            resampling=self.resampling,
            chunks=self.chunks,
            dtype=self.dtype,
            groupby=self.groupby,
            preserve_original_order=self.preserve_original_order or self.query.sortby is not None,
        )

        if self.preprocess:
            ds = self.preprocess(ds, items)              # lazy — fused into dask graph

        if reduce == "latest":
            ds = ds.isel(time=-1)                        # lazy
        elif reduce == "nearest":
            ds = ds.sel(time=anchor.datetime, method="nearest")  # lazy
        elif reduce == "mean":
            ds = ds.mean(dim="time")                     # lazy
        elif reduce == "median":
            ds = ds.median(dim="time")                   # may force compute internally

        ds = self._compute(ds)
        return anchor.with_data(ds).with_stac(items)


@dataclass
class BaseSourceSeries(_SourceBase):
    """Abstract base for fixed-length time-series satellite sources.

    Returns exactly ``n_steps`` GeoTiles, one per calendar slot or scene,
    ordered oldest to newest.

    Calendar modes (``'monthly'``, ``'weekly'``, ``'yearly'``) search a full
    calendar period per slot and collapse scenes via ``composite``. Slots step
    backward from the anchor's calendar period so the last slot always covers
    the anchor's own period (month/week/year).

    ``'per_scene'`` mode discovers the ``n_steps`` most recent scenes before
    ``anchor.datetime`` via STAC sort; no compositing is applied.

    Args:
        n_steps: Number of slots (or scenes) to return.
        slot_mode: Temporal grouping strategy.
        composite: How to aggregate scenes within a calendar slot.
            Ignored for ``'per_scene'``.
    """

    n_steps: int = 4
    slot_mode: SlotMode = "monthly"
    composite: CompositeMode = "mean"

    # ------------------------------------------------------------------ slots

    def _month_slots(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        month_start = anchor.datetime.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        slots = []
        for i in range(self.n_steps - 1, -1, -1):
            start = month_start - relativedelta(months=i)
            last_day = calendar.monthrange(start.year, start.month)[1]
            end = start.replace(day=last_day, hour=23, minute=59, second=59)
            slots.append((replace(self.query, bbox=anchor.wgs84_bbox, datetime=(start, end)), end))
        return slots

    def _week_slots(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        monday = (anchor.datetime - timedelta(days=anchor.datetime.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        slots = []
        for i in range(self.n_steps - 1, -1, -1):
            start = monday - timedelta(weeks=i)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            slots.append((replace(self.query, bbox=anchor.wgs84_bbox, datetime=(start, end)), end))
        return slots

    def _year_slots(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        slots = []
        for i in range(self.n_steps - 1, -1, -1):
            year = anchor.datetime.year - i
            start = dt(year, 1, 1, 0, 0, 0)
            end = dt(year, 12, 31, 23, 59, 59)
            slots.append((replace(self.query, bbox=anchor.wgs84_bbox, datetime=(start, end)), end))
        return slots

    def _build_slot_queries(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        """Return ``(query, slot_dt)`` pairs ordered oldest to newest."""
        if self.slot_mode == "monthly":
            return self._month_slots(anchor)
        if self.slot_mode == "weekly":
            return self._week_slots(anchor)
        return self._year_slots(anchor)

    # ------------------------------------------------------------------ load

    def _load_calendar(self, anchor: GeoTile, bands: Bands | None) -> GeoTile:
        datasets: list[xr.Dataset] = []
        times: list[dt] = []
        all_items: list[pystac.Item] = []

        for query, slot_dt in self._build_slot_queries(anchor):
            items = self.client.search(query)
            if not items:
                raise ValueError(
                    f"No STAC items found for slot {slot_dt.date()} at {anchor.centroid}"
                )
            ds: xr.Dataset = odc_load(
                items,
                geobox=anchor.geobox,
                bands=bands,
                resampling=self.resampling,
                chunks=self.chunks,
                dtype=self.dtype,
                groupby=self.groupby,
                preserve_original_order=self.preserve_original_order or self.query.sortby is not None,
            )
            if self.preprocess:
                ds = self.preprocess(ds, items)
            ds = ds.mean(dim="time") if self.composite == "mean" else ds.median(dim="time")
            datasets.append(self._compute(ds))
            times.append(slot_dt)
            all_items.extend(items)

        time_coord = xr.Variable("time", [np.datetime64(t, "ns") for t in times])
        stacked = xr.concat(datasets, dim=time_coord)
        return GeoTile(geobox=anchor.geobox, datetime=anchor.datetime, data=stacked).with_stac(all_items)

    def _load_per_scene(self, anchor: GeoTile, bands: Bands | None) -> GeoTile:
        query = replace(
            self.query,
            bbox=anchor.wgs84_bbox,
            datetime=(_FAR_PAST, anchor.datetime),
            sortby=[{"field": "datetime", "direction": "desc"}],
            max_items=self.n_steps,
        )
        items = self.client.search(query)
        if len(items) < self.n_steps:
            raise ValueError(
                f"Found {len(items)} scenes, need {self.n_steps} at {anchor.centroid}"
            )

        datasets: list[xr.Dataset] = []
        times: list[dt] = []
        for item in reversed(items):  # oldest → newest
            scene_dt = item.datetime
            if scene_dt is None:
                raise ValueError(f"STAC item {item.id!r} has no datetime")
            ds: xr.Dataset = odc_load(
                [item],
                geobox=anchor.geobox,
                bands=bands,
                resampling=self.resampling,
                chunks=self.chunks,
                dtype=self.dtype,
                groupby=self.groupby,
            ).isel(time=0)
            if self.preprocess:
                ds = self.preprocess(ds, [item])
            datasets.append(self._compute(ds))
            times.append(scene_dt)

        time_coord = xr.Variable("time", [np.datetime64(t, "ns") for t in times])
        stacked = xr.concat(datasets, dim=time_coord)
        return GeoTile(geobox=anchor.geobox, datetime=anchor.datetime, data=stacked).with_stac(items)

    def load(self, anchor: GeoTile, bands: Bands | None = None) -> GeoTile:
        """Load all slots and return a single time-series GeoTile, oldest first.

        Args:
            anchor: Reference tile providing bbox, geobox, and datetime.
            bands: Band names to load. Defaults to all bands in the collection.

        Raises:
            ValueError: If any slot has no STAC items, fewer scenes than ``n_steps``
                are found (``per_scene``), or nodata exceeds threshold.
        """
        if self.slot_mode == "per_scene":
            return self._load_per_scene(anchor, bands)
        return self._load_calendar(anchor, bands)

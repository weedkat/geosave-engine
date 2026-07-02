from __future__ import annotations

import calendar
import warnings
from dataclasses import replace
from datetime import datetime as dt, timedelta
from typing import Any, Literal, Protocol, TypedDict

import logging
import numpy as np
import pystac
import xarray as xr
from dateutil.relativedelta import relativedelta
from odc.stac import load as odc_load
from dask.diagnostics import ProgressBar
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing_extensions import Self

from geosave_engine.geodata.core import GeoTile
from geosave_engine.geodata.stac.query import StacQuery

logger = logging.getLogger(__name__)

CompositeMode = Literal["mean", "median", "nearest", "latest"]
SlotMode = Literal["daily", "weekly", "monthly", "yearly", "per_scene"]
Bands = list[str] | tuple[str, ...]

_FAR_PAST = dt(2000, 1, 1)


class SearchClient(Protocol):
    """Structural protocol for any client that can search a STAC catalog."""

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...


class SourceArgs(TypedDict, total=False):
    """Constructor kwargs for Source, used with Unpack in client.source() overloads."""
    n_slots: int
    slot_mode: SlotMode
    composite: CompositeMode
    max_nodata_fraction: float
    resampling: str
    groupby: Literal["time", "solar_day", "id"]
    chunks: dict[str, int | Literal["auto"]] | None
    dtype: str
    preserve_original_order: bool | None


class Source:
    """Generic satellite data source. Loads raster tiles from a STAC collection.

    Subclasses override ``preprocess`` to apply collection-specific radiometric scaling.
    Call ``raw()`` on any instance to skip preprocessing and receive raw odc-stac output.

    Args:
        client: STAC client used to search for items.
        collection_id: STAC collection identifier (as returned by ``client.get_collections()``).
        n_slots: Number of time slots to return.
        slot_mode: Temporal grouping per slot.
        composite: How to reduce scenes within a calendar slot.
            ``'nearest'``/``'latest'`` pick one scene; ``'mean'``/``'median'`` aggregate all.
        max_nodata_fraction: Reject tiles where nodata fraction exceeds this value.
        resampling: Resampling method passed to odc-stac.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for spatial dimensions.
        dtype: Output dtype passed to odc-stac.
        preserve_original_order: Preserve STAC item order in the time axis.
    """

    _use_preprocess: bool = True

    def __init__(
        self,
        client: SearchClient,
        *,
        collection_id: str,
        n_slots: int = 1,
        slot_mode: SlotMode = "daily",
        composite: CompositeMode = "nearest",
        max_nodata_fraction: float = 0.0,
        resampling: str = "bilinear",
        groupby: Literal["time", "solar_day", "id"] = "solar_day",
        chunks: dict[str, int | Literal["auto"]] | None = None,
        dtype: str = "float32",
        preserve_original_order: bool | None = None,
    ) -> None:
        self.client = client
        self.collection_id = collection_id
        self.n_slots = n_slots
        self.slot_mode = slot_mode
        self.composite = composite
        self.max_nodata_fraction = max_nodata_fraction
        self.resampling = resampling
        self.groupby = groupby
        self.chunks: dict[str, int | Literal["auto"]] = chunks if chunks is not None else {"x": 1024, "y": 1024}
        self.dtype = dtype
        self.preserve_original_order = preserve_original_order
        self.query: StacQuery = StacQuery(collections=[collection_id])

    def preprocess(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:  # noqa: ARG002
        """Apply collection-specific radiometric scaling.

        Base implementation is identity (no scaling). Subclasses override for
        collection-specific transforms (e.g. Sentinel-2 scale+offset, HLS ×0.0001).

        Args:
            ds: Raw odc-stac output dataset.
            items: STAC items used to load ``ds``; may carry scale/offset metadata.

        Returns:
            Scaled dataset ready for downstream pipelines.
        """
        return ds

    def raw(self) -> Self:
        """Disable preprocessing for this source instance.

        Useful when a model expects raw DN values (e.g. ``GraniteGeospatialBiomass``
        expects HLS DN, not reflectance).

        Returns:
            Self with preprocessing skipped on ``load()``.

        Examples:
            >>> src = pc.source("hls2-s30").raw()  # DN values, no ×0.0001 scaling
        """
        self._use_preprocess = False
        return self

    def with_filter(self, expr: dict[str, Any]) -> Self:
        """Add a raw CQL2 filter expression to the query.

        Args:
            expr: CQL2 expression built via ``CQL2`` helpers.

        Examples:
            >>> source.with_filter(CQL2.lte("eo:cloud_cover", 20))
        """
        self.query.with_filter(expr)
        return self

    def get_item_example(self) -> pystac.Item:
        """Fetch one STAC item to inspect available properties.

        Returns:
            One ``pystac.Item``; call ``.properties`` to see filterable fields.

        Raises:
            ValueError: If no items found for this collection.
        """
        query = StacQuery(collections=[self.collection_id], max_items=1)
        items = self.client.search(query)
        if not items:
            raise ValueError(f"No items found for collection {self.collection_id!r}")
        return items[0]

    def get_bands_metadata(self) -> dict:
        """Fetch asset metadata for this source's collection."""
        query = StacQuery(collections=[self.collection_id], max_items=1)
        items = self.client.search(query)
        if not items:
            warnings.warn(f"No items found when fetching bands metadata for {self.collection_id!r}")
            return {}
        return {name: asset.extra_fields for name, asset in items[0].assets.items()}

    # ------------------------------------------------------------------ slots

    def _day_slots(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        day_start = anchor.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        slots = []
        for i in range(self.n_slots - 1, -1, -1):
            start = day_start - timedelta(days=i)
            end = start.replace(hour=23, minute=59, second=59)
            slots.append((replace(self.query, bbox=anchor.wgs84_bbox, datetime=(start, end)), end))
        return slots

    def _month_slots(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        month_start = anchor.datetime.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        slots = []
        for i in range(self.n_slots - 1, -1, -1):
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
        for i in range(self.n_slots - 1, -1, -1):
            start = monday - timedelta(weeks=i)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            slots.append((replace(self.query, bbox=anchor.wgs84_bbox, datetime=(start, end)), end))
        return slots

    def _year_slots(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        slots = []
        for i in range(self.n_slots - 1, -1, -1):
            year = anchor.datetime.year - i
            start = dt(year, 1, 1, 0, 0, 0)
            end = dt(year, 12, 31, 23, 59, 59)
            slots.append((replace(self.query, bbox=anchor.wgs84_bbox, datetime=(start, end)), end))
        return slots

    def _build_slot_queries(self, anchor: GeoTile) -> list[tuple[StacQuery, dt]]:
        if self.slot_mode == "daily":
            return self._day_slots(anchor)
        if self.slot_mode == "weekly":
            return self._week_slots(anchor)
        if self.slot_mode == "monthly":
            return self._month_slots(anchor)
        return self._year_slots(anchor)

    # ------------------------------------------------------------------ load

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((IOError, OSError, ValueError)),
    )
    def _compute(self, data: xr.Dataset) -> xr.Dataset:
        logger.info("Downloading '%s'", self.collection_id)
        with ProgressBar():
            result = data.compute()
        nodata_fraction = result.to_array().isnull().mean().item()
        if nodata_fraction > self.max_nodata_fraction:
            raise ValueError(
                f"Nodata fraction {nodata_fraction:.2f} exceeds threshold {self.max_nodata_fraction}"
            )
        return result

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
            if self._use_preprocess:
                ds = self.preprocess(ds, items)
            if self.composite == "mean":
                ds = ds.mean(dim="time")
            elif self.composite == "median":
                ds = ds.median(dim="time")
            elif self.composite == "latest":
                ds = ds.isel(time=-1)
            else:
                idx = int(abs(ds.time - np.datetime64(anchor.datetime, "ns")).argmin())
                ds = ds.isel(time=idx)
            datasets.append(self._compute(ds))
            times.append(slot_dt)
            all_items.extend(items)

        time_coord = xr.Variable("time", [np.datetime64(t, "ns") for t in times])
        stacked = xr.concat(datasets, dim=time_coord)
        if self.n_slots == 1:
            stacked = stacked.isel(time=0)
        da = stacked.to_array(dim="band")
        if "time" in da.dims:
            da = da.transpose("time", "band", "y", "x")
        else:
            da = da.transpose("band", "y", "x")
        return GeoTile(geobox=anchor.geobox, datetime=anchor.datetime, data=da).with_stac(all_items)

    def _load_per_scene(self, anchor: GeoTile, bands: Bands | None) -> GeoTile:
        query = replace(
            self.query,
            bbox=anchor.wgs84_bbox,
            datetime=(_FAR_PAST, anchor.datetime),
            sortby=[{"field": "datetime", "direction": "desc"}],
            max_items=self.n_slots,
        )
        items = self.client.search(query)
        if len(items) < self.n_slots:
            raise ValueError(
                f"Found {len(items)} scenes, need {self.n_slots} at {anchor.centroid}"
            )

        datasets: list[xr.Dataset] = []
        times: list[dt] = []
        for item in reversed(items):
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
            if self._use_preprocess:
                ds = self.preprocess(ds, [item])
            datasets.append(self._compute(ds))
            times.append(scene_dt)

        time_coord = xr.Variable("time", [np.datetime64(t, "ns") for t in times])
        stacked = xr.concat(datasets, dim=time_coord)
        da = stacked.to_array(dim="band").transpose("time", "band", "y", "x")
        return GeoTile(geobox=anchor.geobox, datetime=anchor.datetime, data=da).with_stac(items)

    def load(self, anchor: GeoTile, bands: Bands | None = None) -> GeoTile:
        """Load all slots and return single GeoTile with time dim, oldest first.

        Args:
            anchor: Reference tile providing bbox, geobox, and datetime.
            bands: Band names to load. Defaults to all bands.

        Returns:
            GeoTile with time dimension of length ``n_slots``.

        Raises:
            ValueError: If any slot has no STAC items, or nodata exceeds threshold.
        """
        if self.slot_mode == "per_scene":
            return self._load_per_scene(anchor, bands)
        return self._load_calendar(anchor, bands)

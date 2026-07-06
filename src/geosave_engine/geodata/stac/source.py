from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from datetime import datetime as dt
from typing import Any, Literal, Protocol, TypedDict

import pystac
import xarray as xr
from dask.diagnostics import ProgressBar
from odc.stac import load as odc_load
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from geosave_engine.geodata.core import GeoTile

from .query import StacQuery

logger = logging.getLogger(__name__)

Bands = list[str] | tuple[str, ...]


class SearchClient(Protocol):
    """Structural protocol for any client that can search a STAC catalog."""

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...


class SourceArgs(TypedDict, total=False):
    """Constructor kwargs for Source, used with Unpack in client.source() overloads."""

    bands: Bands
    max_nodata_fraction: float
    resampling: str
    groupby: Literal["solar_day", "id", "time"]
    chunks: dict[str, int | Any] | None
    dtype: str


class Source:
    """Generic satellite data source. Loads every scene in an anchor's window, exactly as published.

    No radiometric preprocessing, no compositing — always returns raw values
    per real STAC item, one GeoTile per matching scene. Apply scale/offset or
    temporal compositing as explicit downstream pipeline steps instead,
    reading from per-scene cache entries this produces.

    Args:
        client: STAC client used to search for items.
        collection_id: STAC collection identifier (as returned by `client.get_collections()`).
        bands: Band names to load for every anchor.
        max_nodata_fraction: Reject tiles where nodata fraction exceeds this value.
        resampling: Resampling method passed to odc-stac.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for spatial dimensions.
        dtype: Output dtype passed to odc-stac.
    """

    def __init__(
        self,
        client: SearchClient,
        *,
        collection_id: str,
        bands: Bands | None = None,
        max_nodata_fraction: float = 0.0,
        resampling: str = "bilinear",
        groupby: Literal["solar_day", "id", "time"] = "solar_day",
        chunks: dict[str, int | Any] | None = None,
        dtype: str = "uint16",
    ) -> None:
        self.client = client
        self.collection_id = collection_id
        self.bands = bands
        self.max_nodata_fraction = max_nodata_fraction
        self.resampling = resampling
        self.groupby = groupby
        self.chunks: dict[str, int | Any] = chunks if chunks is not None else {"x": 1024, "y": 1024}
        self.dtype = dtype
        self.query: StacQuery = StacQuery(collections=[collection_id])

    def get_bands_metadata(self) -> dict[str, Any]:
        """Fetch asset metadata for this source's collection.

        Returns:
            `{band_name: asset.extra_fields}` for one example item. Empty
            dict if the collection has no items yet.
        """
        query = StacQuery(collections=[self.collection_id], max_items=1)
        items = self.client.search(query)
        if not items:
            warnings.warn(f"No items found when fetching bands metadata for {self.collection_id!r}")
            return {}
        return {name: asset.extra_fields for name, asset in items[0].assets.items()}
    
    def set_bands(self, bands: Bands) -> None:
        """Set the bands to load for this source.

        Args:
            bands: Band names to load for every anchor.
        """
        self.bands = bands

    def load(self, anchor: GeoTile) -> list[GeoTile]:
        """Load every scene within anchor's own datetime window, for this source's bands.

        A degenerate window (a resolved single instant) naturally finds at
        most whatever's on that exact day. A real window finds every
        matching scene. No lookback — the window searched is exactly
        anchor's own, nothing wider.

        Args:
            anchor: Reference tile providing bbox, geobox, and datetime window.

        Returns:
            One GeoTile per matching scene that passes `max_nodata_fraction`,
            chronological order — no time dimension, each stamped with its
            own real acquisition datetime (never anchor's). Empty list if
            nothing matches, or everything found is rejected as nodata.
        """
        start, end = anchor.date_range
        query = replace(
            self.query,
            bbox=anchor.wgs84_bbox,
            datetime=(start, end),
            sortby=[{"field": "datetime", "direction": "asc"}],
        )
        items = self.client.search(query)
        if not items:
            return []

        ds: xr.Dataset = odc_load(
            items,
            geobox=anchor.geobox,
            bands=self.bands,
            resampling=self.resampling,
            chunks=self.chunks,
            dtype=self.dtype,
            groupby=self.groupby,
        )
        ds = self.download(ds)

        tiles: list[GeoTile] = []
        for t in ds.time.values:
            slice_ds = ds.sel(time=t)
            nodata_fraction = slice_ds.to_array().isnull().mean().item()
            if nodata_fraction > self.max_nodata_fraction:
                continue
            da = slice_ds.to_array(dim="band").transpose("band", "y", "x")
            item_dt = dt.fromisoformat(str(t.astype("datetime64[s]")))
            tiles.append(GeoTile(geobox=anchor.geobox, datetime=item_dt, data=da).with_stac(items))
        return tiles

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((IOError, OSError, ValueError)),
    )
    def download(self, data: xr.Dataset) -> xr.Dataset:
        """Compute a lazy odc-stac dataset.

        Args:
            data: Lazy (dask-backed) dataset from `odc_load`.

        Returns:
            Computed (in-memory) dataset.
        """
        logger.info("Downloading '%s'", self.collection_id)
        with ProgressBar():
            return data.compute()

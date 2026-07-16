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

from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.tile import GeoAnchor, GeoTile

from .query import StacQuery

logger = logging.getLogger(__name__)

Bands = list[str] | tuple[str, ...]


class SearchClient(Protocol):
    """Structural protocol for any client that can search a STAC catalog."""

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...


class StacSourceArgs(TypedDict, total=False):
    """Constructor kwargs for StacSource, used with Unpack in client.source() overloads."""

    bands: Bands
    max_nodata_fraction: float
    resampling: str
    groupby: Literal["solar_day", "id", "time"]
    chunks: dict[str, int | Any] | None
    dtype: str


class StacSource:
    """Generic satellite data source. Loads every scene in an anchor's window, exactly as published.

    No radiometric preprocessing, no compositing — always returns raw values
    per real STAC item. A range window comes back as one GeoTile with a time
    axis (one step per matching scene); a resolved single instant comes back
    with no time axis. Apply scale/offset or temporal compositing as explicit
    downstream pipeline steps.

    Args:
        client: STAC client used to search for items.
        collection: STAC collection identifier (as returned by `client.get_collections()`).
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
        collection: str,
        bands: Bands | None = None,
        max_nodata_fraction: float = 0.0,
        resampling: str = "bilinear",
        groupby: Literal["solar_day", "id", "time"] = "solar_day",
        chunks: dict[str, int | Any] | None = None,
        dtype: str = "uint16",
    ) -> None:
        self.client = client
        self.collection = collection
        self.bands = bands
        self.max_nodata_fraction = max_nodata_fraction
        self.resampling = resampling
        self.groupby = groupby
        self.chunks: dict[str, int | Any] = chunks if chunks is not None else {"x": 1024, "y": 1024}
        self.dtype = dtype
        self.query: StacQuery = StacQuery(collections=[collection])

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
    
    def set_bands(self, bands: Bands) -> None:
        """Set the bands to load for this source.

        Args:
            bands: List of band names to load.
        """
        self.bands = bands

    def load(self, anchor: GeoAnchor) -> GeoTile:
        """Load every scene within anchor's own datetime window, for this source's bands.

        A degenerate window (a resolved single instant) naturally finds at
        most whatever's on that exact day. A real window finds every
        matching scene. No lookback — the window searched is exactly
        anchor's own, nothing wider.

        Args:
            anchor: Reference bbox, geobox, and datetime window.

        Returns:
            One GeoTile. `(band, y, x)`, no time axis, if exactly one scene
            matched. `(time, band, y, x)` if several did — each time step
            stamped with its own real acquisition instant; the tile's own
            `datetime` stays anchor's requested window.

        Raises:
            AnchorFetchError: Nothing matched anchor's bbox/datetime window,
                or everything found was rejected as nodata.
        """
        start, end = anchor.start, anchor.end
        query = replace(
            self.query,
            bbox=anchor.wgs84_bbox,
            datetime=(start, end),
            sortby=[{"field": "datetime", "direction": "asc"}],
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
        ds = self.download(ds)

        kept_times = []
        kept_slices: list[xr.DataArray] = []
        for t in ds.time.values:
            slice_ds = ds.sel(time=t)
            nodata_fraction = slice_ds.to_array().isnull().mean().item()
            if nodata_fraction > self.max_nodata_fraction:
                continue
            kept_times.append(t)
            kept_slices.append(slice_ds.to_array(dim="band").transpose("band", "y", "x"))

        if not kept_slices:
            raise AnchorFetchError(
                f"{self.collection!r}: {len(items)} scene(s) matched but all exceeded "
                f"nodata threshold (anchor at {anchor.centroid}, window {start}–{end})"
            )

        if len(kept_slices) == 1:
            item_dt = dt.fromisoformat(str(kept_times[0].astype("datetime64[s]")))
            return GeoTile(geobox=anchor.geobox, datetime=item_dt, data=kept_slices[0]).with_stac(items)

        da = xr.concat(kept_slices, dim="time").assign_coords(time=kept_times)
        return GeoTile(geobox=anchor.geobox, datetime=anchor.datetime, data=da).with_stac(items)

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
        logger.info("Downloading '%s'", self.collection)
        with ProgressBar():
            return data.compute()

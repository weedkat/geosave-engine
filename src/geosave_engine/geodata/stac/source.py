"""One STAC collection loaded onto an anchor's grid."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime as dt
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

import odc.stac
from pydantic import BaseModel, ConfigDict, Field

from geosave_engine.geodata.attrs import read_asset_fields
from geosave_engine.geodata.errors import AnchorFetchError

from .query import StacQuery

from .stamp import StacGroupby, stamp_stac

if TYPE_CHECKING:
    import pystac

    from geosave_engine.geodata.core import GeoAnchor
    from geosave_engine.geodata import Dataset

Bands = list[str] | tuple[str, ...]
Resampling = str | dict[str, str]
ChunkSize = int | Literal["auto"]


def _default_chunks() -> dict[str, ChunkSize]:
    """Return the default spatial Dask chunks."""
    return {"x": 1024, "y": 1024}


class SearchClient(Protocol):
    """Any client that can run a STAC search and read a collection."""

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...

    def collection(self, collection: str) -> pystac.Collection: ...


class StacSourceConfig(BaseModel):
    """How one source loads its pixels.

    Args:
        bands: Band names to load. None loads whatever the matched items
            declare.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for the spatial dims. None loads eagerly.
        resampling: Resampling odc-stac applies, one mode or one per band.
            None keeps each asset's default.
        dtype: Output dtype. None keeps each collection's own.
        nodata: Nodata value, paired with `dtype`. None keeps each asset's own.
        fail_on_error: False skips a scene that fails to read instead of
            raising.
        item_properties: Item properties captured per acquisition, e.g.
            `("platform", "eo:cloud_cover")`. Empty captures identity only;
            None captures every property each item declares.
        asset_fields: Asset fields captured onto the variable they describe.
            Empty captures none; None captures every field each asset declares.
        stac_cfg: Per-collection band overrides odc-stac applies, correcting
            nodata, dtype, or asset aliases a catalog publishes wrongly.
        pool: Reader threads odc-stac uses. None takes its default.
        progress: Progress bar odc-stac reports reads through. None is silent.
        patch_url: Called on every asset href before it is read, for a catalog
            whose URLs need signing at load time.
        preserve_original_order: True keeps the matched items in search order
            rather than sorting them by time.

    Examples:
        >>> StacSourceConfig(bands=["B04", "B08"], item_properties=None)
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    bands: Bands | None = None
    groupby: StacGroupby = "solar_day"
    chunks: dict[str, ChunkSize] | None = Field(default_factory=_default_chunks)
    resampling: Resampling | None = None
    dtype: str | None = None
    nodata: float | None = None
    fail_on_error: bool = True
    item_properties: tuple[str, ...] | None = ()
    asset_fields: tuple[str, ...] | None = ()
    stac_cfg: dict[str, Any] | None = None
    pool: int | None = None
    progress: Any | None = None
    patch_url: Callable[[str], str] | None = None
    preserve_original_order: bool = False


_UNSET: Any = object()


class StacSource:
    """One STAC collection, loaded onto an anchor's own grid.

    Pixels arrive as the provider published them, with no radiometric scaling.
    A fresh source loads on every default; narrow it with `set_config` and
    `set_query` before `load`.

    Args:
        client: Client used to search the catalog.
        collection: Collection ID to load.

    Examples:
        >>> from geosave_engine.geodata import configure_gdal
        >>> configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)
        >>> source = client.source("sentinel-2-l2a").set_config(bands=["B04", "B08"])
        >>> ds = source.load(anchor)
        >>> ds.gs.attrs.root.get(StacMetadata).properties()
        ('eo:cloud_cover', 'platform')
    """

    def __init__(self, client: SearchClient, *, collection: str) -> None:
        """Bind the client and collection, on default settings.

        Args:
            client: Client used to search the catalog.
            collection: Collection ID to load.
        """
        self.client = client
        self.collection = collection
        self.config = StacSourceConfig()
        self.query = StacQuery(collections=[collection])

    def set_config(
        self,
        *,
        bands: Bands | None = _UNSET,
        groupby: StacGroupby = _UNSET,
        chunks: dict[str, ChunkSize] | None = _UNSET,
        resampling: Resampling | None = _UNSET,
        dtype: str | None = _UNSET,
        nodata: float | None = _UNSET,
        fail_on_error: bool = _UNSET,
        item_properties: Sequence[str] | None = _UNSET,
        asset_fields: Sequence[str] | None = _UNSET,
        stac_cfg: dict[str, Any] | None = _UNSET,
        pool: int | None = _UNSET,
        progress: Any | None = _UNSET,
        patch_url: Callable[[str], str] | None = _UNSET,
        preserve_original_order: bool = _UNSET,
    ) -> Self:
        """Merge load settings onto the current config.

        Only the fields passed change; the rest keep their current value. See
        `StacSourceConfig` for what each field means.

        Args:
            bands: Band names to load. None loads whatever the matched items
                declare.
            groupby: How odc-stac groups scenes along the time axis.
            chunks: Dask chunk sizes for the spatial dims. None loads eagerly.
            resampling: Resampling odc-stac applies, one mode or one per band.
                None keeps each asset's default.
            dtype: Output dtype. None keeps each collection's own.
            nodata: Nodata value, paired with `dtype`. None keeps each asset's
                own.
            fail_on_error: False skips a scene that fails to read instead of
                raising.
            item_properties: Item properties captured per acquisition. Empty
                captures identity only; None captures every property.
            asset_fields: Asset fields captured onto the variable they
                describe. Empty captures none; None captures every field each
                asset declares.
            stac_cfg: Per-collection band overrides odc-stac applies.
            pool: Reader threads odc-stac uses. None takes its default.
            progress: Progress bar odc-stac reports reads through. None is
                silent.
            patch_url: Called on every asset href before it is read.
            preserve_original_order: True keeps the matched items in search
                order rather than sorting them by time.

        Returns:
            This source, for chaining.

        Raises:
            ValidationError: A value does not satisfy `StacSourceConfig`.

        Examples:
            >>> source.set_config(bands=["B04", "B08"], groupby="time")
        """
        changes: dict[str, Any] = {
            "bands": bands,
            "groupby": groupby,
            "chunks": chunks,
            "resampling": resampling,
            "dtype": dtype,
            "nodata": nodata,
            "fail_on_error": fail_on_error,
            "item_properties": item_properties,
            "asset_fields": asset_fields,
            "stac_cfg": stac_cfg,
            "pool": pool,
            "progress": progress,
            "patch_url": patch_url,
            "preserve_original_order": preserve_original_order,
        }
        updates = {name: v for name, v in changes.items() if v is not _UNSET}
        self.config = type(self.config)(**{**self.config.model_dump(), **updates})
        return self

    def set_query(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        intersects: dict[str, Any] | None = None,
        datetime: dt | str | tuple[dt, dt] | None = None,
        ids: Sequence[str] | None = None,
        filter: str | None = None,
        sortby: Sequence[str] | None = None,
        max_items: int | None = None,
        limit: int | None = None,
    ) -> Self:
        """Replace the search narrowing merged with each anchor's extent.

        Every call rebuilds the query from scratch on this source's own
        collection.

        Args:
            bbox: WGS84 bounding box `(min_lon, min_lat, max_lon, max_lat)`.
            intersects: GeoJSON geometry to intersect.
            datetime: Instant, ISO 8601 string, or `(start, end)` range.
            ids: Item IDs to fetch directly, bypassing spatial and temporal
                narrowing.
            filter: CQL2 text expression, e.g. `"eo:cloud_cover <= 10"`.
            sortby: Field paths, each optionally prefixed `-` for descending
                or `+` for ascending, e.g. `"-properties.eo:cloud_cover"`.
            max_items: Client-side cap on items returned.
            limit: Server page-size hint.

        Returns:
            This source, for chaining.

        Raises:
            ValueError: `bbox` is not a valid WGS84 box, `filter` is not valid
                CQL2 text, `sortby` has an empty field name, or `max_items` or
                `limit` is below one.

        Examples:
            >>> source.set_query(
            ...     filter="eo:cloud_cover <= 10", sortby=["-properties.eo:cloud_cover"]
            ... )
        """
        query = StacQuery(
            collections=[self.collection],
            bbox=bbox,
            intersects=intersects,
            datetime=datetime,
            ids=list(ids) if ids is not None else None,
            max_items=max_items,
            limit=limit,
        )
        if filter is not None:
            query = query.set_filter(filter)
        if sortby is not None:
            query = query.sort_by(*sortby)
        self.query = query
        return self

    def __repr__(self) -> str:
        """Describe the collection and the settings it loads with.

        Returns:
            Multi-line summary of the collection and its config.
        """
        lines = "\n".join(
            f"  {key}: {value!r}" for key, value in self.config.model_dump().items()
        )
        return f"{type(self).__name__}\n  collection: {self.collection!r}\n{lines}"

    def sample_item(self) -> pystac.Item | None:
        """Fetch one item from the collection to read its shape.

        Use it to discover asset names and property keys before writing a
        filter.

        Returns:
            One item, or None when the collection is empty.
        """
        found = self.client.search(
            StacQuery(collections=[self.collection], max_items=1)
        )
        return found[0] if found else None

    def list_item_properties(self) -> tuple[str, ...]:
        """Name the per-acquisition properties this collection publishes.

        These are the names `StacSourceConfig.item_properties` accepts.

        Returns:
            Property names, sorted. Empty when the collection has no items.

        Examples:
            >>> [p for p in source.list_item_properties() if "sun" in p]
            ['view:sun_azimuth', 'view:sun_elevation']
        """
        sample = self.sample_item()
        return () if sample is None else tuple(sorted(sample.properties))

    def list_asset_fields(self) -> tuple[str, ...]:
        """Name the asset fields this collection publishes.

        These are the names `StacSourceConfig.asset_fields` accepts.

        Returns:
            Field names, sorted. Empty when the collection has no items.

        Examples:
            >>> [f for f in source.list_asset_fields() if "wavelength" in f]
            ['center_wavelength', 'full_width_half_max']
        """
        sample = self.sample_item()
        if sample is None:
            return ()
        return tuple(
            sorted(
                {
                    key
                    for asset in sample.assets.values()
                    for key in read_asset_fields(asset)
                }
            )
        )

    def load(self, anchor: GeoAnchor) -> Dataset:
        """Load this collection onto an anchor's grid.

        Collection metadata becomes `ACDD` and the matched items become
        `StacMetadata`. Each variable carries its own asset's `CFVariable` and
        `Packing`, so pixels stay the published DN.

        Args:
            anchor: Grid and datetime window to load.

        Returns:
            Dataset on the anchor's exact geobox. Its arrays are
            Dask-backed unless `chunks=None` was configured.

        Raises:
            AnchorFetchError: The search matched no items.
            ValueError: Matched items disagree on one asset's decoding, or the
                loaded Dataset carries no locatable grid.

        Examples:
            >>> ds = source.load(anchor)
            >>> ds.gs.attrs.root.get(StacMetadata).properties()
            ('eo:cloud_cover', 'platform')
        """
        matched = self.client.search(self._search_query(anchor))
        if not matched:
            raise AnchorFetchError(
                f"no {self.collection!r} items matched the anchor's extent and window"
            )

        data = odc.stac.load(matched, geobox=anchor.geobox, **self._load_options())
        described = stamp_stac(
            data,
            matched,
            self.client.collection(self.collection),
            groupby=self.config.groupby,
            item_properties=self.config.item_properties,
            asset_fields=self.config.asset_fields,
        )
        return described.gs.write_crs()

    def _search_query(self, anchor: GeoAnchor) -> StacQuery:
        """Narrow this source's search to one anchor.

        The anchor's footprint and window are used unless `query` already
        pinned a datetime, which a fixed-vintage collection needs.

        Args:
            anchor: Grid and datetime window to load.

        Returns:
            Query narrowed to the anchor.
        """
        left, bottom, right, top = anchor.geobox.geographic_extent.boundingbox
        bbox = (left, bottom, right, top)
        window = (
            self.query.datetime if self.query.datetime is not None else anchor.timespan
        )
        return replace(self.query, bbox=bbox, datetime=window)

    def _load_options(self) -> dict[str, Any]:
        """Assemble the keyword arguments odc-stac's load takes.

        Returns:
            Load settings, excluding the items and the geobox.
        """
        options: dict[str, Any] = {
            "bands": self.config.bands,
            "groupby": self.config.groupby,
            "chunks": self.config.chunks,
            "resampling": self.config.resampling,
            "fail_on_error": self.config.fail_on_error,
            "preserve_original_order": self.config.preserve_original_order,
        }
        for name in ("dtype", "nodata", "stac_cfg", "pool", "progress", "patch_url"):
            value = getattr(self.config, name)
            if value is not None:
                options[name] = value
        return options

"""StacSource: one STAC collection loaded onto an anchor's grid. See StacSource for details."""
from __future__ import annotations

import copy
import warnings
from dataclasses import replace
from datetime import datetime as dt
from typing import Any, Callable, Literal, NotRequired, Protocol, Self, Sequence, TypedDict, Unpack

import pystac
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr
from odc.stac import load as odc_load
from pydantic import BaseModel, ConfigDict, Field, field_validator

from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.extensions import StacItemRecord
from geosave_engine.geodata.spatial import GeoAnchor, GeoRaster
from geosave_engine.geodata.utils.array import cf_to_da
from geosave_engine.utils.fn import UNSET, Unset

from .query import StacQuery
from .records import parse_items, select_release

Bands = list[str] | tuple[str, ...]
Resampling = str | dict[str, str]

# Options this class owns — through load_kwargs they'd contradict the anchor's grid or a named field.
_NAMED_LOAD_KWARGS = frozenset(
    {
        "bands",
        "bbox",
        "chunks",
        "crs",
        "dtype",
        "fail_on_error",
        "geobox",
        "groupby",
        "nodata",
        "patch_url",
        "resampling",
        "resolution",
    }
)


def _default_chunks() -> dict[str, int | Literal["auto"]]:
    """Default spatial Dask chunks.

    Returns:
        `{"x": 1024, "y": 1024}` — the read unit every window is cut out of.
    """
    return {"x": 1024, "y": 1024}


def _warn_unbucketed(collection: str, records: Sequence[StacItemRecord]) -> None:
    """Warn when a collection's own bucketing is about to read as single days.

    A collection dating items by validity range has already bucketed them —
    an annual map, a 16-day composite. Nothing here infers a cadence from
    that, so the steps bucket as days until the caller resamples.

    Args:
        collection: Collection being loaded, named in the warning.
        records: Parsed records for this load.
    """
    spans = [(record.start_datetime, record.end_datetime) for record in records]
    if not spans or any(start is None or end is None for start, end in spans):
        return
    days = sorted({round((end - start).total_seconds() / 86_400) for start, end in spans})  # type: ignore[operator]
    if days[0] < 1:
        return
    covered = f"{days[0]}" if len(days) == 1 else f"{days[0]}\u2013{days[-1]}"
    warnings.warn(
        f"{collection!r} items declare {covered}-day validity ranges, but these steps bucket as "
        "single days — call resample_time() to bucket them",
        stacklevel=3,
    )


class SearchClient(Protocol):
    """Structural protocol for any client that can search a STAC catalog."""

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...


class StacSourceConfig(BaseModel):
    """odc-stac load tuning for one StacSource.

    Args:
        bands: Band names to load. None loads whatever the matched items declare.
        resampling: Resampling passed to odc-stac. None keeps its
            asset-level default; a mapping selects per band.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for the spatial dims. None loads eagerly,
            a real mode of its own rather than a missing default.
        dtype: Output dtype passed to odc-stac. None keeps each collection's own.
        nodata: Nodata value passed to odc-stac, paired with `dtype` — e.g.
            `-9999` for HLS. None uses each asset's own declared nodata.
        fail_on_error: False skips a scene odc-stac fails to read instead of raising.
        release: Which single release to keep when a collection republishes
            one product over time — "latest" the newest, "nearest" the one
            closest to the anchor's window. Either searches every date
            rather than the anchor's window. None keeps every matched item,
            the normal case for an observation series.
        properties: Extension property keys to carry in each provenance
            record, keyed as STAC publishes them. None reads
            `DEFAULT_PROPERTIES`; spread it to add to them rather than
            replace, e.g. `[*DEFAULT_PROPERTIES, "s2:processing_baseline"]`.
        patch_url: Rewrite each asset's URL right before odc-stac reads it —
            e.g. re-sign, or redirect a moved bucket. None reads them as published.
        load_kwargs: Extra odc-stac `load()` kwargs this model doesn't name
            (e.g. `stac_cfg`, `fuse_func`), merged in as-is.

    Raises:
        ValueError: `load_kwargs` sets an option a named field already owns.
    """

    model_config = ConfigDict(frozen=True)

    bands: Bands | None = None
    resampling: Resampling | None = None
    groupby: Literal["solar_day", "id", "time"] = "solar_day"
    chunks: dict[str, int | Literal["auto"]] | None = Field(default_factory=_default_chunks)
    dtype: str | None = None
    nodata: float | None = None
    fail_on_error: bool = True
    release: Literal["latest", "nearest"] | None = None
    properties: list[str] | None = None
    patch_url: Callable[[str], str] | None = None
    load_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("load_kwargs")
    @classmethod
    def _validate_load_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Reject options already represented by named config fields.

        Args:
            values: The `load_kwargs` mapping being validated.

        Returns:
            The mapping unchanged.

        Raises:
            ValueError: A key collides with a named field or with the anchor's grid.
        """
        collision = set(values) & _NAMED_LOAD_KWARGS
        if collision:
            raise ValueError(f"load_kwargs duplicates named option(s) {sorted(collision)}")
        return values


class ConfigChanges(TypedDict):
    """Keyword form of `StacSourceConfig`, every field optional.

    Holds exactly `StacSourceConfig`'s own field names and types so
    `set_config` keeps typed keywords without restating their defaults.
    """

    bands: NotRequired[Bands | None]
    resampling: NotRequired[Resampling | None]
    groupby: NotRequired[Literal["solar_day", "id", "time"]]
    chunks: NotRequired[dict[str, int | Literal["auto"]] | None]
    dtype: NotRequired[str | None]
    nodata: NotRequired[float | None]
    fail_on_error: NotRequired[bool]
    release: NotRequired[Literal["latest", "nearest"] | None]
    properties: NotRequired[list[str] | None]
    patch_url: NotRequired[Callable[[str], str] | None]
    load_kwargs: NotRequired[dict[str, Any]]


class StacSource:
    """One STAC collection, loaded onto an anchor's own grid.

    Loads pixels and records which items they came from, exactly as the
    provider publishes them — no radiometric scaling. Time bucketing and
    windowing are the caller's own steps on the returned raster.

    Args:
        client: STAC client used to search for items.
        collection: STAC collection identifier.
        config: Load tuning. None takes every default.

    Examples:
        >>> source = client.source("sentinel-2-l2a").set_config(bands=["B04", "B08"])
        >>> monthly = source.load(anchor).resample_time("MS", "median")
        >>> windows = list(monthly.time_windows(4, stride=1))
    """

    def __init__(
        self,
        client: SearchClient,
        *,
        collection: str,
        config: StacSourceConfig | None = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.query: StacQuery = StacQuery(collections=[collection])
        self.config = config if config is not None else StacSourceConfig()

    def __repr__(self) -> str:
        query = {"filter": self.query.filter, "max_items": self.query.max_items, "limit": self.query.limit}
        lines = "\n".join(f"  {key}: {value!r}" for key, value in {**query, **self.config.model_dump()}.items())
        return f"{type(self).__name__}\n  collection: {self.collection!r}\n{lines}"

    def set_config(self, *, inplace: bool = False, **changes: Unpack[ConfigChanges]) -> Self:
        """Change load tuning. Only the fields named actually change.

        Args:
            inplace: Mutate self and return it. Default leaves self
                untouched and returns a copy carrying the changes.
            **changes: Any `StacSourceConfig` field.

        Returns:
            Self if `inplace`, otherwise a new StacSource.

        Raises:
            ValidationError: A value doesn't fit its field.

        Examples:
            >>> source = source.set_config(bands=["B04", "B08"], groupby="time")
        """
        target = self if inplace else copy.copy(self)
        if changes:
            # constructor, not model_copy — a bad value raises here rather than deep inside load()
            target.config = StacSourceConfig(**{**self.config.model_dump(), **changes})
        return target

    def set_query(
        self,
        *,
        inplace: bool = False,
        filter: str | Unset = UNSET,
        datetime: dt | str | tuple[dt, dt] | None | Unset = UNSET,
        max_items: int | None | Unset = UNSET,
        limit: int | None | Unset = UNSET,
    ) -> Self:
        """Change how this source searches.

        Args:
            inplace: Mutate self and return it. Default leaves self
                untouched and returns a copy carrying the changes.
            filter: CQL2 text merged into the existing filter with `and`
                (e.g. `"eo:cloud_cover <= 10"`). Every call adds another
                clause; there is no way to clear one already set.
            datetime: Search window this source uses instead of the
                anchor's — for a collection whose items are dated outside
                the window you are loading, such as a DEM or a fixed-vintage
                land cover. None searches the anchor's own window.
            max_items: Maximum matched items returned across all pages.
                None removes the client-side cap.
            limit: STAC server page-size hint. None uses the server's own.

        Returns:
            Self if `inplace`, otherwise a new StacSource.

        Raises:
            ValueError: `filter` isn't valid CQL2 text.

        Examples:
            >>> source = source.set_query(filter="eo:cloud_cover <= 10", max_items=200)
            >>> dem = client.source("cop-dem-glo-30").set_query(datetime="2021-01-01/2021-12-31")
        """
        target = self if inplace else copy.copy(self)
        if not isinstance(filter, Unset):
            target.query = target.query.set_filter(filter)
        if not isinstance(datetime, Unset):
            target.query = replace(target.query, datetime=datetime)
        if not isinstance(max_items, Unset):
            target.query = replace(target.query, max_items=max_items)
        if not isinstance(limit, Unset):
            target.query = replace(target.query, limit=limit)
        return target

    def get_bands_metadata(self) -> dict[str, Any]:
        """Fetch asset metadata for this source's collection.

        Returns:
            `{band_name: asset.extra_fields}` for one example item. Empty
            dict if the collection has no items yet.
        """
        items = self.client.search(StacQuery(collections=[self.collection], max_items=1))
        if not items:
            warnings.warn(f"No items found when fetching bands metadata for {self.collection!r}")
            return {}
        return {name: asset.extra_fields for name, asset in items[0].assets.items()}

    def get_stac_metadata(self) -> dict[str, Any]:
        """Fetch full STAC item metadata for this source's collection.

        Returns:
            `item.to_dict()` for one example item — properties, assets,
            geometry, everything available to build a CQL2 filter. Empty
            dict if the collection has no items yet.
        """
        items = self.client.search(StacQuery(collections=[self.collection], max_items=1))
        if not items:
            warnings.warn(f"No items found when fetching metadata for {self.collection!r}")
            return {}
        return items[0].to_dict()

    def load(self, anchor: GeoAnchor) -> GeoRaster:
        """Read this collection over one anchor, whole.

        Args:
            anchor: Reference geobox and datetime window.

        Returns:
            Lazy GeoRaster on `anchor.geobox`, dims `(time, band, y, x)`,
            carrying a `StacProvenance` over every matched item. Its own
            time span is the anchor's, or — when this source searches its
            own window — the span of what actually matched. Call
            `resample_time` and `time_windows` on it to reach model inputs.

        Raises:
            AnchorFetchError: Nothing matched the searched bbox and datetime
                window, or a matched item declares no usable datetime.
            ValueError: `anchor` is timeless and this source declares no
                `datetime` of its own, or the collection returned a CF
                dataset this library can't read.

        Examples:
            >>> raster = source.load(anchor)
            >>> raster.stac.items[0].properties["eo:cloud_cover"]
            4.2
        """
        config = self.config
        # an edition is chosen across dates, so neither it nor an explicit datetime reads the anchor's window
        own_window = self.query.datetime is not None or config.release is not None
        window = self.query.datetime
        if not own_window:
            if anchor.timespan is None:
                raise ValueError(
                    "StacSource.load() needs a dated anchor, or a source searching its own dates — "
                    "set_query(datetime=...) or set_config(release=...)"
                )
            start, end = anchor.start, anchor.end
            assert start is not None and end is not None
            window = (start, end)

        query = replace(self.query, bbox=anchor.geographic_bounds, datetime=window)
        items = self.client.search(query)
        if not items:
            raise AnchorFetchError(
                f"{self.collection!r}: no scenes matched the searched bbox/datetime window "
                f"(anchor at {anchor.geographic_centroid}, window {window})"
            )

        if config.release is not None:
            items = select_release(items, config.release, anchor.timespan)

        odc_kwargs: dict[str, Any] = dict(
            bands=config.bands,
            resampling=config.resampling,
            chunks=config.chunks,
            dtype=config.dtype,
            nodata=config.nodata,
            fail_on_error=config.fail_on_error,
            groupby=config.groupby,
            patch_url=config.patch_url,
            **config.load_kwargs,
        )
        ds: xr.Dataset = odc_load(items, geobox=anchor.geobox, **odc_kwargs)

        try:
            da = cf_to_da(ds)
        except ValueError as e:
            raise ValueError(f"{self.collection!r} returned an incompatible CF dataset: {e}") from e

        # config.nodata is an explicit override for whatever the adapter read off the assets
        if config.nodata is not None:
            da = da.rio.write_nodata(config.nodata, inplace=True)

        records = parse_items(items, config.properties)
        _warn_unbucketed(self.collection, records.items)
        # a source searching its own window returns steps the anchor's span doesn't cover, so let them date it
        attached = anchor.rebase(timespan=None) if own_window else anchor
        # same anchor-attach path prediction output goes through — validates the geobox, stamps the header
        return attached.to_raster(da).rebase(stac={"items": records.items})

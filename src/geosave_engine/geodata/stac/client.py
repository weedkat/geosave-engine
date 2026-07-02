from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, overload

import pystac
import planetary_computer
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from typing_extensions import Unpack
from urllib3.util import Retry

from .query import StacQuery
from geosave_engine.geodata.source.base import Source
from geosave_engine.geodata.source.sentinel_2 import Sentinel2Source
from geosave_engine.geodata.source.hls import HLSSource

if TYPE_CHECKING:
    from geosave_engine.geodata.source.base import SourceArgs

_SOURCE_REGISTRY: dict[str, type[Source]] = {
    "sentinel-2-l2a": Sentinel2Source,
    "sentinel-2-l1c": Sentinel2Source,
    "hls2-s30":       HLSSource,
    "hls2-l30":       HLSSource,
}


class StacClient:
    """STAC catalog client with search and typed source construction.

    Use classmethods to connect to a provider:

    Examples:
        >>> cdse = StacClient.cdse()
        >>> pc   = StacClient.planetary_computer()
        >>> e84  = StacClient.element84()
    """

    def __init__(self, client: Client) -> None:
        self._client = client
        self._collection_ids: set[str] | None = None
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        self._client._stac_io = StacApiIO(max_retries=retry_strategy)

    # ---------------------------------------------------------------- connect

    @classmethod
    def cdse(cls) -> StacClient:
        """Connect to Copernicus Data Space Ecosystem STAC API."""
        return cls(Client.open("https://stac.dataspace.copernicus.eu/v1/"))

    @classmethod
    def planetary_computer(cls) -> StacClient:
        """Connect to Microsoft Planetary Computer STAC API (signed assets)."""
        return cls(Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1/",
            modifier=planetary_computer.sign_inplace,
        ))

    @classmethod
    def element84(cls) -> StacClient:
        """Connect to Element84 Earth Search (AWS) STAC API."""
        return cls(Client.open("https://earth-search.aws.element84.com/v1/"))

    # ---------------------------------------------------------------- search

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]:
        """Run search and return all matching items.

        Args:
            query: ``StacQuery`` or raw pystac-client search params dict.

        Returns:
            List of matching ``pystac.Item`` objects.
        """
        if isinstance(query, StacQuery):
            query = query.to_search_params()
        return list(self._client.search(**query).items())

    def search_iter(self, query: StacQuery | dict[str, Any]) -> Iterable[pystac.Item]:
        """Run search and yield items lazily, page by page.

        Args:
            query: ``StacQuery`` or raw pystac-client search params dict.

        Returns:
            Generator of ``pystac.Item`` objects.
        """
        if isinstance(query, StacQuery):
            query = query.to_search_params()
        yield from self._client.search(**query).items()

    def get_collections(self) -> list[pystac.Collection]:
        """Return all collections available on this catalog."""
        return list(self._client.get_collections())

    # ---------------------------------------------------------------- source

    def _cached_collection_ids(self) -> set[str]:
        if self._collection_ids is None:
            self._collection_ids = {c.id for c in self.get_collections()}
        return self._collection_ids

    def _validate_collection(self, collection: str) -> None:
        ids = self._cached_collection_ids()
        if collection not in ids:
            raise ValueError(
                f"Collection {collection!r} not found on this STAC endpoint. "
                f"Call get_collections() to see what is available."
            )

    @overload
    def source(
        self,
        collection: Literal["sentinel-2-l2a", "sentinel-2-l1c"],
        **kwargs: Unpack[SourceArgs],
    ) -> Sentinel2Source: ...

    @overload
    def source(
        self,
        collection: Literal["hls2-s30", "hls2-l30"],
        **kwargs: Unpack[SourceArgs],
    ) -> HLSSource: ...

    @overload
    def source(self, collection: str, **kwargs: Unpack[SourceArgs]) -> Source: ...

    def source(self, collection: str, **kwargs: Unpack[SourceArgs]) -> Source:
        """Create a typed source for a STAC collection on this client.

        Validates that the collection exists on this endpoint before returning.
        Known collections return a typed subclass with preprocessing and filter helpers.
        Unknown collections fall back to generic ``Source`` with no preprocessing.

        Args:
            collection: STAC collection ID. Discover via ``get_collections()``.
            **kwargs: Forwarded to ``Source.__init__`` — see ``SourceArgs``.

        Returns:
            {
                "sentinel-2-l2a" | "sentinel-2-l1c": Sentinel2Source,
                "hls2-s30" | "hls2-l30": HLSSource,
                str: Source,
            }

        Raises:
            ValueError: If ``collection`` does not exist on this endpoint.

        Examples:
            >>> cdse = StacClient.cdse()
            >>> src = (
            ...     cdse.source("sentinel-2-l2a", slot_mode="monthly", composite="median")
            ...     .max_cloud_cover(20)
            ... )
        """
        self._validate_collection(collection)
        source_cls = _SOURCE_REGISTRY.get(collection, Source)
        return source_cls(self, collection_id=collection, **kwargs)

    def source_raw(self, collection: str, **kwargs: Unpack[SourceArgs]) -> Source:
        """Create a source with preprocessing disabled.

        Use when a model expects raw DN values (e.g. ``GraniteGeospatialBiomass``
        expects HLS DN, not reflectance scaled by ×0.0001).

        Args:
            collection: STAC collection ID.
            **kwargs: Forwarded to ``Source.__init__`` — see ``SourceArgs``.

        Returns:
            Source instance with preprocessing skipped on ``load()``.

        Raises:
            ValueError: If ``collection`` does not exist on this endpoint.

        Examples:
            >>> pc = StacClient.planetary_computer()
            >>> src = pc.source_raw("hls2-s30", slot_mode="daily")
        """
        return self.source(collection, **kwargs).raw()

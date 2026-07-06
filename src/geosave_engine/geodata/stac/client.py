from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import pystac
import planetary_computer
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from typing_extensions import Unpack
from urllib3.util import Retry

from .query import StacQuery
from .source import Source

if TYPE_CHECKING:
    from .source import SourceArgs


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

    def collection_ids(self) -> set[str]:
        """STAC collection IDs available on this endpoint, memoized after first call."""
        if self._collection_ids is None:
            self._collection_ids = {c.id for c in self.get_collections()}
        return self._collection_ids

    def validate_collection(self, collection_id: str) -> None:
        """Raise if `collection_id` doesn't exist on this endpoint.

        Args:
            collection_id: STAC collection ID to check.

        Raises:
            ValueError: If `collection_id` isn't in `collection_ids()`.
        """
        if collection_id not in self.collection_ids():
            raise ValueError(
                f"Collection {collection_id!r} not found on this STAC endpoint. "
                f"Call get_collections() to see what is available."
            )

    def source(self, collection_id: str, **kwargs: Unpack[SourceArgs]) -> Source:
        """Create a source for a STAC collection on this client.

        Validates that the collection exists on this endpoint before returning.
        Source always returns raw values as the provider publishes them — no
        radiometric scaling. Apply scale/offset as an explicit pipeline step.

        Args:
            collection_id: STAC collection ID. Discover via `get_collections()`.
            **kwargs: Forwarded to `Source.__init__` — see `SourceArgs`.

        Raises:
            ValueError: If `collection_id` does not exist on this endpoint.

        Examples:
            >>> cdse = StacClient.cdse()
            >>> src = cdse.source("sentinel-2-l1c", bands=["B02", "B03", "B04"], max_nodata_fraction=0.1)
            >>> tiles = src.load(anchor)
        """
        self.validate_collection(collection_id)
        return Source(self, collection_id=collection_id, **kwargs)

"""STAC catalog client."""

from __future__ import annotations

from typing import Any

import planetary_computer
import pystac
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3.util import Retry

from .query import StacQuery
from .source import StacSource

CDSE_URL = "https://stac.dataspace.copernicus.eu/v1/"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
ELEMENT84_URL = "https://earth-search.aws.element84.com/v1/"


class StacClient:
    """Search one STAC catalog and build sources on it.

    Args:
        client: Open pystac-client session.

    Examples:
        >>> client = StacClient.cdse()
        >>> source = client.source("sentinel-2-l2a")
    """

    def __init__(self, client: Client) -> None:
        """Bind the session and set its retry policy.

        Args:
            client: Open pystac-client session.
        """
        self._client = client
        self._collections: dict[str, pystac.Collection] = {}
        self._client._stac_io = StacApiIO(
            max_retries=Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET", "POST"],
            )
        )

    @classmethod
    def cdse(cls) -> StacClient:
        """Connect to the Copernicus Data Space Ecosystem catalog.

        Returns:
            Client for the CDSE STAC API.
        """
        return cls(Client.open(CDSE_URL))

    @classmethod
    def planetary_computer(cls) -> StacClient:
        """Connect to the Microsoft Planetary Computer catalog, signing assets.

        Returns:
            Client for the Planetary Computer STAC API.
        """
        return cls(
            Client.open(
                PLANETARY_COMPUTER_URL, modifier=planetary_computer.sign_inplace
            )
        )

    @classmethod
    def element84(cls) -> StacClient:
        """Connect to the Element84 Earth Search catalog.

        Returns:
            Client for the Earth Search STAC API.
        """
        return cls(Client.open(ELEMENT84_URL))

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]:
        """Run one search to completion.

        Args:
            query: Query object, or raw pystac-client search parameters.

        Returns:
            Every matching item.
        """
        params = query.to_search_params() if isinstance(query, StacQuery) else query
        return list(self._client.search(**params).items())

    def collections(self) -> set[str]:
        """Name the collections this catalog offers.

        Returns:
            Collection IDs.
        """
        return {collection.id for collection in self._client.get_collections()}

    def collection(self, collection: str) -> pystac.Collection:
        """Fetch one collection, reusing it after the first read.

        Args:
            collection: Collection ID.

        Returns:
            The collection as the catalog publishes it.

        Raises:
            ValueError: `collection` is not on this catalog.
        """
        if collection not in self._collections:
            found = self._client.get_collection(collection)
            if found is None:
                raise ValueError(
                    f"collection {collection!r} not found on this STAC endpoint; "
                    f"call collections() to see what is available"
                )
            self._collections[collection] = found
        return self._collections[collection]

    def source(self, collection: str) -> StacSource:
        """Build a source for one collection on this catalog.

        The source starts on every default. Narrow it with `set_config` and
        `set_query`.

        Args:
            collection: Collection ID. Discover them with `collections`.

        Returns:
            Source for `collection`.

        Raises:
            ValueError: `collection` is not on this catalog.

        Examples:
            >>> source = client.source("sentinel-2-l1c").set_config(bands=["B02"])
        """
        self.collection(collection)
        return StacSource(self, collection=collection)

"""Clients that search a STAC API or a STAC GeoParquet table."""

from __future__ import annotations

from os import PathLike
from pathlib import PurePosixPath
from typing import Any, Literal

import planetary_computer
import pyarrow as pa
import pystac
import rustac
from pystac.utils import datetime_to_str
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3.util import Retry

from geosave_engine.geodata.errors import CollectionNotFoundError
from geosave_engine.geodata.io import geoparquet
from geosave_engine.geodata.io.storage import absolute_location

from . import table
from .query import StacQuery
from .source import StacSource

CDSE_URL = "https://stac.dataspace.copernicus.eu/v1/"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
ELEMENT84_URL = "https://earth-search.aws.element84.com/v1/"

type StacProvider = Literal["planetary_computer", "cdse", "element84"]


class StacClient:
    """Search one STAC catalog and build sources on it.

    Args:
        client: Open pystac-client session.

    Examples:
        >>> client = StacClient.cdse()
        >>> source = client.source("sentinel-2-l2a")
    """

    def __init__(self, client: Client) -> None:
        """Bind the caller-owned session.

        Args:
            client: Open pystac-client session.
        """
        self._client = client
        self._collections: dict[str, pystac.Collection] = {}

    @classmethod
    def open(
        cls, endpoint: StacProvider | str, *, duckdb: rustac.DuckdbClient | None = None
    ) -> StacClient | StacTableClient:
        """Open a built-in provider, a STAC API URL, or a STAC GeoParquet table.

        Args:
            endpoint: Built-in provider name, STAC API URL, or the path or URL
                of a Parquet file.
            duckdb: Configured rustac session for a table, including any
                native DuckDB secrets. None creates a session.

        Returns:
            Client for the catalog, or a table client where `endpoint` names
            a table. Both search with a `StacQuery` and build sources.

        Raises:
            ValueError: `duckdb` is supplied for a STAC API endpoint.

        Examples:
            >>> StacClient.open("samples/catalog.parquet").source("forest")
        """
        # A table is searched by rustac; a STAC API is asked over HTTP.
        suffix = PurePosixPath(endpoint).suffix.lower()
        if suffix in geoparquet.FILE_SUFFIXES:
            return StacTableClient(endpoint, duckdb=duckdb)
        if duckdb is not None:
            raise ValueError("duckdb is only supported for a STAC GeoParquet table")
        if endpoint == "planetary_computer":
            return cls.planetary_computer()
        if endpoint == "cdse":
            return cls.cdse()
        if endpoint == "element84":
            return cls.element84()
        return cls(Client.open(endpoint, stac_io=_default_io()))

    @classmethod
    def cdse(cls) -> StacClient:
        """Connect to the Copernicus Data Space Ecosystem catalog.

        Returns:
            Client for the CDSE STAC API.
        """
        return cls(Client.open(CDSE_URL, stac_io=_default_io()))

    @classmethod
    def planetary_computer(cls) -> StacClient:
        """Connect to the Microsoft Planetary Computer catalog, signing assets.

        Returns:
            Client for the Planetary Computer STAC API.
        """
        return cls(
            Client.open(
                PLANETARY_COMPUTER_URL,
                modifier=planetary_computer.sign_inplace,
                stac_io=_default_io(),
            )
        )

    @classmethod
    def element84(cls) -> StacClient:
        """Connect to the Element84 Earth Search catalog.

        Returns:
            Client for the Earth Search STAC API.
        """
        return cls(Client.open(ELEMENT84_URL, stac_io=_default_io()))

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
            CollectionNotFoundError: `collection` is not on this catalog.
        """
        if collection not in self._collections:
            found = self._client.get_collection(collection)
            if found is None:
                raise CollectionNotFoundError(
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
            CollectionNotFoundError: `collection` is not on this catalog.

        Examples:
            >>> source = client.source("sentinel-2-l1c").set_config(bands=["B02"])
        """
        self.collection(collection)
        return StacSource(self, collection=collection)


class StacTableClient:
    """Search one STAC GeoParquet table with rustac and build sources on it.

    Item properties are columns of a table, so a sort or filter names them
    without the `properties.` prefix a STAC API takes.

    Args:
        path: Parquet file as `stac.table.write` stores it, as a path or URL.
        duckdb: Configured rustac session, needed for a bucket behind
            credentials or a custom endpoint. None creates a session.

    Examples:
        >>> client = StacTableClient("samples/catalog.parquet")
        >>> client.collections()
        {'forest'}
        >>> source = client.source("forest")
    """

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        duckdb: rustac.DuckdbClient | None = None,
    ) -> None:
        """Bind the table, reading nothing yet.

        Args:
            path: Parquet file, as a path or URL.
            duckdb: Caller-configured rustac session, or None to create one.
        """
        self._location = absolute_location(path)
        self._duckdb = duckdb if duckdb is not None else rustac.DuckdbClient()

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]:
        """Run one search over the table.

        Args:
            query: Query object, or raw `rustac` search parameters.

        Returns:
            Every matching Item, its asset hrefs absolute.

        Examples:
            >>> query = StacQuery(collections=["forest"])
            >>> len(client.search(query.set_filter("eo:cloud_cover <= 10")))
            3
        """
        given = query.to_search_params() if isinstance(query, StacQuery) else query
        params = dict(given)

        # rustac takes one RFC 3339 string, a range as "start/end".
        when = params.get("datetime")
        if isinstance(when, tuple):
            start, end = when
            params["datetime"] = f"{datetime_to_str(start)}/{datetime_to_str(end)}"
        elif when is not None and not isinstance(when, str):
            params["datetime"] = datetime_to_str(when)

        items = []
        for fields in self._duckdb.search(self._location, **params):
            item = pystac.Item.from_dict(fields)
            # A table stores asset hrefs relative to itself.
            item.set_self_href(self._location)
            item.make_asset_hrefs_absolute()
            items.append(item)
        return items

    def collections(self) -> set[str]:
        """Name the collections this table's rows belong to.

        Returns:
            Collection IDs.
        """
        return set(self._described())

    def collection(self, collection: str) -> pystac.Collection:
        """Read one collection of this table.

        Args:
            collection: Collection ID.

        Returns:
            The Collection the table stores under that id, or the one rustac
            derives from its rows where the table stores none.

        Raises:
            CollectionNotFoundError: No stored Collection and no row carries
                `collection`.
        """
        described = self._described()
        if collection not in described:
            raise CollectionNotFoundError(
                f"collection {collection!r} not found in {self._location}; "
                f"call collections() to see what it holds"
            )
        return described[collection]

    def source(self, collection: str) -> StacSource:
        """Build a source for one collection of this table.

        Args:
            collection: Collection ID. Discover them with `collections`.

        Returns:
            Source for `collection`, on every default.

        Raises:
            CollectionNotFoundError: `collection` is not in this table.

        Examples:
            >>> source = client.source("forest").set_config(bands=["red", "nir"])
        """
        self.collection(collection)
        return StacSource(self, collection=collection)

    def _described(self) -> dict[str, pystac.Collection]:
        """Return every collection of the table, keyed by id.

        Returns:
            {
                "<collection id>": the Collection the table stores, or the
                    one derived from the rows carrying that id,
            }
        """
        described = {}
        for fields in self._duckdb.get_collections(self._location):
            described[fields["id"]] = pystac.Collection.from_dict(fields)
        # What a writer stored says more than what rows alone can; the same
        # session reads it, so one configuration reaches a remote table.
        metadata = self._duckdb.query_to_table(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'stac-geoparquet'",
            [self._location],
        )
        for row in pa.table(metadata).to_pylist():
            described.update(table.parse_collections(row["value"]))
        return described


def _default_io() -> StacApiIO:
    """Configure retries before a built-in catalog session is opened."""
    return StacApiIO(
        max_retries=Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
    )

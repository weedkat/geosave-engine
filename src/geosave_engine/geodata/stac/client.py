from __future__ import annotations

import os
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


def credentials_for(provider: str) -> dict[str, str]:
    """Look up a STAC provider's static AWS-style raster credentials, namespaced by prefix.

    Reads ``{PROVIDER}_AWS_ACCESS_KEY_ID`` / ``{PROVIDER}_AWS_SECRET_ACCESS_KEY``
    (required) and ``{PROVIDER}_AWS_S3_ENDPOINT`` (optional) from the
    environment — e.g. ``provider="cdse"`` reads ``CDSE_AWS_ACCESS_KEY_ID`` etc.
    Namespaced per provider (not one generic ``AWS_ACCESS_KEY_ID``) so multiple
    providers' credentials can coexist in the same process without colliding.

    Ready to pass straight into ``rasterio.Env(**credentials_for(provider))``
    scoped around one provider's fetch — scoping is what keeps it correct
    under concurrent/sequential fetches for a different provider, not just
    having the right value somewhere in the environment.

    Not needed for providers that sign asset URLs instead of using static
    keys (e.g. Planetary Computer's ``sign_inplace`` modifier).

    Args:
        provider: STAC provider name — same string as ``RequireSpec.provider``
            or a ``StacClient`` classmethod name (e.g. ``"cdse"``).

    Returns:
        {
            "AWS_ACCESS_KEY_ID": str,
            "AWS_SECRET_ACCESS_KEY": str,
            "AWS_S3_ENDPOINT": str,  # only present if {PROVIDER}_AWS_S3_ENDPOINT is set
        }

    Raises:
        ValueError: If ``{PROVIDER}_AWS_ACCESS_KEY_ID`` or
            ``{PROVIDER}_AWS_SECRET_ACCESS_KEY`` isn't set.
    """
    prefix = provider.upper()
    access_key = os.environ.get(f"{prefix}_AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get(f"{prefix}_AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise ValueError(
            f"Missing credentials for provider {provider!r}: "
            f"set {prefix}_AWS_ACCESS_KEY_ID and {prefix}_AWS_SECRET_ACCESS_KEY"
        )
    creds = {"AWS_ACCESS_KEY_ID": access_key, "AWS_SECRET_ACCESS_KEY": secret_key}
    endpoint = os.environ.get(f"{prefix}_AWS_S3_ENDPOINT")
    if endpoint:
        creds["AWS_S3_ENDPOINT"] = endpoint
    return creds


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
        self._collections: set[str] | None = None
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

    @classmethod
    def local(cls, url: str) -> StacClient:
        """Connect to a self-hosted STAC API (e.g. stac-fastapi-pgstac).

        A local pgstac instance is just another STAC endpoint — same
        `.search()`/`.source()` interface as `cdse()`/`planetary_computer()`/
        `element84()`, no separate caching mechanism needed.

        Args:
            url: Base URL of the STAC API (e.g. "http://localhost:8080").
        """
        return cls(Client.open(url))

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

    def collections(self) -> set[str]:
        """STAC collection IDs available on this endpoint, memoized after first call."""
        if self._collections is None:
            self._collections = {c.id for c in self.get_collections()}
        return self._collections

    def validate_collection(self, collection: str) -> None:
        """Raise if `collection` doesn't exist on this endpoint.

        Args:
            collection: STAC collection ID to check.

        Raises:
            ValueError: If `collection` isn't in `collections()`.
        """
        if collection not in self.collections():
            raise ValueError(
                f"Collection {collection!r} not found on this STAC endpoint. "
                f"Call get_collections() to see what is available."
            )

    def source(self, collection: str, **kwargs: Unpack[SourceArgs]) -> Source:
        """Create a source for a STAC collection on this client.

        Validates that the collection exists on this endpoint before returning.
        Source always returns raw values as the provider publishes them — no
        radiometric scaling. Apply scale/offset as an explicit pipeline step.

        Args:
            collection: STAC collection ID. Discover via `get_collections()`.
            **kwargs: Forwarded to `Source.__init__` — see `SourceArgs`.

        Raises:
            ValueError: If `collection` does not exist on this endpoint.

        Examples:
            >>> cdse = StacClient.cdse()
            >>> src = cdse.source("sentinel-2-l1c", bands=["B02", "B03", "B04"], max_nodata_fraction=0.1)
            >>> tiles = src.load(anchor)
        """
        self.validate_collection(collection)
        return Source(self, collection=collection, **kwargs)

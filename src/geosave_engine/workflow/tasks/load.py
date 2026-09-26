"""Load one model source as a native lazy raster."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import cache

from prefect.concurrency.sync import concurrency
import xarray as xr
from pystac_client import Client
from pystac_client.exceptions import APIError
from requests.exceptions import RequestException

from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import RasterRequirement


@contextmanager
def source_concurrency(sources: Iterable[SourceConfig]) -> Iterator[None]:
    """Occupy configured Prefect limits while loading lazy sources."""
    names = sorted(
        {source.concurrency for source in sources if source.concurrency is not None}
    )
    if not names:
        yield
        return
    with concurrency(names, strict=True):
        yield


def load_raster(
    anchor: GeoAnchor,
    source: SourceConfig,
    requirement: RasterRequirement,
) -> xr.Dataset:
    """Load one model source on an anchor as a lazy raster.

    Args:
        anchor: Exact output grid and time window.
        source: Runtime STAC query and pixel-loading settings.
        requirement: Model-owned raster source requirement.

    Returns:
        Lazy raster selected and validated against the requirement.

    Raises:
        ValueError: If the requirement has no STAC collection or endpoints.
        ConnectionError: If no configured endpoint publishes the collection.
    """
    requirement = RasterRequirement.model_validate(requirement)
    collection = requirement.collection
    endpoints = requirement.endpoints
    if collection is None or endpoints is None:
        raise ValueError("Raster source needs collection and endpoints")

    endpoint_urls = tuple(str(endpoint) for endpoint in endpoints)
    client = _open_client(collection, endpoint_urls)
    stac = client.source(collection)
    stac.query = source.query.to_query(collection)
    stac.config = source.load
    if requirement.variables is not None:
        stac.set_config(bands=requirement.variables)
    return requirement.select_raster(stac.load(anchor))


def _endpoint_unavailable(error: Exception) -> bool:
    """Return whether an endpoint failure is safe to retry elsewhere."""
    if isinstance(error, APIError):
        status = getattr(error, "status_code", None)
        if status is not None:
            return status == 404 or status >= 500
        cause = error.__cause__ or error.__context__
        seen = {id(error)}
        while cause is not None and id(cause) not in seen:
            if isinstance(cause, RequestException) and not isinstance(
                cause, ValueError
            ):
                return True
            seen.add(id(cause))
            cause = cause.__cause__ or cause.__context__
        return False
    return isinstance(error, RequestException) and not isinstance(error, ValueError)


@cache
def _open_client(collection: str, endpoint_urls: tuple[str, ...]) -> StacClient:
    """Open and cache the first catalog publishing a collection."""
    failures = []
    last_error: Exception | None = None
    for url in endpoint_urls:
        try:
            client = Client.open(url)
            try:
                metadata = client.get_collection(collection)
            except KeyError as error:
                if error.args != (f"Collection {collection} not found on catalog",):
                    raise
                metadata = None
        except (RequestException, APIError) as error:
            if not _endpoint_unavailable(error):
                raise
            last_error = error
        else:
            if metadata is not None and metadata.id == collection:
                return StacClient(client)
            last_error = LookupError(f"Collection {collection!r} not found")
        status = getattr(last_error, "status_code", None)
        cause = f"HTTP {status}: {last_error}" if status is not None else str(last_error)
        failures.append(f"{url}: {cause}")
    raise ConnectionError(
        "STAC endpoints unavailable: " + "; ".join(failures)
    ) from last_error

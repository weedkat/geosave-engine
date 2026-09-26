"""Load one model source as a native lazy raster."""

from __future__ import annotations

import xarray as xr
from pystac_client import Client
from pystac_client.exceptions import APIError
from requests.exceptions import RequestException

from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.source import StacSource, StacSourceConfig
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import RasterRequirement


class RasterLoader:
    """Bind one model requirement to a reachable STAC source."""

    def __init__(self, requirement: RasterRequirement) -> None:
        self.requirement = RasterRequirement.model_validate(requirement)
        collection = self.requirement.collection
        endpoints = self.requirement.endpoints
        if collection is None or endpoints is None:
            raise ValueError("Raster source needs collection and endpoints")
        self.collection = collection
        self.endpoints = endpoints

    @staticmethod
    def endpoint_unavailable(error: Exception) -> bool:
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

    def open_client(self) -> StacClient:
        """Open the first endpoint that publishes the required collection."""
        failures = []
        last_error: Exception | None = None
        for endpoint in self.endpoints:
            url = str(endpoint)
            try:
                client = Client.open(url)
                try:
                    collection = client.get_collection(self.collection)
                except KeyError as error:
                    expected = (
                        f"Collection {self.collection} not found on catalog",
                    )
                    if error.args != expected:
                        raise
                    collection = None
            except (RequestException, APIError) as error:
                if not self.endpoint_unavailable(error):
                    raise
                last_error = error
            else:
                if (
                    collection is not None
                    and collection.id == self.collection
                ):
                    return StacClient(client)
                last_error = LookupError(
                    f"Collection {self.collection!r} not found"
                )
            status = getattr(last_error, "status_code", None)
            cause = (
                f"HTTP {status}: {last_error}"
                if status is not None
                else str(last_error)
            )
            failures.append(f"{url}: {cause}")
        raise ConnectionError(
            "STAC endpoints unavailable: " + "; ".join(failures)
        ) from last_error

    def load(self, config: SourceConfig, anchor: GeoAnchor) -> xr.Dataset:
        """Load, select, and validate the required raster lazily."""
        client = self.open_client()
        source = StacSource(client, collection=self.collection)
        source.query = config.query.to_query(self.collection)
        settings = config.load.model_dump()
        if self.requirement.variables is not None:
            settings["bands"] = self.requirement.variables
        source.config = StacSourceConfig.model_validate(settings)
        raster = source.load(anchor)
        return self.requirement.select_raster(raster)

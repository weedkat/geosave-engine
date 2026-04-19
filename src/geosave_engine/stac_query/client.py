from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import planetary_computer
import pystac
import pystac_client

from geosave_engine.stac_query.query import StacQuery


@runtime_checkable
class StacClient(Protocol):
    """A STAC search backend.

    Implementations wrap a specific catalog (CDSE, Planetary Computer, Earth
    Search) and return `pystac.Item` objects for a given `StacQuery`.
    """

    def search(self, query: StacQuery) -> Iterable[pystac.Item]: ...


class CdseClient:
    """Copernicus Data Space Ecosystem STAC client.

    No authentication is needed for catalog search.

    For reading assets via `stackstac`, set CDSE S3 credentials in your
    project `.env` — the GeoSave CLI loads them before running any script::

        # .env
        AWS_ACCESS_KEY_ID=<cdse-s3-access-key>
        AWS_SECRET_ACCESS_KEY=<cdse-s3-secret-key>
        AWS_S3_ENDPOINT_URL=https://eodata.dataspace.copernicus.eu

    Request S3 credentials at https://dataspace.copernicus.eu → User Settings.

    Example::

        from geosave_engine.stac_query import CdseClient, QuerySentinel2
        import stackstac

        client = CdseClient()
        items = list(client.search(
            QuerySentinel2(collections=["SENTINEL-2"], datetime="2024-01-01/2024-03-31")
            .max_cloud_cover(10)
        ))
        stack = stackstac.stack(items)  # AWS_* env vars loaded from .env
        result = stack.compute()
    """

    STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

    def __init__(self, *, stac_url: str | None = None) -> None:
        self._stac_url = stac_url or self.STAC_URL

    def search(self, query: StacQuery) -> pystac_client.ItemSearch:
        client = pystac_client.Client.open(self._stac_url)
        client.add_conforms_to("ITEM_SEARCH")
        results = client.search(**query.to_search_params())
        return results


class PlanetaryComputerClient:
    """Microsoft Planetary Computer STAC client.

    Free service — no API key required.  Assets are automatically signed
    in-place via `planetary_computer.sign_inplace`, so `stackstac` can read
    COG assets directly without any additional configuration.

    Use `"sentinel-2-l2a"` as the collection name.

    Example::

        from geosave_engine.stac_query import PlanetaryComputerClient, QuerySentinel2
        import stackstac

        client = PlanetaryComputerClient()
        items = list(client.search(
            QuerySentinel2(collections=["sentinel-2-l2a"], datetime="2023-06-01/2023-06-30")
            .max_cloud_cover(10)
        ))
        stack = stackstac.stack(items)  # items already signed
        result = stack.compute()
    """

    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(self, *, stac_url: str | None = None) -> None:
        self._stac_url = stac_url or self.STAC_URL

    def search(self, query: StacQuery) -> pystac_client.ItemSearch:
        client = pystac_client.Client.open(
            self._stac_url,
            modifier=planetary_computer.sign_inplace,
        )
        results = client.search(**query.to_search_params())
        return results

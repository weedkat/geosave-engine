from __future__ import annotations

import pystac_client
from geosave_engine.geodata.stac_client.base_client import BaseStacClient


class CdseClient(BaseStacClient):
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

    def __init__(self) -> None:
        client = pystac_client.Client.open(self.STAC_URL)
        client.add_conforms_to("ITEM_SEARCH")
        super().__init__(client=client)
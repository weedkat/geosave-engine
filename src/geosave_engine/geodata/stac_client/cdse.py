from __future__ import annotations

import pystac_client

from geosave_engine.geodata.core.client import BaseStacClient
from geosave_engine.geodata.core.constants import CDSE_STAC_URL


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

        from geosave_engine.geodata.stac_client import CdseClient
        from geosave_engine.geodata.stac_query.sentinel2 import Sentinel2Query
        import stackstac

        client = CdseClient()
        items = client.search(
            Sentinel2Query(
                collections=["sentinel-2-l1c"], datetime="2024-01-01/2024-03-31"
            ).max_cloud_cover(10)
        )
        stack = stackstac.stack(items)  # AWS_* env vars loaded from .env
        result = stack.compute()
    """

    def __init__(self) -> None:
        client = pystac_client.Client.open(CDSE_STAC_URL)
        client.add_conforms_to("ITEM_SEARCH")
        super().__init__(client=client)
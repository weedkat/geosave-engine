from __future__ import annotations

import pystac_client

from geosave_engine.geodata.core.client import BaseStacClient
from geosave_engine.geodata.core.constants import EARTH_SEARCH_STAC_URL


class EarthSearchClient(BaseStacClient):
    """Element84 Earth Search v1 STAC client.

    Free public service — no authentication required.  The `sentinel-2-l2a`
    collection serves **COG** assets from `sentinel-cogs.s3.us-west-2.amazonaws.com`
    (free, no requester-pays).  Range reads and windowed crops work reliably.

    Note: `sentinel-2-l1c` is also available but still JP2 on a requester-pays
    bucket — use `sentinel-2-l2a` or `sentinel-2-c1-l2a` for COG access.

    Example::

        from geosave_engine.geodata.stac_client import EarthSearchClient
        from geosave_engine.geodata.stac_query.sentinel2 import Sentinel2Query

        client = EarthSearchClient()
        items = client.search(
            Sentinel2Query(
                collections=["sentinel-2-l2a"], datetime="2024-06-01/2024-06-30"
            ).max_cloud_cover(20)
        )
    """

    def __init__(self) -> None:
        client = pystac_client.Client.open(EARTH_SEARCH_STAC_URL)
        super().__init__(client=client)

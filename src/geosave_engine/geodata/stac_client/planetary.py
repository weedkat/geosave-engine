from __future__ import annotations

import planetary_computer
import pystac_client

from geosave_engine.geodata.core.client import BaseStacClient
from geosave_engine.geodata.core.constants import PLANETARY_STAC_URL


class PlanetaryComputerClient(BaseStacClient):
    """Microsoft Planetary Computer STAC client.

    Free service — no API key required.  Assets are automatically signed
    in-place via `planetary_computer.sign_inplace`, so `stackstac` can read
    COG assets directly without any additional configuration.

    Use `"sentinel-2-l2a"` as the collection name.

    Example::

        from geosave_engine.geodata.stac_client import PlanetaryComputerClient
        from geosave_engine.geodata.stac_query.sentinel2 import Sentinel2Query
        import stackstac

        client = PlanetaryComputerClient()
        items = client.search(
            Sentinel2Query(
                collections=["sentinel-2-l2a"], datetime="2023-06-01/2023-06-30"
            ).max_cloud_cover(10)
        )
        stack = stackstac.stack(items)  # items already signed
        result = stack.compute()
    """

    def __init__(self) -> None:
        client = pystac_client.Client.open(PLANETARY_STAC_URL, modifier=planetary_computer.sign_inplace)
        super().__init__(client=client)

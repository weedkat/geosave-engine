from __future__ import annotations

import pystac_client
import planetary_computer
from geosave_engine.geodata.stac_client.base_client import BaseStacClient


class PlanetaryComputerClient(BaseStacClient):
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

    def __init__(self) -> None:
        client = pystac_client.Client.open(self.STAC_URL, modifier=planetary_computer.sign_inplace)
        super().__init__(client=client)

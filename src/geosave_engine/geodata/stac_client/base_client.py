import pystac
import pystac_client

from geosave_engine.geodata.stac_query.base_query import BaseStacQuery


class BaseStacClient:
    """A STAC search backend.

    Implementations wrap a specific catalog (CDSE, Planetary Computer, Earth
    Search) and return `pystac.Item` objects for a given `BaseStacQuery`.
    """

    def __init__(self, client: pystac_client.Client) -> None:
        self.client = client

    def search(self, query: BaseStacQuery) -> list[pystac.Item]:
        results = self.client.search(**query.to_search_params())
        return list(results.items())

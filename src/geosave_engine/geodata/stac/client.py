import pystac
from pystac_client import Client
from urllib3.util import Retry
from pystac_client.stac_api_io import StacApiIO
import planetary_computer

from typing import Any, Iterable

from .query import StacQuery

class StacClient:
    """A STAC search backend.

    Implementations wrap a specific catalog (CDSE, Planetary Computer, Earth
    Search) and return `pystac.Item` objects for a given `BaseStacQuery`.
    """

    def __init__(self, client: Client) -> None:
        self.client = client
        retry_strategy = Retry(
            total=5,                  # Max 5 attempts
            backoff_factor=1,         # Wait 1s, 2s, 4s, 8s... between retries
            status_forcelist=[502, 503, 504], # Retry on these specific errors
            allowed_methods=["GET", "POST"]   # STAC search can be both
        )
        self.client._stac_io = StacApiIO(max_retries=retry_strategy)
        
    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]:
        if isinstance(query, StacQuery):
            query = query.to_search_params()
        
        search_obj = self.client.search(**query)

        return list(search_obj.items())

    def search_iter(self, query: StacQuery | dict[str, Any]) -> Iterable[pystac.Item]:
        """Returns a generator that fetches pages only when needed."""
        if isinstance(query, StacQuery):
            query = query.to_search_params()
            
        search_obj = self.client.search(**query)
        yield from search_obj.items()

    def get_collections(self) -> list[pystac.Collection]:
        return list(self.client.get_collections())


    @classmethod
    def cdse(cls) -> "StacClient":
        client = Client.open("https://stac.dataspace.copernicus.eu/v1/")
        return cls(client)
    
    @classmethod
    def planetary_computer(cls) -> "StacClient":
        client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/", modifier=planetary_computer.sign_inplace)
        return cls(client)

    @classmethod
    def element84(cls) -> "StacClient":
        client = Client.open("https://earth-search.aws.element84.com/v1/")
        return cls(client)
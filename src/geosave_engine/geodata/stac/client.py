import pystac
import pystac_client
from urllib3.util import Retry
from pystac_client.stac_api_io import StacApiIO

from typing import Any, Iterable
from .query import BaseQuery

class StacClient:
    """A STAC search backend.

    Implementations wrap a specific catalog (CDSE, Planetary Computer, Earth
    Search) and return `pystac.Item` objects for a given `BaseStacQuery`.
    """

    def __init__(self, url: str) -> None:
        retry_strategy = Retry(
            total=5,                  # Max 5 attempts
            backoff_factor=1,         # Wait 1s, 2s, 4s, 8s... between retries
            status_forcelist=[502, 503, 504], # Retry on these specific errors
            allowed_methods=["GET", "POST"]   # STAC search can be both
        )
        
        stac_io = StacApiIO(max_retries=retry_strategy)
        self.client = pystac_client.Client.open(url, stac_io=stac_io)

    def search(self, query: BaseQuery | dict[str, Any]) -> list[pystac.Item]:
        if isinstance(query, BaseQuery):
            query = query.to_search_params()
        
        search_obj = self.client.search(**query)

        return list(search_obj.items())

    def search_iter(self, query: BaseQuery | dict[str, Any]) -> Iterable[pystac.Item]:
        """Returns a generator that fetches pages only when needed."""
        if isinstance(query, BaseQuery):
            query = query.to_search_params()
            
        search_obj = self.client.search(**query)
        yield from search_obj.items()

    def get_collections(self) -> list[pystac.Collection]:
        return list(self.client.get_collections())


    @classmethod
    def cdse(cls) -> "StacClient":
        return cls("https://cdse-catalog.copernicus.eu/stac/v1")
    
    @classmethod
    def planetary_computer(cls) -> "StacClient":
        return cls("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    @classmethod
    def element84(cls) -> "StacClient":
        return cls("https://earth-search.aws.element84.com/v0")
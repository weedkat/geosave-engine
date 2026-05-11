from .client import StacClient
from .query import BaseQuery, Sentinel2L1CQuery, Sentinel2L2AQuery, StacQuery

__all__ = ["BaseQuery", "Query", "StacClient", "Sentinel2L1CQuery", "Sentinel2L2AQuery", "StacQuery"]

class Query:
    """Factory for STAC queries."""
    sentinel_2_l1c = Sentinel2L1CQuery
    sentinel_2_l2a = Sentinel2L2AQuery

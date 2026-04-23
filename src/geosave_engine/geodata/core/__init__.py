"""Geodata core: base abstractions, constants, parameter bundles, and convenience
re-exports for the concrete STAC clients, queries, and ingestors.

Importing from this module is the easiest entry point::

    from geosave_engine.geodata.core import (
        Sentinel2L1CIngestor, Sentinel2Query, CdseClient,
        LoadRequest, GeoTiffOptions, SENTINEL2_L1C,
    )
"""
from geosave_engine.geodata.core.client import BaseStacClient
from geosave_engine.geodata.core.constants import (
    CDSE_STAC_URL,
    CQL2Key,
    EARTH_SEARCH_STAC_URL,
    PLANETARY_STAC_URL,
    SENTINEL2_L1C,
    SENTINEL2_L2A,
)
from geosave_engine.geodata.core.ingestion import BaseIngestor
from geosave_engine.geodata.core.query import BaseStacQuery
from geosave_engine.geodata.core.types import GeoTiffOptions, LoadRequest
from geosave_engine.geodata.ingestion.sentinel2_l1c import Sentinel2L1CIngestor
from geosave_engine.geodata.ingestion.sentinel2_l2a import Sentinel2L2AIngestor
from geosave_engine.geodata.stac_client.cdse_client import CdseClient
from geosave_engine.geodata.stac_client.element84_client import EarthSearchClient
from geosave_engine.geodata.stac_client.planetary_client import PlanetaryComputerClient
from geosave_engine.geodata.stac_query.sentinel2 import Sentinel2Query

__all__ = [
    "BaseIngestor",
    "BaseStacClient",
    "BaseStacQuery",
    "CDSE_STAC_URL",
    "CQL2Key",
    "CdseClient",
    "EARTH_SEARCH_STAC_URL",
    "EarthSearchClient",
    "GeoTiffOptions",
    "LoadRequest",
    "PLANETARY_STAC_URL",
    "PlanetaryComputerClient",
    "SENTINEL2_L1C",
    "SENTINEL2_L2A",
    "Sentinel2L1CIngestor",
    "Sentinel2L2AIngestor",
    "Sentinel2Query",
]

"""Concrete raster ingestors by satellite / product level."""
from geosave_engine.geodata.ingestion.sentinel2_l1c import Sentinel2L1CIngestor
from geosave_engine.geodata.ingestion.sentinel2_l2a import Sentinel2L2AIngestor

__all__ = [
    "Sentinel2L1CIngestor",
    "Sentinel2L2AIngestor",
]

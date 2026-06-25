from .base import BaseSource, BaseSourceSeries
from .sentinel_2 import Sentinel2L1C, Sentinel2L2A, Sentinel2L1CSeries, Sentinel2L2ASeries

__all__ = [
    "BaseSource",
    "BaseSourceSeries",
    "Sentinel2L1C",
    "Sentinel2L2A",
    "Sentinel2L1CSeries",
    "Sentinel2L2ASeries",
    "Source",
]


class Source:
    """Factory for pipeline source layers."""
    sentinel_2_l2a = Sentinel2L2A
    sentinel_2_l1c = Sentinel2L1C
    sentinel_2_l2a_series = Sentinel2L2ASeries
    sentinel_2_l1c_series = Sentinel2L1CSeries

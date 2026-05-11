from .base import BaseSource, SourceData
from .sentinel_2 import Sentinel2L1C, Sentinel2L2A

__all__ = ["BaseSource", "Sentinel2L1C", "Sentinel2L2A", "Source", "SourceData"]


class Source:
    """Factory for pipeline source layers."""
    sentinel_2_l2a = Sentinel2L2A
    sentinel_2_l1c = Sentinel2L1C
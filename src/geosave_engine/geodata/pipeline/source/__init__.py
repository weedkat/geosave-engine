from datetime import timedelta

from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import Sentinel2L1CQuery, Sentinel2L2AQuery

from .base import BaseSource, OdcLoadConfig, SourceData
from .sentinel_2 import Sentinel2L1C, Sentinel2L2A

__all__ = ["BaseSource", "OdcLoadConfig", "Sentinel2L1C", "Sentinel2L2A", "Source", "SourceData"]


class Source:
    """Factory for pipeline source layers."""

    @staticmethod
    def sentinel_2_l2a(
        layer_name: str,
        client: StacClient,
        query: Sentinel2L2AQuery | None = None,
        time_range: timedelta = timedelta(days=30),
        bands: list[str] | None = None,
        resampling: str = "bilinear",
        chunks: dict[str, int] | None = None,
        dtype: str = "float32",
    ) -> Sentinel2L2A:
        """Create a lazy Sentinel-2 L2A source layer."""
        return Sentinel2L2A(
            layer_name=layer_name,
            client=client,
            query=query if query is not None else Sentinel2L2AQuery(),
            time_range=time_range,
            odc_config=OdcLoadConfig(
                bands=bands,
                resampling=resampling,
                chunks=chunks if chunks is not None else {"x": 2048, "y": 2048},
                dtype=dtype,
            ),
        )

    @staticmethod
    def sentinel_2_l1c(
        layer_name: str,
        client: StacClient,
        query: Sentinel2L1CQuery | None = None,
        time_range: timedelta = timedelta(days=30),
        bands: list[str] | None = None,
        resampling: str = "bilinear",
        chunks: dict[str, int] | None = None,
        dtype: str = "float32",
    ) -> Sentinel2L1C:
        """Create a lazy Sentinel-2 L1C source layer."""
        return Sentinel2L1C(
            layer_name=layer_name,
            client=client,
            query=query if query is not None else Sentinel2L1CQuery(),
            time_range=time_range,
            odc_config=OdcLoadConfig(
                bands=bands,
                resampling=resampling,
                chunks=chunks if chunks is not None else {"x": 2048, "y": 2048},
                dtype=dtype,
            ),
        )

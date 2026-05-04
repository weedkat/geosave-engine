from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

import pystac
import xarray as xr
from odc.stac import load as odc_load

from geosave_engine.geodata.pipeline.anchor import Anchor
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import BaseQuery


@dataclass(frozen=True, kw_only=True)
class OdcLoadConfig:
    """Configuration passed to odc-stac loader."""

    bands: list[str] | None = None
    resampling: str = "bilinear"
    chunks: dict[str, int] = field(default_factory=lambda: {"x": 2048, "y": 2048})
    dtype: str = "float32"


@dataclass
class SourceData:
    """Typed container bundling a loaded Dataset with its originating STAC items."""

    ds: xr.Dataset
    items: list[pystac.Item]


@dataclass(frozen=True, kw_only=True)
class BaseSource(ABC):
    """Template for all satellite sources. Subclasses implement query-building and processing."""

    layer_name: str
    client: StacClient
    time_range: timedelta = field(default_factory=lambda: timedelta(days=30))
    odc_config: OdcLoadConfig = field(default_factory=OdcLoadConfig)

    def load(self, anchor: Anchor) -> SourceData:
        """Search STAC, download pixels aligned to anchor, apply source-specific processing."""
        query = self._build_query(anchor)
        items = self.client.search(query)
        if not items:
            raise ValueError(
                f"No STAC items found for layer '{self.layer_name}' "
                f"using query {query.to_search_params()}"
            )

        load_kwargs: dict = {
            "geobox": anchor.to_geobox(),
            "resampling": self.odc_config.resampling,
            "chunks": self.odc_config.chunks,
            "dtype": self.odc_config.dtype,
        }
        if self.odc_config.bands is not None:
            load_kwargs["bands"] = self.odc_config.bands

        ds = odc_load(items, **load_kwargs)
        processed = self._apply_processing(ds, items)
        return SourceData(ds=processed, items=items)

    @abstractmethod
    def _build_query(self, anchor: Anchor) -> BaseQuery:
        """Build STAC query for this source, injecting anchor bbox and datetime."""

    @abstractmethod
    def _apply_processing(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        """Apply source-specific processing (radiometry, scaling, etc.)."""

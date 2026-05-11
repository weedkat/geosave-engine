import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import timedelta

import pystac
import xarray as xr
from odc.stac import load as odc_load

from geosave_engine.geodata.pipeline.anchor import Anchor
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import StacQuery


@dataclass
class SourceData:
    """Typed container bundling a loaded Dataset with its originating STAC items."""

    ds: xr.Dataset
    items: list[pystac.Item]


@dataclass(frozen=True, kw_only=True)
class BaseSource(ABC):
    """Template for all satellite sources. Subclasses implement processing."""

    name: str
    client: StacClient
    time_range: timedelta = field(default_factory=lambda: timedelta(days=1))
    query: StacQuery
    max_nodata_fraction: float = 1.0

    # ODC load config — overridable per-source
    bands: list[str] | None = None
    resampling: str = "bilinear"
    chunks: dict[str, int] = field(default_factory=lambda: {"x": 2048, "y": 2048})
    dtype: str = "float32"

    def _build_query(self, anchor: Anchor) -> StacQuery:
        """Build STAC query injecting anchor bbox and datetime."""
        start = anchor.datetime - self.time_range
        end = anchor.datetime + self.time_range
        return replace(self.query, bbox=anchor.bbox, datetime=(start, end))

    def load(self, anchor: Anchor) -> SourceData | None:
        """Search STAC, download pixels aligned to anchor, apply source-specific processing.

        Returns None if no items found or nodata exceeds max_nodata_fraction.
        """
        query = self._build_query(anchor)
        items = self.client.search(query)
        if not items:
            warnings.warn(
                f"No STAC items found for layer '{self.name}' "
                f"using query {query.to_search_params()}"
            )
            return None

        load_kwargs: dict = {
            "geobox": anchor.to_geobox(),
            "bands": self.bands,
            "resampling": self.resampling,
            "chunks": self.chunks,
            "dtype": self.dtype,
        }

        ds = odc_load(items, **load_kwargs)
        processed = self._apply_processing(ds, items)

        if self.max_nodata_fraction < 1.0:
            all_null = xr.concat(
                [processed[v].isnull() for v in processed.data_vars], dim="_var"
            ).all("_var")
            nodata_fraction = float(all_null.mean().compute())
            if nodata_fraction > self.max_nodata_fraction:
                warnings.warn(
                    f"Layer '{self.name}' has {nodata_fraction:.1%} nodata "
                    f"(threshold: {self.max_nodata_fraction:.1%}), skipping."
                )
                return None

        return SourceData(ds=processed, items=items)

    @abstractmethod
    def _apply_processing(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        """Apply source-specific processing (radiometry, scaling, etc.)."""

import xarray as xr
from dataclasses import dataclass
from typing import Callable, cast

from geosave_engine.geodata.pipeline.source.base import SourceData

ComputeFn = Callable[["dict[str, SourceData]"], xr.DataArray]


@dataclass(frozen=True)
class Derived:
    """A layer computed from cached source data."""

    need_caches: dict[str, list[str]]
    compute_fn: ComputeFn
    layer_name: str

    def compute(self, cache: "dict[str, SourceData]") -> xr.DataArray:
        """Subset declared variables from each cache entry, then call compute_fn."""

        subsetted: dict[str, SourceData] = {}
        for name, vars_list in self.need_caches.items():
            source_data = cache[name]
            subsetted_ds = cast(xr.Dataset, source_data.ds[vars_list] if vars_list else source_data.ds)
            subsetted[name] = SourceData(ds=subsetted_ds, items=source_data.items)

        return self.compute_fn(subsetted)

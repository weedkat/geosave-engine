import warnings
import xarray as xr
from dask.diagnostics import ProgressBar

from geosave_engine.geodata.pipeline.anchor import Anchor, ANCHOR_CACHE_KEY
from geosave_engine.geodata.pipeline.derived import Derived
from geosave_engine.geodata.pipeline.source.base import BaseSource, SourceData


class Pipeline:
    """Orchestrates Anchor → Source → Derived execution with a shared download cache."""

    def __init__(
        self,
        anchor: Anchor,
        sources: BaseSource | list[BaseSource],
        deriveds: Derived | list[Derived],
    ) -> None:
        if isinstance(sources, BaseSource):
            sources = [sources]
        if isinstance(deriveds, Derived):
            deriveds = [deriveds]

        self._anchor = anchor
        self._sources = sources
        self._deriveds = deriveds

    def run(
        self, retry: int = 3, max_nodata_ratio: float = 0.0
    ) -> dict[str, xr.DataArray]:
        """Download all sources once, compute derived layers, tag metadata onto each result.

        Args:
            retry: Re-compute attempts per source when null fraction exceeds max_nodata_ratio.
            max_nodata_ratio: Maximum tolerated null fraction across all bands after compute.
                Raises RuntimeError after all retries exhausted if still exceeded.
        """
        # Phase 1: download — materialize all dask arrays once per source
        cache: dict[str, SourceData] = {}

        for source in self._sources:
            source_data = source.load(self._anchor)
            if not source_data:
                continue

            print(f"Downloading '{source.name}'...")
            
            for attempt in range(retry + 1):
                with ProgressBar():
                    computed_ds = source_data.ds.compute()

                null_frac = max(float(computed_ds[v].isnull().mean()) for v in computed_ds.data_vars)

                if null_frac <= max_nodata_ratio:
                    cache[source.name] = SourceData(ds=computed_ds, items=source_data.items)
                    break
                
                if attempt < retry:
                    warnings.warn(f"Source '{source.name}' has {null_frac:.1%} null (attempt {attempt+1}/{retry+1}), retrying...")
                else:
                    raise RuntimeError(
                        f"Source '{source.name}' exceeded null threshold ({null_frac:.1%} > {max_nodata_ratio:.1%}) "
                        f"after {retry + 1} attempts."
                    )

        # Phase 2: compute derived layers — all source data is now in-memory
        full_cache: dict = {ANCHOR_CACHE_KEY: self._anchor, **cache}
        result: dict[str, xr.DataArray] = {}
        for d in self._deriveds:
            missing = [
                k for k in d.cache_keys() if k != ANCHOR_CACHE_KEY and k not in cache
            ]
            if missing:
                raise RuntimeError(f"Derived '{d.name}' is missing source data for keys: {missing}")
                
            da = d.compute(full_cache)
            da.attrs["datetime"] = self._anchor.datetime
            da.attrs["bbox"] = self._anchor.bbox
            da.attrs["stac_item_ids"] = [
                item.id
                for cache_key in d.cache_keys()
                if cache_key != ANCHOR_CACHE_KEY
                for item in cache[cache_key].items
            ]
            result[d.name] = da

        return result

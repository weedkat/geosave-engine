from dataclasses import dataclass
from typing import Annotated, Callable, Literal

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr

from geosave_engine.geodata.pipeline.anchor import ANCHOR_CACHE_KEY
from geosave_engine.geodata.pipeline.source.base import SourceData

SourceKey = Annotated[str, "Name of a Source layer to pull from the cache"]
ComputeFn = Callable[["dict[SourceKey, SourceData]"], xr.DataArray]
TimeReduce = Literal["median", "mean", "first", "last"]


@dataclass(frozen=True)
class Derived:
    """A layer computed from cached source data."""

    name: str
    sources: SourceKey | list[SourceKey]
    compute_fn: ComputeFn

    def cache_keys(self) -> tuple[SourceKey, ...]:
        """Return the declared cache keys as a normalized tuple."""
        if isinstance(self.sources, str):
            return (self.sources,)
        return tuple(self.sources)

    def compute(self, cache: dict) -> xr.DataArray:
        """Forward only declared cache keys to compute_fn after validating they exist."""
        return self.compute_fn({k: cache[k] for k in self.cache_keys()})

    @classmethod
    def from_cache(
        cls,
        name: str,
        sources: "SourceKey | list[SourceKey]",
        compute_fn: "ComputeFn",
    ) -> "Derived":
        """Create a derived layer with a custom compute function over cache entries."""
        return cls(name=name, sources=sources, compute_fn=compute_fn)

    @classmethod
    def image_from_source(
        cls,
        name: str,
        source: "SourceKey",
        bands: "list[str] | None" = None,
        reduce: "TimeReduce | None" = "median",
    ) -> "Derived":
        """Create an image layer by stacking bands from a single source.

        Args:
            name: Name for this derived layer.
            source: Cache key of the source to pull from.
            bands: Band variable names to stack. None uses all vars in dataset order.
            reduce: Time reduction applied after stacking. None keeps the time dimension,
                producing shape (time, band, y, x). Any reduction produces (band, y, x).
        """

        def compute_fn(cache: dict) -> xr.DataArray:
            ds = cache[source].ds
            band_names = bands if bands is not None else list(ds.data_vars)
            stacked = xr.concat([ds[b] for b in band_names], dim="band").assign_coords(
                band=band_names
            )
            if reduce is None:
                return stacked.transpose("time", "band", ...)
            if reduce == "median":
                return stacked.median(dim="time")
            if reduce == "mean":
                return stacked.mean(dim="time")
            if reduce == "first":
                return stacked.isel(time=0)
            if reduce == "last":
                return stacked.isel(time=-1)

        return cls(name=name, sources=[source], compute_fn=compute_fn)

    @classmethod
    def label_from_anchor(
        cls,
        name: str,
        remap: "dict[int, int] | None" = None,
    ) -> "Derived":
        """Create a derived layer from the anchor's label raster.

        Args:
            name: Name for this derived layer.
            remap: Optional class-value remapping applied to anchor.label.
                Raises ValueError if any pixel value is absent from the dict.
        """

        def compute_fn(cache: dict) -> xr.DataArray:
            label = cache[ANCHOR_CACHE_KEY].label
            if label is None:
                raise ValueError(
                    "Anchor.label is None — build Anchor with from_tiff(load_label=True)"
                )
            if remap is None:
                return label

            unique_vals = set(int(v) for v in np.unique(label.values))
            unmapped = unique_vals - set(remap.keys())
            if unmapped:
                raise ValueError(f"Unmapped label values: {sorted(unmapped)}")

            return label.copy(data=np.vectorize(remap.__getitem__)(label.values))

        return cls(name=name, sources=[ANCHOR_CACHE_KEY], compute_fn=compute_fn)

import xarray as xr
from dask.diagnostics import ProgressBar

from geosave_engine.geodata.pipeline.anchor import Anchor
from geosave_engine.geodata.pipeline.derived import Derived
from geosave_engine.geodata.pipeline.source.base import BaseSource, SourceData

Layer = Anchor | BaseSource | Derived


class Pipeline:
    """Orchestrates Anchor → Source → Derived execution with a local download cache."""

    def __init__(self, *layers: Layer) -> None:
        anchors = [layer for layer in layers if isinstance(layer, Anchor)]
        if len(anchors) != 1:
            raise ValueError(f"Pipeline requires exactly one Anchor, got {len(anchors)}")
        self._layers = layers
        self._anchor: Anchor = anchors[0]

    def run(self) -> dict[str, xr.DataArray]:
        """Load sources once, compute derived layers, tag datetime and bbox onto each result."""
        anchor = self._anchor
        sources = [layer for layer in self._layers if isinstance(layer, BaseSource)]
        derived_layers = [layer for layer in self._layers if isinstance(layer, Derived)]

        cache: dict[str, SourceData] = {}
        for source in sources:
            cache[source.layer_name] = source.load(anchor)

        result: dict[str, xr.DataArray] = {}
        for d in derived_layers:
            missing = [k for k in d.need_caches if k not in cache]
            if missing:
                raise KeyError(
                    f"Derived '{d.layer_name}' needs cache keys not loaded by any Source: {missing}"
                )
            with ProgressBar(f"Computing derived layer '{d.layer_name}'"):
                da = d.compute(cache)
            da.attrs["datetime"] = anchor.datetime
            da.attrs["bbox"] = anchor.bbox
            result[d.layer_name] = da

        return result

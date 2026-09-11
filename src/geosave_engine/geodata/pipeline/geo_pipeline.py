"""One anchor to an ingested surface: the workspace preprocessing seam."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from geosave_engine.geodata.core import stack

if TYPE_CHECKING:
    from geosave_engine.geodata.core import GeoAnchor
    from geosave_engine.geodata import Dataset, DataTree
    from geosave_engine.geodata.stac import StacSource


class GeoPipeline(ABC):
    """Turn one anchor into an ingested raster or stack.

    A workspace subclasses this, overriding `sources` and usually
    `preprocess`; `fetch` and `ingest` compose those hooks. Hooks take and
    return lazy xarray, so the chain stays a Dask graph until a driver runs.

    Examples:
        >>> class Pipeline(GeoPipeline):
        ...     def sources(self) -> dict[str, StacSource]:
        ...         return {"sentinel-2-l2a": client.source("sentinel-2-l2a")}
        >>> tree = Pipeline().ingest(anchor)
    """

    @abstractmethod
    def sources(self) -> dict[str, StacSource]:
        """Name the STAC sources `fetch` loads for one anchor.

        Returns:
            Group name mapped to its source, in the order `fetch` reads them.
            An empty mapping is valid only when `fetch` is also overridden to
            read something that is not STAC.
        """

    def fetch(self, anchor: GeoAnchor) -> dict[str, Dataset]:
        """Load every declared source over one anchor.

        Args:
            anchor: Grid and time window to load.

        Returns:
            Lazy rasters keyed by group name, in `sources` order. Grids are
            not reconciled here; `preprocess` composes them into a stack.

        Raises:
            NotImplementedError: `sources` is empty and this method was not
                overridden.
            AnchorFetchError: A source matched no items for the anchor.
        """
        sources = self.sources()
        if not sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares no sources() -- "
                "override sources() or fetch()"
            )
        return {name: source.load(anchor) for name, source in sources.items()}

    def preprocess(
        self, raw: dict[str, Dataset]
    ) -> dict[str, Dataset] | Dataset | DataTree:
        """Derive the final layers from the fetched rasters.

        Pixel math and metadata only, no I/O. Keep every operation
        chunk-local so the result stays lazy. Returns `raw` unchanged by
        default.

        Args:
            raw: Rasters `fetch` loaded, keyed by group name.

        Returns:
            The surface to ingest: a group mapping passed to `stack`, an
            already-built stack, or one bare raster ingested on its own.
        """
        return raw

    def ingest(self, anchor: GeoAnchor) -> DataTree | Dataset:
        """Fetch and preprocess one anchor.

        Args:
            anchor: Anchor to ingest.

        Returns:
            Lazy stack of the final layers, or the single raster `preprocess`
            returned. No pixel has been read.

        Raises:
            ValueError: `preprocess` returned an empty mapping, or groups
                whose grids disagree.
        """
        surface = self.preprocess(self.fetch(anchor))
        ingested = stack(surface) if isinstance(surface, dict) else surface
        return cast("DataTree | Dataset", ingested)

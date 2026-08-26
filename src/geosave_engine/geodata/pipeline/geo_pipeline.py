"""GeoPipeline: turn one anchor into an ingested surface. See GeoPipeline for details."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, Unpack, cast

import torch

from geosave_engine.geodata.spatial import (
    DEFAULT_LAYER,
    GeoAnchor,
    GeoRaster,
    GeoStack,
    GeoTileStack,
    LayerName,
)
from geosave_engine.geodata.stac.source import StacSource

if TYPE_CHECKING:
    from geosave_engine.geodata.datastore import LitDataStore
    from geosave_engine.geodata.utils.io import ZarrOptions
    from geosave_engine.geodata.extensions import TilerMode
    from geosave_engine.geodata.spatial import ContextFn, TensorSample, TimeWindow


class TileOptions(TypedDict, total=False):
    """How `windows` cuts a surface — `GeoStack.tiles`' own keywords.

    Args:
        tile_size_px: Window side length in pixels. None uses the shorter axis.
        stride_px: Distance between window origins. None uses `tile_size_px`.
        overlap: Forwarded to the tiler; wins over `stride_px` when both are set.
        mode: How a trailing window's overhang is filled.
        vector: True gives each window the reference layer's features.
        time: `(length, stride)` in reference-layer steps, or a bare length.
        name: Extra text folded into each cut's `group_id`.
        context_fn: Called with each window's own reference tile; its result
            becomes that window's `model_context`.
    """

    tile_size_px: NotRequired[int | None]
    stride_px: NotRequired[int | None]
    overlap: NotRequired[int | float | tuple[int, int] | None]
    mode: NotRequired[TilerMode]
    vector: NotRequired[bool]
    time: NotRequired[TimeWindow | None]
    name: NotRequired[str | None]
    context_fn: NotRequired[ContextFn | None]


class GeoPipeline(ABC):
    """Turn one anchor into an ingested surface, ready to write or window.

    Override `sources()` and `preprocess()`; `fetch()` and `ingest()` rarely
    need it. Hooks take and return unbounded types, so the chain stays a
    dask graph until `ingest_to_zarr` writes it.

    Examples:
        >>> class Pipeline(GeoPipeline):
        ...     def sources(self) -> dict[LayerName, StacSource]:
        ...         return {"image": client.source("sentinel-2-l2a")}
        >>> Pipeline().ingest_to_zarr(anchor, "data/raw")
    """

    @abstractmethod
    def sources(self) -> dict[LayerName, StacSource]:
        """Named sources `fetch()` loads for one anchor.

        Returns:
            Layer name to StacSource, in the order `fetch()` returns them.
            An empty mapping is only valid alongside an overridden `fetch()`,
            which is how a pipeline reads something that isn't STAC.
        """

    def fetch(self, anchor: GeoAnchor) -> dict[LayerName, GeoRaster]:
        """Load every source over one anchor.

        Args:
            anchor: Anchor to load.

        Returns:
            Lazy layers keyed by source name, in `sources()` order. Nothing
            here asserts they share a grid — compose them into a GeoStack
            once `preprocess` has aligned them.

        Raises:
            NotImplementedError: `sources()` declares none and this method
                wasn't overridden.
            AnchorFetchError: A source matched no scenes for this anchor.
        """
        sources = self.sources()
        if not sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares no sources() — override sources() or fetch()"
            )
        return {name: source.load(anchor) for name, source in sources.items()}

    def preprocess(self, raw: dict[LayerName, GeoRaster]) -> dict[LayerName, GeoRaster] | GeoStack | GeoRaster:
        """Derive the final layers from the fetched ones.

        Pixel math and metadata, no I/O. Keep every operation chunk-local —
        elementwise, or windowed with a bounded halo — so the result stays
        lazy and the surface never has to fit in memory.

        Args:
            raw: Layers `fetch()` loaded, keyed by source name.

        Returns:
            What this pipeline ingests — a dict whose first key anchors the
            stack, a GeoStack whose own reference layer is already set, or a
            single GeoRaster written as a plain raster store. Default:
            passthrough.
        """
        return raw

    def ingest(self, anchor: GeoAnchor) -> GeoStack | GeoRaster:
        """Fetch and preprocess one anchor.

        Args:
            anchor: Anchor to ingest.

        Returns:
            Lazy GeoStack of the final layers, or the single GeoRaster
            `preprocess` returned. No pixel has been read.

        Raises:
            ValueError: `preprocess` returned an empty mapping, or layers
                whose geoboxes disagree.
        """
        result = self.preprocess(self.fetch(anchor))
        if isinstance(result, (GeoRaster, GeoStack)):
            return result
        return GeoStack(result, reference_layer=next(iter(result), None))

    def windows(self, anchor: GeoAnchor, **tiles: Unpack[TileOptions]) -> Iterator[GeoTileStack]:
        """Ingest one anchor and cut it into model windows.

        Args:
            anchor: Anchor to ingest.
            **tiles: Cutting options — see `TileOptions`.

        Yields:
            One GeoTileStack per window. A `preprocess` that returned one
            bare raster is packed under `DEFAULT_LAYER` first, so every
            window is layer-keyed whatever the pipeline produced. Lazy —
            no pixel is read here.

        Raises:
            ValueError: `time` was given and the reference layer is timeless,
                or a layer has no step over some time window.
        """
        surface = self.ingest(anchor)
        if isinstance(surface, GeoRaster):
            surface = GeoStack({DEFAULT_LAYER: surface})
        yield from surface.tiles(**tiles)

    def ingest_to_tensor(
        self,
        anchor: GeoAnchor,
        *,
        batch_size: int = 1,
        bands: dict[LayerName, list[str]] | None = None,
        drop_last: bool = False,
        **tiles: Unpack[TileOptions],
    ) -> Iterator[TensorSample]:
        """Ingest one anchor straight into batches a model can read.

        The prediction path — no store is written and no DataLoader is
        needed. Pixels are read one batch at a time, so a surface far larger
        than memory streams through.

        Args:
            anchor: Anchor to ingest.
            batch_size: Windows collated into each batch, and read in one
                dask pass — larger batches share more chunk reads.
            bands: Per layer, band names to keep in that order. A layer
                absent here keeps every band.
            drop_last: True discards a trailing batch smaller than
                `batch_size`, as a DataLoader would.
            **tiles: Cutting options — see `TileOptions`.

        Yields:
            {
                "layers": {
                    "<layer>": torch.Tensor,  # (batch, band, y, x) or (batch, time, band, y, x)
                },
                "anchor": [GeoAnchor],  # one per window, in batch order
                "model_context": {
                    "<key>": torch.Tensor | list | str | None,
                },
            }

        Raises:
            ValueError: `batch_size` isn't positive, `time` was given and the
                reference layer is timeless, or a layer has a time gap.

        Examples:
            >>> for batch in Pipeline().ingest_to_tensor(anchor, tile_size_px=512, batch_size=8):
            ...     logits = model(batch["layers"]["image"])
        """
        from geosave_engine.geodata.datasets import stack_samples
        from geosave_engine.geodata.spatial import read_windows, tensor_context

        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        for group in batched(self.windows(anchor, **tiles), batch_size):
            if drop_last and len(group) < batch_size:
                return
            # one dask pass over the whole batch, so windows sharing a chunk read it once
            pixels = read_windows(list(group), bands)
            samples = [
                {
                    "layers": {name: torch.from_numpy(array) for name, array in read.items()},
                    "anchor": window.anchor,
                    "model_context": tensor_context(window.model_context),
                }
                for window, read in zip(group, pixels, strict=True)
            ]
            yield cast("TensorSample", stack_samples(samples))

    def ingest_to_litdata(
        self,
        anchor: GeoAnchor,
        store: LitDataStore,
        *,
        write_mode: Literal["append", "overwrite"] | None = None,
        read_batch: int = 16,
        **tiles: Unpack[TileOptions],
    ) -> Path | str:
        """Ingest one anchor and pack its windows into a litdata store.

        The dataset path — every window becomes one training sample, keyed by
        layer, carrying the georeference numpy loses. Takes a built store, not
        a path, so one store grows across many anchors under one locked config.

        Args:
            anchor: Anchor to ingest.
            store: Store to write into, already configured.
            write_mode: None raises if the store path already holds one,
                `"append"` grows it, `"overwrite"` replaces it. Named apart
                from `TileOptions`' own `mode`, which is the tiler's.
            read_batch: Windows a worker reads in one Dask pass. Larger
                batches collapse more shared chunk reads, at more memory
                held per worker.
            **tiles: Cutting options — see `TileOptions`.

        Returns:
            The store root.

        Raises:
            ValueError: This anchor yielded no window, or a layer is named
                `"geo"`/`"model_context"`, which a sample reserves.

        Examples:
            >>> store = LitDataStore("data/train", chunk_bytes="64MB")
            >>> for anchor in anchors:
            ...     Pipeline().ingest_to_litdata(anchor, store, tile_size_px=512, write_mode="append")
        """
        from geosave_engine.geodata.datastore.litdata import batch_to_samples

        if read_batch < 1:
            raise ValueError(f"read_batch must be positive, got {read_batch}")
        batches = [list(group) for group in batched(self.windows(anchor, **tiles), read_batch)]
        if not batches:
            raise ValueError(f"no window cut from {anchor.stem} — nothing to write")
        return store.write(batches, fn=batch_to_samples, mode=write_mode)

    def ingest_to_zarr(
        self,
        anchor: GeoAnchor,
        root: str | Path,
        *,
        chunk_px: int | None = 512,
        progress: bool = True,
        **options: Unpack[ZarrOptions],
    ) -> Path:
        """Ingest one anchor and write it, the point every pixel is finally read.

        A GeoStack writes one zarr group per layer, every layer computing
        together; a single GeoRaster writes the store root, readable by
        anything that reads Zarr.

        Args:
            anchor: Anchor to ingest.
            root: Directory the store is written under. Its name comes from
                the result's own `anchor.stem`, so re-ingesting an anchor
                overwrites rather than duplicates.
            chunk_px: Spatial (y/x) chunk side length. `time` is never split.
            progress: Show a dask progress bar while pixels compute.
            **options: Passed to `xarray.Dataset.to_zarr` — see `ZarrOptions`.

        Returns:
            Written store path.
        """
        result = self.ingest(anchor)
        path = Path(root) / f"{result.anchor.stem}.zarr"
        return result.to_zarr(path, chunk_px=chunk_px, progress=progress, **options)

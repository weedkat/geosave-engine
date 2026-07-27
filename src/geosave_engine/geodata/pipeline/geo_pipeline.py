from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Iterator

import torch

from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.stac.source import StacSource, download
from geosave_engine.geodata.tile import GeoAnchor, GeoTile, GeoStack

LayerName = str

log = logging.getLogger(__name__)


class GeoPipeline(ABC):
    """Build one or more GeoStacks for one anchor.

    Override `sources()` (named `StacSource`s to fetch) and, if needed,
    `preprocess` (derive final layers from the fetched ones). See `fetch`
    for how sources compose into aligned samples.

    Examples:
        >>> class ToyPipeline(GeoPipeline):
        ...     def __init__(self) -> None:
        ...         self.client = my_stac_client
        ...
        ...     def sources(self) -> dict[str, StacSource]:
        ...         return {"image": self.client.source("collection-id")}
        ...
        >>> pipeline = ToyPipeline()
        >>> for stack in pipeline.ingest(anchor):
        ...     stack.plot()
    """

    def sources(self) -> dict[str, StacSource]:
        """Named sources `ingest` composes for one anchor.

        Override — build the STAC client once in `__init__` (e.g.
        `self.client = StacClient.planetary_computer()`) and store it as an
        attribute, so this method only ever builds cheap `StacSource`
        objects from it. `StacClient.collections()` (used by `.source()`'s
        own collection-existence check) memoizes after its first call, so
        calling this on every access stays cheap too — no need to cache it.

        Default: none.
        """
        return {}

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        """Derive final layers from fetched raw layers. Pure — no I/O.

        Plain dict in, plain dict out — deliberately not GeoStack here:
        building `{"layer": tile, ...}` by hand is the easy, natural shape
        for an override to construct (see `workspace/modules/data_pipeline.py`'s
        `Pipeline.preprocess`); `ingest()` wraps the result back into a
        GeoStack itself.

        Args:
            raw: Layer name to GeoTile map — one already-bucketed sample
                composed from each source's own `.load(anchor)` stream.

        Returns:
            Layer name to GeoTile map. Default: passthrough.
        """
        return raw

    def context(self, tiles: dict[LayerName, GeoTile]) -> dict[str, torch.Tensor]:
        """Extra per-sample tensor keys to merge into a rendered sample.

        Override — derive whatever a specific model chain needs from these
        tiles' anchors (e.g. `temporal_coords`/`location_coords` for a
        Prithvi-family encoder). Must return tensors, no batch dim —
        `stack_samples` only stacks `torch.Tensor` values into a batch;
        anything else passes through unbatched. Forwarded straight into
        `GeoStack.to_tensor` as its `context_fn` by `ingest_to_tensor`
        below — pass this same bound method as a `GeoDataset`'s own
        `context_fn` to get identical derivation for offline training
        reading this pipeline's saved output.

        Args:
            tiles: Layer name to GeoTile map for one aligned, preprocessed
                sample — same shape `preprocess` receives.

        Returns:
            Extra keys to merge into the sample. Default: none.
        """
        return {}

    def fetch(self, anchor: GeoAnchor) -> Iterator[dict[str, GeoTile]]:
        """Every raw sample one anchor becomes, aligned across sources.

        Each source's own `.load(anchor, lazy_load=True)` yields its own
        independent stream of already-bucketed, still-lazy `GeoTile`s (see
        `StacSource.load`) — no two sources' streams are guaranteed to line
        up by position, so samples get composed by real time overlap
        instead: whichever source produced the most tiles for this anchor
        drives iteration, every other source is searched for the tile
        whose own window covers the driving tile's start. Nothing
        downloads until a sample's alignment is fully confirmed, and each
        real tile downloads at most once even if it ends up matching
        several driving tiles (e.g. one static DEM tile spanning many
        Sentinel-2 months).

        No usable data anywhere for this anchor (`AnchorFetchError` from
        some source's own search) or no aligned tile in some other source
        for a given sample is the same outcome either way — skipped, not
        raised.

        Args:
            anchor: Raw anchor, e.g. straight off an `AnchorSource`.

        Yields:
            Layer name to GeoTile map, one per aligned sample, computed.
        """
        sources = self.sources()
        if not sources:
            return

        try:
            streams = {name: list(source.load(anchor, lazy_load=True)) for name, source in sources.items()}
        except AnchorFetchError as e:
            log.debug("No usable data for anchor %s, skipping: %s", anchor.stem, e)
            return

        reference_name = max(streams, key=lambda name: len(streams[name]))
        downloaded: dict[int, GeoTile] = {}

        for ref_tile in streams[reference_name]:
            raw = {reference_name: ref_tile}
            for name, tiles in streams.items():
                if name == reference_name:
                    continue
                match = next((t for t in tiles if t.start <= ref_tile.start <= t.end), None)
                if match is None:
                    break
                raw[name] = match
            else:
                try:
                    for name, tile in raw.items():
                        if id(tile) not in downloaded:
                            downloaded[id(tile)] = download(
                                tile, max_nodata_fraction=sources[name].max_nodata_fraction
                            )
                except IOError as e:
                    log.debug("Sample not usable, skipping: %s", e)
                    continue
                yield {name: downloaded[id(tile)] for name, tile in raw.items()}

    def ingest(self, anchor: GeoAnchor) -> Iterator[GeoStack]:
        """Every sample one raw anchor becomes.

        Built on `fetch()` — same skip-on-empty/skip-on-misalignment
        behavior. `preprocess()` runs once per composed sample; only its
        own `AnchorFetchError` is treated as skip-not-crash, anything else
        propagates.

        Args:
            anchor: Raw anchor, e.g. straight off an `AnchorSource`.

        Yields:
            One `GeoStack` per aligned, preprocessed sample.
        """
        for raw in self.fetch(anchor):
            try:
                yield GeoStack(**self.preprocess(raw))
            except AnchorFetchError as e:
                log.debug("Sample not usable, skipping: %s", e)
                continue

    def ingest_to_tensor(
        self,
        anchor: GeoAnchor,
        *,
        sel_bands: dict[str, list[str]] | None = None,
        dtype_override: dict[str, torch.dtype] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Ingest one anchor, yielding rendered tensor samples without saving.

        For predicting/streaming straight from a live source — no disk
        round trip. Same one-anchor contract as `ingest` — looping over many
        anchors is the caller's own plain loop around this, not this
        method's job (see class docstring). Plain generator, not a Dataset:
        wrap the caller's loop in a one-off IterableDataset if a DataLoader
        needs one (and shard by `torch.utils.data.get_worker_info()` there
        if using `num_workers>1`, since this generator itself has no
        worker-awareness). Built on `ingest` — same
        bucketing/reducing/stacking, same skip-on-empty-bucket behavior.
        Every yielded sample is run through `self.context` (override to add
        keys — see there), same as `preprocess`, not a call-time override.

        Args:
            anchor: Raw anchor, e.g. straight off an `AnchorSource`.
            sel_bands: Layer name to band names to keep. Default keeps all
                bands each tile carries.
            dtype_override: Layer name to torch dtype to cast that layer's
                tensor to. Default keeps the tensor's ingested dtype.

        Yields:
            Tensor dict per sample, rendered via `GeoStack.to_tensor`.
        """
        for stack in self.ingest(anchor):
            yield stack.to_tensor(sel_bands, dtype_override, self.context)

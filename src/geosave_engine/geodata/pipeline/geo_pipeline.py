from __future__ import annotations

import logging
from abc import ABC
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

import torch
from tqdm import tqdm

from geosave_engine.geodata.tile import GEOSTACK_SUFFIX, GeoAnchor, GeoTile, GeoStack
from geosave_engine.geodata.pipeline.manifest import ManifestWriter

log = logging.getLogger(__name__)


class SourceProtocol(Protocol):
    """Structural interface for anything a GeoPipeline can declare in `sources`."""

    def load(self, anchor: GeoAnchor) -> GeoTile: ...


class GeoPipeline(ABC):
    """Build model input (a GeoStack) for one anchor.

    Split in two steps: `fetch` (I/O — pull raw layers from `sources`) then
    `preprocess` (pure — derive final layers, no I/O). Building a training
    dataset from many anchors is a separate, external concern (see
    `save_dataset`), not something a live-serving use of the same pipeline
    would ever need.

    Building a Dataset from anchors — saved or streamed — is not this
    class's job either; see `save_dataset`/`stream_ingest`, or a
    LightningDataModule that constructs `GeoDataset` directly.

    Examples:
        >>> class ToyPipeline(GeoPipeline):
        ...     @cached_property
        ...     def sources(self) -> dict[str, SourceProtocol]:
        ...         return {"image": my_stac_client.source("collection-id")}
        ...
        >>> pipeline = ToyPipeline()
        >>> layers = pipeline.ingest(anchor)
    """

    @property
    def sources(self) -> dict[str, SourceProtocol]:
        """Named sources `fetch` loads for one anchor.

        Override — use `cached_property` if building a source needs a live
        client (STAC auth etc), so constructing the pipeline just to call
        `.context()` stays network-free.

        Default: none. Override this for the common case (one `.load(anchor)`
        call per named layer), or override `fetch` directly for pipelines
        whose anchor already carries its data (no I/O left to do).
        """
        return {}

    def fetch(self, anchor: GeoAnchor) -> dict[str, GeoTile]:
        """Load every declared source for one anchor. I/O only.

        Args:
            anchor: Spatial/date anchor produced by an ingest source.

        Returns:
            Layer name to raw (unprocessed) GeoTile map.

        Raises:
            AnchorFetchError: A source has no usable data for this anchor —
                raised by the source itself (see e.g. `StacSource.load`).
        """
        return {name: source.load(anchor) for name, source in self.sources.items()}

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        """Derive final layers from fetched raw layers. Pure — no I/O.

        Args:
            raw: Layer name to GeoTile map, as returned by `fetch`.

        Returns:
            Layer name to GeoTile map. Default: passthrough.
        """
        return raw

    def ingest(self, anchor: GeoAnchor) -> dict[str, GeoTile]:
        """Build one or more GeoTile layers for one anchor.

        Args:
            anchor: Spatial/date anchor produced by an ingest source.

        Returns:
            Layer name to GeoTile map.
        """
        return self.preprocess(self.fetch(anchor))

    @staticmethod
    def context(tiles: dict[str, GeoTile]) -> dict[str, Any]:
        """Per-sample metadata attached to a rendered sample's ``"context"`` key.

        Override to pull whatever fields matter for this pipeline — plain
        Python, no fixed registry. Default: none.

        Args:
            tiles: Layer name to GeoTile map for one anchor.

        Returns:
            Empty dict unless overridden.
        """
        return {}


def save_dataset(
    pipeline: GeoPipeline,
    anchors: Iterable[GeoAnchor],
    root: str | Path,
    *,
    limit: int | None = None,
    save_stac: bool | Sequence[str] = False,
) -> None:
    """Ingest anchors through pipeline, saving each as a GeoStack.

    Building a training dataset in bulk — a separate concern from what
    GeoPipeline itself does (build model input for one anchor). Resumable:
    skips anchors already done or errored on a prior run.

    Args:
        pipeline: Pipeline whose ``ingest()`` builds each anchor's layers.
        anchors: Anchors to ingest — typically ``source.to_anchors(limit=...)``,
            but any iterable works (e.g. a hand-built list for anchors that
            don't fit an existing AnchorSource).
        root: Workspace root; one subdirectory per anchor created inside.
        limit: Cap on how many anchors to consider from ``anchors`` this
            call — a quick way to test against a handful of anchors without
            re-slicing the iterable yourself. ``None`` considers all of them.
        save_stac: Forwarded to ``GeoStack.save`` — ``True``/``False`` for
            every layer, or a list of layer names to save STAC provenance
            for only those (e.g. just the one real STAC-sourced layer,
            skipping derived layers that would just duplicate its sidecar).
    """
    root = Path(root)
    manifest = ManifestWriter(root)
    layer_metadata: dict[str, dict[str, Any]] = {}

    if limit is not None:
        anchors = islice(anchors, limit)

    for anchor in tqdm(anchors, desc=f"Saving {root.name}", unit="anchor", total=limit):
        stem = anchor.stem
        if manifest.is_processed(stem):
            log.debug("Skipping anchor (already processed): %s", stem)
            continue
        manifest.add(stem)
        try:
            layers = pipeline.ingest(anchor)
            store_name = f"{stem}{GEOSTACK_SUFFIX}"
            GeoStack(**layers).save(root / store_name, save_stac=save_stac)
            manifest.mark_done(stem, store=store_name)
            for name, tile in layers.items():
                layer_metadata.setdefault(name, tile.metadata)
        except Exception as e:
            log.error("Failed to ingest anchor %s: %s", stem, e)
            manifest.mark_error(stem, str(e))

    if layer_metadata:
        manifest.set_metadata({"pipeline": type(pipeline).__name__, "layers": layer_metadata})


def stream_ingest(
    pipeline: GeoPipeline,
    anchors: Iterable[GeoAnchor],
    *,
    sel_bands: dict[str, list[str]] | None = None,
    dtype_override: dict[str, torch.dtype] | None = None,
) -> Iterator[dict[str, Any]]:
    """Ingest anchors through pipeline, yielding rendered samples without saving.

    For predicting/streaming straight from a live source — no disk round
    trip. Plain generator, not a Dataset: wrap it in a one-off
    IterableDataset at the call site if a DataLoader needs one (and shard
    by `torch.utils.data.get_worker_info()` there if using `num_workers>1`,
    since this generator itself has no worker-awareness).

    Args:
        pipeline: Pipeline whose ``ingest()`` builds each anchor's layers.
        anchors: Anchors to ingest — typically ``source.to_anchors(limit=...)``.
        sel_bands: Layer name to band names to keep. Default keeps all
            bands ``ingest`` returns for that layer.
        dtype_override: Layer name to torch dtype to cast that layer's
            tensor to. Default keeps the tensor's ingested dtype.

    Yields:
        Tensor dict per anchor, rendered via ``GeoStack.to_tensor``.
    """
    for anchor in anchors:
        stack = GeoStack(**pipeline.ingest(anchor))
        yield stack.to_tensor(sel_bands, dtype_override, pipeline.context)

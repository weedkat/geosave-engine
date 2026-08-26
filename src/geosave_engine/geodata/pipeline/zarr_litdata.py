"""Bulk conversion helpers between per-anchor .zarr stores and one packed LitDataStore.

zarr_to_litdata/litdata_to_zarr bodies are blocked on GeoStack's own
redesign around GeoRaster (the old tile-level GeoStack they were built
against no longer exists) — _scan_zarr_root doesn't depend on it, stays live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Unpack

import zarr

from geosave_engine.geodata.datastore.litdata import LitDataStore, LitDataStoreConfig

DEFAULT_ANCHORS_PER_BATCH = 1000


def _scan_zarr_root(paths: list[Path], root: Path, layer_name: str, required: set[str]) -> list[tuple[Path, list[str]]]:
    """Metadata-only filter over candidate .zarr stores — no GeoStack built.

    Args:
        paths: Candidate .zarr store paths.
        root: Their common root, for the error message.
        layer_name: Layer name for an ungrouped (bare-GeoTile) zarr.
        required: Layer names every included anchor must carry.

    Returns:
        `(path, layer_names)` per included anchor.

    Raises:
        ValueError: No anchor satisfies `required`, or included anchors
            don't all carry the same layer names.
    """
    included: list[tuple[Path, list[str]]] = []
    layer_sets: dict[frozenset[str], list[Path]] = {}
    for path in paths:
        available = sorted(zarr.open_group(path, mode="r").group_keys())
        # keep only anchors that carry every required layer name
        if available:
            if not required <= set(available):
                continue
        # an ungrouped zarr (bare GeoTile) has exactly one layer: layer_name
        elif not required <= {layer_name}:
            continue

        layer_set = frozenset(available) if available else frozenset({layer_name})
        layer_sets.setdefault(layer_set, []).append(path)
        included.append((path, available))

    if not included:
        raise ValueError(f"No .zarr stores under {root} satisfy required_layers={sorted(required)}")
    if len(layer_sets) > 1:
        raise ValueError(
            f"Anchors under {root} don't share one layer set — LitDataStore needs a single, "
            f"consistent field set per store; found {dict(layer_sets)}"
        )
    return included


def zarr_to_litdata(
    root: str | Path,
    output_dir: str | Path,
    layer_name: str = "image", # not important if zarr is geostack
    required_layers: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    mode: Literal["append", "overwrite"] | None = None,
    anchors_per_batch: int = DEFAULT_ANCHORS_PER_BATCH,
    **config: Unpack[LitDataStoreConfig],
) -> Path | str:
    """Rglob root/**/*.zarr, pack matches into one litdata LitDataStore.

    Scans matches first, then writes them into output_dir in batches.

    Args:
        root: Directory to rglob **/*.zarr under, any depth.
        output_dir: LitDataStore path to write into.
        layer_name: Layer name for an ungrouped (bare-GeoTile) zarr.
        required_layers: Layer names to require; a missing anchor is skipped.
        include: Glob pattern(s) an anchor's relative path must match one of.
        exclude: Glob pattern(s) an anchor's relative path must not match.
        mode: Forwarded to the first write() call; later batches always append.
        anchors_per_batch: Anchors per write() call, bounds peak memory. Default 1000.
        **config: LitDataStoreConfig — this store's locked optimize() config.

    Returns:
        output_dir, from LitDataStore.write()'s last call.

    Raises:
        ValueError: No matching .zarr stores, none satisfy required_layers,
            anchors_per_batch isn't positive, or anchors don't share one layer set.
        NotImplementedError: Always, for now — see module docstring.
    """
    raise NotImplementedError("zarr_to_litdata needs GeoStack, blocked pending its redesign around GeoRaster")


def litdata_to_zarr(
    store: LitDataStore,
    output_dir: str | Path,
    overwrite: bool = True,
) -> list[Path]:
    """Rebuild one .zarr per LitDataStore sample — reverses zarr_to_litdata.

    Each sample's layers rebuild into one GeoStack, written flat (no
    subdirs), named from its own "source_path" context key when present, else positionally.

    Args:
        store: LitDataStore to read from.
        output_dir: Root directory to write .zarr stores under.
        overwrite: False raises instead of replacing an existing anchor.

    Returns:
        Written paths, one per sample, in store order.

    Raises:
        ValueError: A sample has no "geo" field — not a GeoStack-shaped
            store — or two samples' source_path stems collide (flattening
            would silently overwrite one with the other).
        NotImplementedError: Always, for now — see module docstring.
    """
    raise NotImplementedError("litdata_to_zarr needs GeoStack, blocked pending its redesign around GeoRaster")

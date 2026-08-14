"""Bulk conversion helpers between per-anchor .zarr stores and one packed SampleStore."""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import zarr
from typing_extensions import Unpack

from geosave_engine.geodata.spatial import GeoAnchor, GeoStack, GeoTile
from geosave_engine.geodata.datastore.sample import SampleStore, SampleStoreConfig

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
            f"Anchors under {root} don't share one layer set — SampleStore needs a single, "
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
    **config: Unpack[SampleStoreConfig],
) -> Path | str:
    """Rglob root/**/*.zarr, pack matches into one litdata SampleStore.

    Scans matches first, then writes them into output_dir in batches.

    Args:
        root: Directory to rglob **/*.zarr under, any depth.
        output_dir: SampleStore path to write into.
        layer_name: Layer name for an ungrouped (bare-GeoTile) zarr.
        required_layers: Layer names to require; a missing anchor is skipped.
        include: Glob pattern(s) an anchor's relative path must match one of.
        exclude: Glob pattern(s) an anchor's relative path must not match.
        mode: Forwarded to the first write() call; later batches always append.
        anchors_per_batch: Anchors per write() call, bounds peak memory. Default 1000.
        **config: SampleStoreConfig — this store's locked optimize() config.

    Returns:
        output_dir, from SampleStore.write()'s last call.

    Raises:
        ValueError: No matching .zarr stores, none satisfy required_layers,
            anchors_per_batch isn't positive, or anchors don't share one layer set.
    """
    if anchors_per_batch <= 0:
        raise ValueError(f"anchors_per_batch must be positive, got {anchors_per_batch}")

    # Glob relevant zarr files
    root = Path(root)
    paths = sorted(root.rglob("*.zarr"))
    if include is not None:
        paths = [p for p in paths if any(fnmatch(p.relative_to(root).as_posix(), pat) for pat in include)]
    if exclude is not None:
        paths = [p for p in paths if not any(fnmatch(p.relative_to(root).as_posix(), pat) for pat in exclude)]
    if not paths:
        raise ValueError(f"No .zarr stores found under {root} (after include/exclude filtering)")

    # Validate anchors and collect layer info
    required = set(required_layers) if required_layers else set()
    included = _scan_zarr_root(paths, root, layer_name, required)

    # Build one GeoStack per anchor, reading its actual on-disk chunk size off its own lazy tile
    def _load(path: Path, available: list[str]) -> GeoStack:
        if available:
            stack = GeoStack.from_zarr(path, load_data=False)
        else:
            stack = GeoStack(**{layer_name: GeoTile.from_zarr(path, load_data=False)})
        chunks = next(iter(stack.tiles.values())).data.chunks
        chunk_px = chunks[-2][0] if chunks else None
        return stack.with_context(
            {**stack.context, "source_path": str(path.relative_to(root)), "chunk_px": chunk_px}
        )

    # Write in batches
    store = SampleStore(output_dir, **config)
    result: Path | str = output_dir
    for i, start in enumerate(range(0, len(included), anchors_per_batch)):
        batch = [_load(path, available) for path, available in included[start : start + anchors_per_batch]]
        result = store.write(batch, mode=mode if i == 0 else "append")
    return result


def litdata_to_zarr(
    store: SampleStore,
    output_dir: str | Path,
    overwrite: bool = True,
) -> list[Path]:
    """Rebuild one .zarr per SampleStore sample — reverses zarr_to_litdata.

    Each sample's layers rebuild into one GeoStack via GeoAnchor.from_dict
    per layer (this sample's shared geobox + that layer's own geotag).
    Needs GeoTag.bands recorded at write time to restore band names; a
    layer with none comes back as a single implicit band. Always writes
    flat (no subdirs — nesting scattered subdirs back on export re-fragments
    exactly what packing into one SampleStore avoided), named from a
    "source_path" context key's own stem when present (zarr_to_litdata's
    own — typically a GeoAnchor.stem, already collision-resistant by
    design), else positionally (000000.zarr, 000001.zarr, ...). A
    "chunk_px" context key, if present, re-chunks the rebuilt zarr to match
    what the source was actually written with — no caller input, same as
    source_path; absent, GeoStack.to_zarr's own default applies.

    Args:
        store: SampleStore to read from.
        output_dir: Root directory to write .zarr stores under.
        overwrite: False raises instead of replacing an existing anchor.

    Returns:
        Written paths, one per sample, in store order.

    Raises:
        ValueError: A sample has no "geotags" field — not a GeoStack-shaped
            store — or two samples' source_path stems collide (flattening
            would silently overwrite one with the other).
    """
    output_dir = Path(output_dir)
    written: list[Path] = []
    seen_names: set[str] = set()

    for i in range(len(store)):
        sample = store[i]
        if "geotags" not in sample:
            raise ValueError(f"sample {i} has no 'geotags' field — not a GeoStack-shaped store")

        tiles: dict[str, GeoTile] = {}
        for layer_name, geotag in sample["geotags"].items():
            anchor = GeoAnchor.from_dict({"geobox": sample["geobox"], "geotag": geotag})
            bands = list(anchor.geotag.bands) if anchor.geotag.bands else None
            tiles[layer_name] = anchor.to_geotile(sample[layer_name], names=bands)
        stack = GeoStack(**tiles)

        context = {k: v for k, v in sample.items() if k not in {"geobox", "geotags", *tiles}}
        if context:
            stack = stack.with_context(context)

        source_path = context.get("source_path")
        name = Path(source_path).stem if isinstance(source_path, str) else f"{i:06d}"
        if name in seen_names:
            raise ValueError(
                f"sample {i}'s source_path stem {name!r} collides with an earlier sample's — "
                "source anchors weren't uniquely named, can't flatten safely"
            )
        seen_names.add(name)

        zarr_kwargs = {"chunk_px": context["chunk_px"]} if "chunk_px" in context else {}
        out_path = output_dir / f"{name}.zarr"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stack.to_zarr(out_path, overwrite=overwrite, **zarr_kwargs)
        written.append(out_path)

    return written

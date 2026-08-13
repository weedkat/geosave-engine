"""Bulk conversion helpers between per-anchor .zarr stores and one packed SampleStore."""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import zarr
from typing_extensions import Unpack

from geosave_engine.geodata.spatial import GeoAnchor, GeoStack, GeoTile
from geosave_engine.geodata.datastore.sample import SampleStore, SampleStoreConfig


def _chunk_px_of(group: zarr.Group) -> int | None:
    """Read the on-disk y/x chunk size zarr actually stored for one group's arrays.

    Reads straight off zarr's own array metadata, not xarray/dask's derived
    view of it — zarr has no ragged chunks, one fixed size per dim, so
    whichever array happens to be first is representative of the whole
    group (GeoStack.to_zarr writes one chunk_px for every layer/variable).

    Args:
        group: Zarr group holding this layer's variable(s) — a GeoStack
            layer's own subgroup, or the store root for a bare GeoTile.

    Returns:
        Chunk side length, or None if the group has no arrays at all.
    """
    for _, array in group.arrays():
        return array.chunks[-2]  # canonical per-variable dim order: (y, x) or (time, y, x)
    return None


def zarr_to_litdata(
    root: str | Path,
    output_dir: str | Path,
    layer_name: str = "image",
    required_layers: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    mode: Literal["append", "overwrite"] | None = None,
    **config: Unpack[SampleStoreConfig],
) -> Path | str:
    """Rglob root/**/*.zarr, pack each store into one litdata sample via SampleStore.write().

    Each .zarr loads lazily (no pixel data read here) as one GeoStack — a
    grouped store (a GeoStack product) loads its named layers as-is; an
    ungrouped store (a bare GeoTile product) loads as one layer named
    layer_name. Every included anchor's path relative to root, and the
    on-disk chunk_px it was actually written with, are stashed into its
    packed sample's context as "source_path"/"chunk_px" — litdata_to_zarr
    reads both back automatically (path as metadata only, chunk_px to
    rebuild with matching chunking), and they ride along into
    StoreDataset/SampleStore.to_parquet's own manifest for free too.

    Args:
        root: Directory to rglob **/*.zarr under, any depth.
        output_dir: SampleStore path to write into.
        layer_name: Layer name for an ungrouped (bare-GeoTile) zarr. Ignored for a grouped one.
        required_layers: Layer names to require. An anchor missing one is
            silently excluded, same gate as StackDataset — not raised.
        include: Glob pattern(s) an anchor's root-relative path must match
            at least one of. None includes everything rglob found.
        exclude: Glob pattern(s) an anchor's root-relative path must not
            match any of. None excludes nothing.
        mode: Forwarded to SampleStore.write() — None raises if output_dir
            already holds a store, "append" grows it, "overwrite" replaces it.
        **config: SampleStoreConfig — this store's locked litdata.optimize() config.

    Returns:
        output_dir, as SampleStore.write() returns it.

    Raises:
        ValueError: No .zarr stores found under root (after include/exclude
            filtering), none satisfy required_layers, or the included
            anchors don't all carry the same layer names — SampleStore
            needs one consistent field set across every sample in one
            write() call (a per-anchor context key mismatch, e.g. some
            anchors carrying custom context data others don't, isn't
            checked here — surfaces as SampleStore's own field-mismatch
            error instead).
    """
    root = Path(root)
    paths = sorted(root.rglob("*.zarr"))
    if include is not None:
        paths = [p for p in paths if any(fnmatch(p.relative_to(root).as_posix(), pat) for pat in include)]
    if exclude is not None:
        paths = [p for p in paths if not any(fnmatch(p.relative_to(root).as_posix(), pat) for pat in exclude)]
    if not paths:
        raise ValueError(f"No .zarr stores found under {root} (after include/exclude filtering)")

    required = set(required_layers) if required_layers else set()
    stacks: list[GeoStack] = []
    layer_sets: dict[frozenset[str], list[Path]] = {}
    for path in paths:
        root_group = zarr.open_group(path, mode="r")
        layer_groups = dict(root_group.groups())
        available = sorted(layer_groups)
        if available:
            if not required <= set(available):
                continue  # missing a required layer — silently excluded, same as StackDataset
            stack = GeoStack.from_zarr(path, load_data=False)
            chunk_px = _chunk_px_of(layer_groups[available[0]])
        else:
            if not required <= {layer_name}:
                continue  # a bare GeoTile can only ever produce one layer, named layer_name
            stack = GeoStack(**{layer_name: GeoTile.from_zarr(path, load_data=False)})
            chunk_px = _chunk_px_of(root_group)  # bare GeoTile's arrays sit at the store root

        stack = stack.with_context(
            {**stack.context, "source_path": str(path.relative_to(root)), "chunk_px": chunk_px}
        )
        layer_sets.setdefault(frozenset(stack.tiles), []).append(path)
        stacks.append(stack)

    if not stacks:
        raise ValueError(f"No .zarr stores under {root} satisfy required_layers={required_layers}")
    if len(layer_sets) > 1:
        raise ValueError(
            f"Anchors under {root} don't share one layer set — SampleStore needs a single, "
            f"consistent field set per store; found {dict(layer_sets)}"
        )

    return SampleStore(output_dir, **config).write(stacks, mode=mode)


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

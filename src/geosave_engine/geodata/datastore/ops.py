"""Bulk import helpers — build a SampleStore from another on-disk format."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import zarr
from typing_extensions import Unpack

from geosave_engine.geodata.spatial import GeoStack, GeoTile
from geosave_engine.geodata.datastore.sample import SampleStore, SampleStoreConfig


def zarr_to_litdata(
    root: str | Path,
    output_dir: str | Path,
    layer_name: str = "image",
    required_layers: list[str] | None = None,
    mode: Literal["append", "overwrite"] | None = None,
    **config: Unpack[SampleStoreConfig],
) -> Path | str:
    """Glob root/*.zarr, pack each store into one litdata sample via SampleStore.write().

    Each .zarr loads lazily (no pixel data read here) as one GeoStack — a
    grouped store (a GeoStack product) loads its named layers as-is; an
    ungrouped store (a bare GeoTile product) loads as one layer named layer_name.

    Args:
        root: Directory to glob root/*.zarr under.
        output_dir: SampleStore path to write into.
        layer_name: Layer name for an ungrouped (bare-GeoTile) zarr. Ignored for a grouped one.
        required_layers: Layer names to require from a grouped zarr. None loads whatever's present.
        mode: Forwarded to SampleStore.write() — None raises if output_dir
            already holds a store, "append" grows it, "overwrite" replaces it.
        **config: SampleStoreConfig — this store's locked litdata.optimize() config.

    Returns:
        output_dir, as SampleStore.write() returns it.

    Raises:
        ValueError: No .zarr stores found under root.
    """
    paths = sorted(Path(root).glob("*.zarr"))
    if not paths:
        raise ValueError(f"No .zarr stores found under {root}")

    stacks: list[GeoStack] = []
    for path in paths:
        available = sorted(zarr.open_group(path, mode="r").group_keys())
        if available:
            stacks.append(GeoStack.from_zarr(path, required_layers=required_layers, load_data=False))
        else:
            stacks.append(GeoStack(**{layer_name: GeoTile.from_zarr(path, load_data=False)}))

    return SampleStore(output_dir, **config).write(stacks, mode=mode)

"""GeoStack: the collection counterpart to GeoTile.

Design summary (converged across discussion, not just this file):
    - GeoStack is a frozen dataclass, concrete, never subclassed per
      project — same as GeoTile. No declared schema/spec of any kind —
      every GeoTile already describes itself (geobox, dtype, bands,
      metadata), so a stack of them needs nothing more than the tiles.
    - Construction is one step: `GeoStack(**tiles)`, e.g.
      `GeoStack(**pipeline.ingest(anchor))`. No header-only state, no
      `.with_data()` — every real caller already has the full
      `dict[str, GeoTile]` in hand before it needs a stack at all.
    - Alignment is real geometry, not a name/stem comparison: `__init__`
      runs multi-tile stacks through `align()` (narrows every tile's
      geobox to their common intersection) and keeps the aligned result,
      not the originals — auto-correcting minor discrepancies rather than
      just rejecting them.
    - `save`/`load` write/read a *directory* — one plain `GeoTile.to_zarr`
      store per layer, named `{layer}.zarr`, inside one folder per anchor.
      Not named `to_zarr`/`from_zarr`: this class doesn't write zarr
      itself, it orchestrates `GeoTile`'s own zarr I/O per layer. Every
      layer is written and read back with GeoTile's serialization
      completely unchanged — no second on-disk format to reconcile with,
      no ambiguity about what a given `.zarr` store is.
    - No cross-directory stem-matching: every layer for one anchor is
      written into the same folder in the same `save()` call, so grouping
      is correct by construction, not reassembled after the fact from
      separate top-level layer directories.
    - `to_tensor()` always attaches one bare `GeoAnchor` per layer (via
      `GeoTile.to_anchor()`, pixel data/STAC stripped) under
      `sample["anchors"]` (`dict[LayerName, GeoAnchor]`) — not a single
      collapsed "the stack's anchor": `align()` guarantees identical
      `geobox` across tiles but not `datetime`/`metadata`/`polygon`, so
      picking one representative tile would silently lose the others'
      real values. A consumer that needs exactly one (e.g. rebuilding
      output georeferencing) picks whichever layer's anchor it actually
      means, instead of getting an arbitrary "first tile" stand-in.
    - The folder itself carries a `.geostack` suffix (`save`/`load` both
      require and enforce it) — same convention as `.zarr`/`.tif`, or
      Sentinel-2's own `.SAFE` product directories. Lets discovery over a
      (possibly deeply nested, grouped) tree of anchor folders use one
      unambiguous name-based glob — `root.rglob("*.geostack")` — instead of
      guessing "is this directory a real anchor folder" from the mere
      presence of `.zarr` stores inside it, which a stray unrelated store
      elsewhere in the tree could fool, and instead of a hidden marker file
      that doesn't show up browsing the tree normally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from typing_extensions import Unpack

from geosave_engine.geodata.tile.geotile import GeoTile, align

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy as np

    from geosave_engine.geodata.utils.geovis import PlotKwargs

LayerName = str
GEOSTACK_SUFFIX = ".geostack"


@dataclass(frozen=True)
class GeoStack:
    """Named group of GeoTiles for one anchor — the collection counterpart to GeoTile.

    GeoTile is one raster; GeoStack is the named set of them that make
    up one training or inference sample, all sharing one anchor. Carries
    no schema of its own — each tile already describes itself.

    Args:
        **tiles: Layer name to GeoTile, e.g. `GeoStack(**pipeline.ingest(anchor))`.
            Auto-aligned to their common spatial intersection via `align()`
            when there's more than one.

    Examples:
        >>> stack = GeoStack(sentinel_2_l1c=s2_tile, cloud_mask=mask_tile)
        >>> stack.save("data/train/13.000000_52.000000_20240115_10m.geostack")
        >>> sample = stack.to_tensor()
    """

    tiles: dict[LayerName, GeoTile]

    def __init__(self, **tiles: GeoTile) -> None:
        aligned = dict(zip(tiles.keys(), align(*tiles.values()))) if len(tiles) > 1 else tiles
        object.__setattr__(self, "tiles", aligned)

    def __repr__(self) -> str:
        layers = ", ".join(f"{name}={tile!r}" for name, tile in self.tiles.items())
        return f"{type(self).__name__}({layers})"

    def add(self, name: LayerName, tile: GeoTile) -> "GeoStack":
        """Return new GeoStack with one more layer, realigned.

        Pure — same shape as GeoTile.with_data: self is untouched, a new
        instance comes back. Runs every tile (existing + new) through
        align() again, same as construction. A name already present is
        overwritten by tile.

        Args:
            name: Layer name.
            tile: Tile to add under name.

        Returns:
            New GeoStack with name -> tile merged in.
        """
        return GeoStack(**{**self.tiles, name: tile})

    def plot(self, **kwargs: Unpack[PlotKwargs]) -> tuple[plt.Figure, np.ndarray]:
        """Plot every layer — thin wrapper, see `geosave_engine.geodata.utils.geovis.plot`.

        All layers share one anchor (same location, same date by
        construction), so this always facets one panel per layer — never a
        mosaic, since there's nothing else in the group to mosaic with.

        Args:
            **kwargs: Forwarded to `geovis.plot` (`cmap`, `class_map`,
                `color_map`, `rgb_bands`, `cols`, `title`).

        Returns:
            `(Figure, ndarray of Axes)`.
        """
        from geosave_engine.geodata.utils.geovis import plot

        return plot(list(self.tiles.values()), **kwargs)

    def save(self, path: str | Path, save_stac: bool | Sequence[LayerName] = False) -> Path:
        """Write every tile as its own zarr store inside a `.geostack` folder.

        Args:
            path: Output directory, must end in `.geostack`; created if
                missing. Each layer writes to `path/{layer}.zarr`.
            save_stac: `True`/`False` applies to every layer uniformly. A
                list of layer names saves STAC provenance for only those —
                useful when most layers are derived from (or have no) real
                STAC search results and would just duplicate/pad out one
                real source layer's sidecar with nothing new.

        Returns:
            The written directory path.

        Raises:
            ValueError: If path doesn't end in `.geostack`.
        """
        path = Path(path)
        if path.suffix != GEOSTACK_SUFFIX:
            raise ValueError(f"Expected a {GEOSTACK_SUFFIX} path, got: {path}")
        wanted = save_stac if isinstance(save_stac, bool) else set(save_stac)
        for name, tile in self.tiles.items():
            layer_save_stac = wanted if isinstance(wanted, bool) else name in wanted
            tile.to_zarr(path / f"{name}.zarr", save_stac=layer_save_stac)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        required_layers: list[LayerName] | None = None,
        load_data: bool = False,
    ) -> "GeoStack":
        """Read every {layer}.zarr inside a `.geostack` folder into one GeoStack.

        Args:
            path: Directory written by save(), must end in `.geostack`.
            required_layers: Layer names to require. None loads every
                `*.zarr` store present in path.
            load_data: Materialise all pixels into memory; default lazy.

        Returns:
            GeoStack with one GeoTile per loaded layer.

        Raises:
            ValueError: If path doesn't end in `.geostack`.
            KeyError: If a name in required_layers has no matching
                `{layer}.zarr` store in path.
        """
        path = Path(path)
        if path.suffix != GEOSTACK_SUFFIX:
            raise ValueError(f"Expected a {GEOSTACK_SUFFIX} path, got: {path}")
        available = [p.stem for p in sorted(path.glob("*.zarr"))]
        names = required_layers if required_layers is not None else available
        missing = set(names) - set(available)
        if missing:
            raise KeyError(f"Layer(s) {sorted(missing)} not found in {path} — available: {sorted(available)}")
        return cls(**{name: GeoTile.from_zarr(path / f"{name}.zarr", load_data=load_data) for name in names})

    def to_tensor(
        self,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        context_fn: Callable[[dict[LayerName, GeoTile]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Render carried tiles as one model sample.

        Args:
            sel_bands: Layer name to band names to keep. Default keeps all
                bands the tile carries.
            dtype_override: Layer name to torch dtype to cast that layer's
                tensor to. Default keeps the tensor's saved dtype.
            context_fn: Optional, takes `self.tiles` and returns extra keys
                to merge into the sample — e.g. a model-specific derivation
                of `temporal_coords`/`location_coords` from the tiles'
                anchors. Applied *after* `"anchors"` is set, so a returned
                `"anchors"` key would override it; every other key is purely
                additive. `None` skips this entirely — no keys beyond
                layers + `"anchors"` are added. Kept generic on purpose: this
                function has no idea what a caller's `context_fn` computes
                or why, it just merges the result.

        Returns:
            Tensor dict keyed by each layer's raw name, plus `"anchors"` —
            `dict[LayerName, GeoAnchor]`, one bare anchor (no pixel data)
            per layer, always present regardless of layer content — plus
            whatever `context_fn` returned, if given.
        """
        sel_bands = sel_bands or {}
        dtype_override = dtype_override or {}

        sample: dict[str, Any] = {}
        for layer_name, tile in self.tiles.items():
            tensor = tile.to_tensor(sel_bands.get(layer_name), squeeze=False)
            dtype = dtype_override.get(layer_name)
            if dtype is not None:
                tensor = tensor.to(dtype)
            sample[layer_name] = tensor

        sample["anchors"] = {name: tile.to_anchor() for name, tile in self.tiles.items()}
        if context_fn is not None:
            sample.update(context_fn(self.tiles))
        return sample

"""GeoStack: the collection counterpart to GeoTile. See GeoStack for details."""
from __future__ import annotations

import shutil
import torch
import zarr
import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
from typing_extensions import Unpack
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.tile.geotile import GeoTile, _write_stac
from geosave_engine.geodata.tile.ops import align_spatial
from geosave_engine.geodata.utils.geodata import da_to_ds
from geosave_engine.geodata.utils.io import to_zarr

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from geosave_engine.geodata.utils.geovis import PlotKwargs

LayerName = str
SaveMode = Literal["overwrite", "append", "error"]


class GeoStack:
    """Named group of GeoTiles for one anchor — the collection counterpart to GeoTile.

    GeoTile is one raster; GeoStack is the named set of them that make
    up one training or inference sample, all sharing one anchor. Carries
    no schema of its own — each tile already describes itself.

    Args:
        *args: Positional `GeoTile`s, auto-named `layer_0`, `layer_1`, ...
            A `GeoStack` is flattened instead — its own layers merge in
            under their real names, not auto-named. Mix freely with
            `**tiles` as long as names don't collide.
        **tiles: Layer name to GeoTile, e.g. `GeoStack(**pipeline.ingest(anchor))`.
            Auto-aligned to their common spatial intersection via
            `align_spatial()` when there's more than one.

    Raises:
        ValueError: Two positional args (including a flattened `GeoStack`'s
            own layers) produce the same name, or a positional arg's name
            collides with an explicit keyword name.

    Examples:
        >>> stack = GeoStack(sentinel_2_l1c=s2_tile, cloud_mask=mask_tile)
        >>> stack.save("data/train/13.000000_52.000000_20240115_10m.zarr")
        >>> sample = stack.to_tensor()
        >>> combined = GeoStack(stack, ndvi=ndvi_tile)  # flattens stack's own layers in
    """

    def __init__(self, *args: GeoTile | GeoStack, **tiles: GeoTile) -> None:
        positional: dict[LayerName, GeoTile] = {}
        layer_index = 0
        for arg in args:
            new_layers = arg.tiles if isinstance(arg, GeoStack) else {f"layer_{layer_index}": arg}
            if not isinstance(arg, GeoStack):
                layer_index += 1
            collision = set(positional) & set(new_layers)
            if collision:
                raise ValueError(f"layer name(s) {collision} collide across positional args — rename or don't mix")
            positional.update(new_layers)

        collision = set(positional) & set(tiles)
        if collision:
            raise ValueError(
                f"positional arg name(s) {collision} collide with explicit "
                "keyword name(s) — rename the keyword or don't mix"
            )
        named = {**positional, **tiles}
        aligned = dict(zip(named.keys(), align_spatial(*named.values()))) if len(named) > 1 else named
        self.tiles: dict[LayerName, GeoTile] = aligned
        self.geobox: "GeoBox" = next(iter(aligned.values())).geobox

    def __repr__(self) -> str:
        layers = ", ".join(f"{name}={tile!r}" for name, tile in self.tiles.items())
        return f"{type(self).__name__}({layers})"

    def plot(self, cols: int = 4, **kwargs: Unpack[PlotKwargs]) -> tuple[Figure, np.ndarray]:
        """Plot every layer — thin wrapper, see `geosave_engine.geodata.utils.geovis.plot`.

        Passes `self.tiles` through by name (not flattened to a bare list),
        so each panel titles as its own layer name — dict keys are unique,
        so two different layers can never mosaic into each other even if
        they happen to share bands/date/footprint.

        Args:
            **kwargs: Forwarded to `geovis.plot` (`cmap`, `class_map`,
                `color_map`, `rgb_bands`, `cols`, `title`, `show_metadata`).

        Returns:
            `(Figure, ndarray of Axes)`.
        """
        from geosave_engine.geodata.utils.geovis import plot

        return plot(self.tiles, cols=cols, **kwargs)

    def save(self, path: str | Path, mode: SaveMode = "overwrite") -> Path:
        """Write every layer into its own Zarr group inside one store.

        Each group is independently CF-compliant (one variable per band,
        own `time`/attrs) — same as `GeoTile.to_zarr`, just one call per
        layer instead of a whole-store write. STAC provenance writes
        alongside as `<path>/<layer_name>.stac.json`, one per layer that
        actually carries any (`tile.stac` non-empty) — a layer derived from
        another (e.g. a cloud mask built via `to_geotile`) carries none, so
        only the layer that actually came from a real STAC search gets a
        sidecar, with no separate flag needed.

        Args:
            path: Output Zarr store path, must end in `.zarr`.
            mode: `"overwrite"` wipes and rewrites the whole store from
                scratch, so a layer removed since a previous `save()`
                doesn't linger as a stale group. `"append"` adds this
                GeoStack's own layers as new groups into an existing store,
                untouched otherwise — raises if a layer name collides with
                a group already present, so two callers writing different
                layers into the same store (e.g. two different prediction
                writers) can't silently clobber each other. `"error"`
                raises immediately if `path` already exists, writing nothing.

        Returns:
            The written store path.

        Raises:
            ValueError: If path doesn't end in `.zarr`, or mode="append"
                and a layer name collides with a group already present.
            FileExistsError: If `path` already exists and mode is `"error"`.
        """
        path = Path(path)
        if path.suffix != ".zarr":
            raise ValueError(f"Expected a .zarr path, got: {path}")
        if path.exists():
            if mode == "error":
                raise FileExistsError(f"{path} already exists — pass mode='overwrite' or 'append'")
            if mode == "overwrite":
                shutil.rmtree(path)
            elif mode == "append":
                collision = set(zarr.open_group(path, mode="r").group_keys()) & set(self.tiles)
                if collision:
                    raise ValueError(
                        f"{path} already has group(s) {sorted(collision)} — append won't overwrite them"
                    )
        for layer_name, tile in self.tiles.items():
            tag = tile.geotag.model_dump_json(exclude_none=True)
            ds = da_to_ds(tile.data).assign_attrs(tag=tag)
            to_zarr(path, ds, group=layer_name)
            _write_stac(tile.stac, path, group=layer_name)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        required_layers: list[LayerName] | None = None,
        load_data: bool = False,
    ) -> "GeoStack":
        """Read a Zarr store written by save() into one GeoStack.

        Args:
            path: Store written by save(), must end in `.zarr`.
            required_layers: Layer names to require. None loads every layer
                (zarr group) present in the store.
            load_data: Materialise all pixels into memory; default lazy.

        Returns:
            GeoStack with one GeoTile per loaded layer.

        Raises:
            ValueError: If path doesn't end in `.zarr`, or the store has no
                zarr groups at all (not written by GeoStack.save).
            KeyError: If a name in required_layers isn't present in the store.
        """
        path = Path(path)
        if path.suffix != ".zarr":
            raise ValueError(f"Expected a .zarr path, got: {path}")
        available = sorted(zarr.open_group(path, mode="r").group_keys())
        if not available:
            raise ValueError(f"{path} has no zarr groups — not written by GeoStack.save")
        names = required_layers if required_layers is not None else available
        missing = set(names) - set(available)
        if missing:
            raise KeyError(f"Layer(s) {sorted(missing)} not found in {path} — available: {available}")

        tiles = {name: GeoTile.from_zarr(path, group=name, load_data=load_data) for name in names}
        return cls(**tiles)

    def to_tensor(
        self,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
        context_fn: Callable[[dict[LayerName, GeoTile]], dict[str, torch.Tensor]] | None = None,
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
                anchors. Applied *after* `"tiles"` is set, so a returned
                `"tiles"` key would override it; every other key is purely
                additive. `None` skips this entirely — no keys beyond
                layers + `"tiles"` are added. Kept generic on purpose: this
                function has no idea what a caller's `context_fn` computes
                or why, it just merges the result.

        Returns:
            Tensor dict keyed by each layer's raw name, plus `"tiles"` —
            `dict[LayerName, GeoTile]`, the real tile (not a bare anchor)
            per layer, always present regardless of layer content. Data
            stays lazy when the tile itself was loaded lazily (see
            `GeoTile.from_zarr`/`from_geotiff`), so carrying it costs
            nothing beyond spatial identity unless a caller reads `.data`.
            Plus whatever `context_fn` returned, if given.
        """
        sel_bands = sel_bands or {}
        dtype_override = dtype_override or {}

        sample: dict[str, Any] = {}
        for layer_name, tile in self.tiles.items():
            tensor = tile.to_tensor(sel_bands.get(layer_name))
            dtype = dtype_override.get(layer_name)
            if dtype is not None:
                tensor = tensor.to(dtype)
            sample[layer_name] = tensor

        sample["tiles"] = dict(self.tiles)
        if context_fn is not None:
            sample.update(context_fn(self.tiles))
        return sample

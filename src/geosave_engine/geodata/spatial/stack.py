"""GeoStack: the collection counterpart to GeoTile. See GeoStack for details."""
from __future__ import annotations

import json

import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray/Dataset
import torch
import zarr
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing_extensions import Unpack
from odc.geo.geobox import GeoBox
from pydantic import BaseModel

from geosave_engine.geodata.spatial.anchor import _geobox_to_dict
from geosave_engine.geodata.spatial.tile import GeoTile, _write_stac
from geosave_engine.geodata.spatial.ops import align_spatial
from geosave_engine.geodata.utils.geodata import da_to_ds
from geosave_engine.geodata.utils.zarr import to_zarr

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from geosave_engine.geodata.utils.geovis import PlotKwargs

LayerName = str


class LayerInfo(BaseModel):
    """One layer's slice of a to_xcube flat Dataset.

    Args:
        bands: Prefixed band names (`<layer>_<band>`), if the source tile
            had a band dim. None means the layer is its own bare variable,
            named after the layer itself — no synthetic band coordinate.
        had_time: True if the source tile had a time dim.
    """

    bands: list[str] | None
    had_time: bool


# to_tensor()/to_numpy() flatten context straight into the sample dict
# alongside layer tensors — these names must stay free for context to land
# on, not get silently overwritten by (or overwrite) real sample data.
_RESERVED_SAMPLE_KEYS = frozenset({"geobox", "geotags"})

# JSON has no tensor type -- a context value that's a torch.Tensor (the
# common case, e.g. temporal_coords) marks itself this way going in, so
# _context_from_json knows to rebuild a Tensor and not just hand back a
# plain nested list. Anything else round-trips as whatever plain JSON gave back.
_TENSOR_MARKER = "__tensor__"


def _check_context_collision(context: dict[str, Any], layer_names: dict[str, GeoTile]) -> None:
    """Raise if a context key would clobber (or get clobbered by) a layer or a reserved sample key.

    Args:
        context: Candidate `GeoStack.context`.
        layer_names: This stack's own `tiles` (only the keys matter).

    Raises:
        ValueError: A context key collides with a layer name or `"geobox"`/`"geotags"`.
    """
    collision = set(context) & (set(layer_names) | _RESERVED_SAMPLE_KEYS)
    if collision:
        raise ValueError(
            f"context key(s) {collision} collide with a layer name or a reserved "
            f"sample key {sorted(_RESERVED_SAMPLE_KEYS)} — to_tensor()/to_numpy() "
            "flatten context into the same dict, rename the context key(s)"
        )


def _context_to_json(context: dict[str, Any]) -> str:
    """GeoStack.context -> JSON string for a to_zarr root attr.

    Args:
        context: JSON-serializable values; a `torch.Tensor` value is flattened
            via `.tolist()` and tagged so `_context_from_json` rebuilds it.

    Returns:
        JSON-encoded `{key: value}`.
    """
    def encode(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {_TENSOR_MARKER: True, "dtype": str(value.dtype).removeprefix("torch."), "data": value.tolist()}
        return value

    return json.dumps({key: encode(value) for key, value in context.items()})


def _context_from_json(raw: str | None) -> dict[str, Any]:
    """Reverse of `_context_to_json` — rebuilds `torch.Tensor` values the marker tags.

    Args:
        raw: `from_zarr`'s root attr read, `None` if `to_zarr` never wrote one.

    Returns:
        `{key: value}` — a marked value comes back as a `torch.Tensor`
        (original dtype), anything else comes back exactly as JSON gave it.
    """
    if raw is None:
        return {}

    def decode(value: Any) -> Any:
        if isinstance(value, dict) and value.get(_TENSOR_MARKER):
            return torch.tensor(value["data"], dtype=getattr(torch, value["dtype"]))
        return value

    return {key: decode(value) for key, value in json.loads(raw).items()}


class StackTag(BaseModel):
    """Per-layer split info for a to_xcube Dataset, stored as a store attr.

    Args:
        layer_map: Layer name to its LayerInfo in the flat Dataset.
        layer_tags: Layer name to that layer's own GeoTag, JSON-encoded.
    """

    layer_map: dict[str, LayerInfo]
    layer_tags: dict[str, str]


@dataclass(frozen=True, kw_only=True)
class GeoStack:
    """Named group of GeoTiles for one anchor — the collection counterpart to GeoTile.

    GeoTile is one raster; GeoStack is the named set of them that make
    up one training or inference sample, all sharing one anchor. Carries
    no schema of its own — each tile already describes itself. Immutable —
    build a new GeoStack instead of adding layers onto an existing one.

    Args:
        *args: Positional `GeoTile`s, auto-named `layer_0`, `layer_1`, ...
            A nested `GeoStack` flattens in under its own layer names instead,
            its own `context` (if any) carried forward too.
        **tiles: Layer name to GeoTile. Auto-aligned to their common
            spatial intersection via `align_spatial()` when there's more than one.

    Raises:
        ValueError: Two positional args (including a flattened `GeoStack`'s
            own layers) produce the same name, a positional arg's name
            collides with a keyword name, or an inherited `context` key
            collides with a layer name or a reserved sample key
            (`"geobox"`/`"geotags"`).

    Examples:
        >>> stack = GeoStack(sentinel_2_l1c=s2_tile, cloud_mask=mask_tile)
        >>> stack.to_zarr("data/train/13.0000E_52.0000N_5kmx5km_20240115_10m.zarr")
        >>> sample = stack.to_tensor()
        >>> combined = GeoStack(stack, ndvi=ndvi_tile)  # flattens stack's own layers in
        >>> combined = combined.with_context({"temporal_coords": t})  # attach context
    """

    tiles: dict[LayerName, GeoTile]
    geobox: GeoBox = field(init=False)
    context: dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args: GeoTile | GeoStack, **tiles: GeoTile) -> None:
        positional: dict[LayerName, GeoTile] = {}
        layer_index = 0
        inherited_context: dict[str, Any] = {}
        for arg in args:
            new_layers = arg.tiles if isinstance(arg, GeoStack) else {f"layer_{layer_index}": arg}
            if isinstance(arg, GeoStack):
                inherited_context.update(arg.context)
            else:
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
        _check_context_collision(inherited_context, aligned)

        object.__setattr__(self, "tiles", aligned)
        object.__setattr__(self, "geobox", next(iter(aligned.values())).geobox)
        object.__setattr__(self, "context", inherited_context)

    def with_context(self, context: dict[str, Any]) -> "GeoStack":
        """New GeoStack, same tiles/geobox, context replaced (not merged).

        The only way to attach/replace context — deliberately not a
        constructor kwarg: mixing a specifically-typed keyword in with
        `**tiles: GeoTile`'s open-ended splat is exactly what forced a
        `# type: ignore` at every `GeoStack(**some_dict)` call site (a
        literal layer named `"context"` would misroute to it). Splitting
        it into its own method removes that class of call site entirely.

        Args:
            context: Extra per-sample values not tied to any one layer (e.g.
                `temporal_coords`/`location_coords` a `GeoPipeline.context()`
                derived). JSON-serializable, or a `torch.Tensor` (round-trips
                through `to_zarr`/`from_zarr` intact either way — see
                `_context_to_json`). Persisted by `to_zarr`, restored by
                `from_zarr`, merged into `to_tensor()`/`to_numpy()`'s output.

        Returns:
            A new GeoStack — `tiles`/`geobox` reused as-is, `context` replaced.

        Raises:
            ValueError: A context key collides with a layer name or a
                reserved sample key (`"geobox"`/`"geotags"`).
        """
        _check_context_collision(context, self.tiles)
        new = object.__new__(type(self))
        object.__setattr__(new, "tiles", self.tiles)
        object.__setattr__(new, "geobox", self.geobox)
        object.__setattr__(new, "context", context)
        return new

    def __repr__(self) -> str:
        layers = ", ".join(f"{name}={tile!r}" for name, tile in self.tiles.items())
        return f"{type(self).__name__}({layers})"

    def plot(self, cols: int = 4, **kwargs: Unpack[PlotKwargs]) -> tuple[Figure, np.ndarray]:
        """Plot every layer — thin wrapper, see `geosave_engine.geodata.utils.geovis.plot`.

        Args:
            **kwargs: Forwarded to `geovis.plot` (`cmap`, `class_map`,
                `color_map`, `rgb_bands`, `cols`, `title`, `show_metadata`).

        Returns:
            `(Figure, ndarray of Axes)`.
        """
        from geosave_engine.geodata.utils.geovis import plot

        return plot(self.tiles, cols=cols, **kwargs)

    def to_zarr(self, path: str | Path, overwrite: bool = True) -> Path:
        """Write every layer into its own Zarr group inside one store.

        STAC provenance, if any, writes alongside as `<layer_name>.stac.json`.
        `self.context`, if any, writes as one root-level attr — computed once
        here, `from_zarr` restores it, no per-read recomputation.

        Args:
            path: Output Zarr store path, must end in `.zarr`.
            overwrite: False raises instead of replacing an existing group.

        Returns:
            The written store path.

        Raises:
            ValueError: If path doesn't end in `.zarr`, or overwrite=False
                and a layer name collides with a group already present.
        """
        path = Path(path)
        if path.suffix != ".zarr":
            raise ValueError(f"Expected a .zarr path, got: {path}")
        if not overwrite and path.exists():
            collision = set(zarr.open_group(path, mode="r").group_keys()) & set(self.tiles)
            if collision:
                raise ValueError(
                    f"{path} already has group(s) {sorted(collision)} — pass overwrite=True to replace them"
                )
        for layer_name, tile in self.tiles.items():
            tag = tile.geotag.model_dump_json(exclude_none=True)
            ds = da_to_ds(tile.data).assign_attrs(tag=tag)
            to_zarr(path, ds, group=layer_name)
            _write_stac(tile.stac, path, group=layer_name)
        if self.context:
            zarr.open_group(path, mode="a").attrs["context"] = _context_to_json(self.context)
        return path

    @classmethod
    def from_zarr(
        cls,
        path: str | Path,
        required_layers: list[LayerName] | None = None,
        load_data: bool = False,
    ) -> "GeoStack":
        """Read a Zarr store written by to_zarr() into one GeoStack.

        A store with no groups loads its root as one `layer_0` layer.

        Args:
            path: Store written by to_zarr(), must end in `.zarr`.
            required_layers: Layer names to require. None loads every layer present.
            load_data: Materialise all pixels into memory; default lazy.

        Returns:
            GeoStack with one GeoTile per loaded layer, `context` restored
            from `to_zarr`'s root attr (empty if it never wrote one).

        Raises:
            ValueError: If path doesn't end in `.zarr`.
            KeyError: If a name in required_layers isn't present in the store.
        """
        path = Path(path)
        if path.suffix != ".zarr":
            raise ValueError(f"Expected a .zarr path, got: {path}")
        root = zarr.open_group(path, mode="r")
        available = sorted(root.group_keys())
        if not available:
            return cls(GeoTile.from_zarr(path, load_data=load_data))
        names = required_layers if required_layers is not None else available
        missing = set(names) - set(available)
        if missing:
            raise KeyError(f"Layer(s) {sorted(missing)} not found in {path} — available: {available}")

        tiles = {name: GeoTile.from_zarr(path, group=name, load_data=load_data) for name in names}
        stack = cls(**tiles)
        raw_context = root.attrs.get("context")
        context = _context_from_json(raw_context if isinstance(raw_context, str) else None)
        return stack.with_context(context) if context else stack

    def to_tensor(
        self,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, torch.dtype] | None = None,
    ) -> dict[str, Any]:
        """Render carried tiles as one model sample.

        Args:
            sel_bands: Layer name to band names to keep. Default keeps all
                bands the tile carries.
            dtype_override: Layer name to torch dtype to cast that layer's
                tensor to. Default keeps the tensor's saved dtype.

        Returns:
            Tensor dict keyed by layer name, plus `"geobox"` (this stack's
            one shared geobox, JSON-safe), `"geotags"` (`dict[LayerName,
            dict]`, one per-layer geotag, JSON-safe — layers can differ in
            datetime/metadata/polygon even though geobox is shared), and
            `self.context`'s own keys.
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

        sample["geobox"] = _geobox_to_dict(self.geobox)
        sample["geotags"] = {
            layer_name: tile.geotag.model_dump(mode="json", exclude_none=True)
            for layer_name, tile in self.tiles.items()
        }
        sample.update(self.context)
        return sample

    def to_numpy(
        self,
        sel_bands: dict[LayerName, list[str]] | None = None,
        dtype_override: dict[LayerName, np.dtype] | None = None,
    ) -> dict[str, Any]:
        """Render carried tiles as one NumPy sample.

        Args:
            sel_bands: Layer name to band names to keep. Default keeps all
                bands the tile carries.
            dtype_override: Layer name to NumPy dtype to cast that layer's
                array to. Default keeps the array's saved dtype.

        Returns:
            Array dict keyed by layer name, plus `"geobox"` (this stack's
            one shared geobox, JSON-safe), `"geotags"` (`dict[LayerName,
            dict]`, one per-layer geotag, JSON-safe), and `self.context`'s
            own keys as-is — not cast to array (a `torch.Tensor` context
            value stays a tensor; this method's array-casting is only for
            the per-layer image payload it exists for).
        """
        sel_bands = sel_bands or {}
        dtype_override = dtype_override or {}

        sample: dict[str, Any] = {}
        for layer_name, tile in self.tiles.items():
            array = tile.to_numpy(sel_bands.get(layer_name))
            dtype = dtype_override.get(layer_name)
            if dtype is not None:
                array = array.astype(dtype)
            sample[layer_name] = array

        sample["geobox"] = _geobox_to_dict(self.geobox)
        sample["geotags"] = {
            layer_name: tile.geotag.model_dump(mode="json", exclude_none=True)
            for layer_name, tile in self.tiles.items()
        }
        sample.update(self.context)
        return sample

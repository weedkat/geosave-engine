"""GeoStack: the collection counterpart to GeoTile. See GeoStack for details."""
from __future__ import annotations

import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray/Dataset
import torch
import xarray as xr
import zarr
import numpy as np
import warnings

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
from typing_extensions import Unpack
from odc.geo.geobox import GeoBox
from pydantic import BaseModel

from geosave_engine.geodata.tile.geoanchor import GeoTag
from geosave_engine.geodata.tile.geotile import GeoTile, _write_stac
from geosave_engine.geodata.tile.ops import _floor_time, align_spatial
from geosave_engine.geodata.utils.geodata import da_to_ds, default_nodata, validate_da
from geosave_engine.geodata.utils.zarr import from_zarr, to_zarr

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from geosave_engine.geodata.utils.geovis import PlotKwargs
    from geosave_engine.geodata.tile.ops import DatePrecision

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


class StackTag(BaseModel):
    """Per-layer split info for a to_xcube Dataset, stored as a store attr.

    Args:
        layer_map: Layer name to its LayerInfo in the flat Dataset.
        layer_tags: Layer name to that layer's own GeoTag, JSON-encoded.
    """

    layer_map: dict[str, LayerInfo]
    layer_tags: dict[str, str]


class GeoStack:
    """Named group of GeoTiles for one anchor — the collection counterpart to GeoTile.

    GeoTile is one raster; GeoStack is the named set of them that make
    up one training or inference sample, all sharing one anchor. Carries
    no schema of its own — each tile already describes itself.

    Args:
        *args: Positional `GeoTile`s, auto-named `layer_0`, `layer_1`, ...
            A nested `GeoStack` flattens in under its own layer names instead.
        **tiles: Layer name to GeoTile. Auto-aligned to their common
            spatial intersection via `align_spatial()` when there's more than one.

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

        Args:
            **kwargs: Forwarded to `geovis.plot` (`cmap`, `class_map`,
                `color_map`, `rgb_bands`, `cols`, `title`, `show_metadata`).

        Returns:
            `(Figure, ndarray of Axes)`.
        """
        from geosave_engine.geodata.utils.geovis import plot

        return plot(self.tiles, cols=cols, **kwargs)

    def save(self, path: str | Path, overwrite: bool = True) -> Path:
        """Write every layer into its own Zarr group inside one store.

        STAC provenance, if any, writes alongside as `<layer_name>.stac.json`.

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
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        required_layers: list[LayerName] | None = None,
        load_data: bool = False,
    ) -> "GeoStack":
        """Read a Zarr store written by save() into one GeoStack.

        A store with no groups loads its root as one `layer_0` layer.

        Args:
            path: Store written by save(), must end in `.zarr`.
            required_layers: Layer names to require. None loads every layer present.
            load_data: Materialise all pixels into memory; default lazy.

        Returns:
            GeoStack with one GeoTile per loaded layer.

        Raises:
            ValueError: If path doesn't end in `.zarr`.
            KeyError: If a name in required_layers isn't present in the store.
        """
        path = Path(path)
        if path.suffix != ".zarr":
            raise ValueError(f"Expected a .zarr path, got: {path}")
        available = sorted(zarr.open_group(path, mode="r").group_keys())
        if not available:
            return cls(GeoTile.from_zarr(path, load_data=load_data))
        names = required_layers if required_layers is not None else available
        missing = set(names) - set(available)
        if missing:
            raise KeyError(f"Layer(s) {sorted(missing)} not found in {path} — available: {available}")

        tiles = {name: GeoTile.from_zarr(path, group=name, load_data=load_data) for name in names}
        return cls(**tiles)

    def to_xcube(
        self,
        path: str | Path,
        date_precision: "DatePrecision" = 'D',
        chunk_px: int | None = 512,
        zarr_format: Literal[2, 3] | None = 2,
    ) -> Path:
        """Save every layer as one flat xcube-convention Zarr store, e.g.::

            <path>/
              s2_B02       (y, x) or (time, y, x)  # layer "s2", band "B02"
              s2_B03       (y, x) or (time, y, x)
              cloud_mask   (y, x) or (time, y, x)   # bandless layer, bare name
              spatial_ref  # CRS grid-mapping coord, shared by every variable
              var_order    # write order, see to_zarr
            attrs: stack_tag  # layer_map/layer_tags — how from_xcube splits variables back into layers

        Bands named `<layer>_<band>` (or `<layer>` if bandless), reconciled onto
        one shared time axis. Gaps fill with each layer's own nodata, or a
        dtype-sentinel default (warns) if none declared.

        Args:
            path: Output Zarr store path, must end in `.zarr`.
            date_precision: Floor each layer's time coord to this before reconciling.
            chunk_px: Forwarded to `to_zarr`.
            zarr_format: Forwarded to `to_zarr`. Default 2 — xcube-core (as of
                this writing) can't read Zarr format 3 stores.

        Returns:
            The written store path.

        Examples:
            >>> stack = GeoStack(sentinel_2_l1c=s2_tile, cloud_mask=mask_tile)
            >>> stack.to_xcube("data/train/13.000000_52.000000_20240115_10m.zarr")
            >>> back = GeoStack.from_xcube("data/train/13.000000_52.000000_20240115_10m.zarr")
        """
        floored = {name: _floor_time(tile.data, date_precision) for name, tile in self.tiles.items()}

        # mismatched time axes union via outer join; timeless layers broadcast
        time_varying = [n for n in floored if "time" in floored[n].dims]
        if time_varying:
            axes = {tuple(floored[n].time.values.tolist()) for n in time_varying}
            if len(axes) > 1:
                aligned = xr.align(*(floored[n] for n in time_varying), join="outer", exclude=("band", "y", "x"))
                for n, da in zip(time_varying, aligned):
                    if bool(da.isnull().any()):
                        nodata = self.tiles[n].nodata
                        if nodata is None:
                            nodata = default_nodata(floored[n].dtype)
                            warnings.warn(f"to_xcube(): layer {n!r} has no declared nodata, gap-filling with {nodata!r}")
                        da = da.fillna(nodata).astype(floored[n].dtype)
                    floored[n] = da
                    
            target_time = floored[time_varying[0]].time.values
            for n in floored:
                if "time" not in floored[n].dims:
                    floored[n] = floored[n].expand_dims(time=target_time)

        # banded layers concat to one shared dtype; bandless keep their own variable/dtype
        band_pieces: list[xr.DataArray] = []
        datasets: list[xr.Dataset] = []
        layer_map: dict[str, LayerInfo] = {}
        rgb_bands = None
        for name, tile in self.tiles.items():
            da = floored[name]
            if "band" in tile.data.dims:
                band_names = [f"{name}_{b}" for b in tile.bands]
                band_pieces.append(da.assign_coords(band=band_names))
                layer_map[name] = LayerInfo(bands=band_names, had_time=tile.has_time)
            else:
                datasets.append(da.to_dataset(name=name))
                layer_map[name] = LayerInfo(bands=None, had_time=tile.has_time)
            if rgb_bands is None and tile.plot_meta.rgb_bands is not None:
                rgb_bands = tuple(f"{name}_{b}" for b in tile.plot_meta.rgb_bands)
        if band_pieces:
            combined = validate_da(xr.concat(band_pieces, dim="band"))
            datasets.insert(0, combined.to_dataset(dim="band"))
        ds = xr.merge(datasets, compat="override") if len(datasets) > 1 else datasets[0]

        stack_tag = StackTag(
            layer_map=layer_map,
            layer_tags={name: tile.geotag.model_dump_json(exclude_none=True) for name, tile in self.tiles.items()},
        )
        ds = ds.assign_attrs(stack_tag=stack_tag.model_dump(mode="json"))
        if rgb_bands is not None:
            ds = ds.assign_attrs(rgb_bands=rgb_bands)

        return to_zarr(path, ds, chunk_px=chunk_px, zarr_format=zarr_format)

    @classmethod
    def from_xcube(cls, path: str | Path, load_data: bool = False) -> "GeoStack":
        """Load a to_xcube Zarr store back into a GeoStack.

        Args:
            path: Store written by `to_xcube`.
            load_data: Materialise all pixels into memory; default lazy.

        Returns:
            GeoStack with the same layers `to_xcube` was built from.

        Raises:
            ValueError: Store has no `stack_tag` attr — not written by `to_xcube`.
        """
        ds = from_zarr(path)
        raw_stack_tag = ds.attrs.get("stack_tag")
        if raw_stack_tag is None:
            raise ValueError(f"from_xcube(): {path} has no stack_tag attr — not written by to_xcube")
        stack_tag = StackTag.model_validate(raw_stack_tag)

        geobox = ds.odc.geobox
        if load_data:
            ds = ds.load()

        layers: dict[str, GeoTile] = {}
        for name, info in stack_tag.layer_map.items():
            if info.bands is None:
                sel = ds[name]
            else:
                nodata = ds[info.bands[0]].rio.nodata
                sel = ds[info.bands].to_array(dim="band").assign_coords(band=[b[len(name) + 1:] for b in info.bands])
                sel = sel.transpose("time", "band", "y", "x") if "time" in sel.dims else sel.transpose("band", "y", "x")
                if nodata is not None:
                    sel = sel.rio.write_nodata(nodata)
            if not info.had_time and "time" in sel.dims:
                sel = sel.isel(time=0, drop=True)
            layer_tag = GeoTag.model_validate_json(stack_tag.layer_tags[name])
            layers[name] = GeoTile(geobox=geobox, data=validate_da(sel), geotag=layer_tag)
        return cls(**layers)

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
            context_fn: Takes `self.tiles`, returns extra keys merged into
                the sample after `"tiles"` is set. None adds nothing extra.

        Returns:
            Tensor dict keyed by layer name, plus `"tiles"` (`dict[LayerName,
            GeoTile]`) and whatever `context_fn` returned.
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

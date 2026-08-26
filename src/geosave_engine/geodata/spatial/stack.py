"""GeoStack: named raster layers over one surface. See GeoStack for details."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Self, Unpack, cast

from ._stack import LayerName, _SpatialStack
from geosave_engine.geodata.utils.io import to_zarr

from .raster import GeoRaster
from .tile import GeoTile

# GeoStack's own stack-level header key — unrelated to any layer's own extension namespaces.
_STACK_HEADER_KEY = "geosave"

# Where a store root records which layer the stack was anchored on, and the order they were written in.
REFERENCE_KEY = "reference_layer"
LAYERS_KEY = "layers"

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.extensions import TilerMode
    from geosave_engine.geodata.utils.io import ZarrOptions

    from .tile_stack import GeoTileStack
    from .context import ContextFn

# Reference-layer steps per window and how far apart windows start; a bare length strides by itself.
TimeWindow = int | tuple[int, int]


def _span_key(stack: GeoStack) -> str:
    """Text naming the span a stack's timed layers cover together.

    Args:
        stack: Stack to read, timed layers or not.

    Returns:
        `"<start>/<end>"` over every timed layer, empty when none has time.
    """
    spans = [layer.timespan for layer in stack.values() if layer.timespan is not None]
    if not spans:
        return ""
    return f"{min(start for start, _ in spans).isoformat()}/{max(end for _, end in spans).isoformat()}"


class GeoStack(_SpatialStack[GeoRaster]):
    """Named raster layers over one surface — any size, pixels not fully in memory.

    Groups sources a model reads together, e.g. Sentinel-2 with a DEM. Layers
    must already share one geobox; align them with `GeoRaster.reproject_like`.

    Args:
        base: Layers this stack starts from — a mapping, or another GeoStack
            to extend with derived layers. None starts empty.
        reference_layer: Layer whose anchor is this stack's own identity —
            grid, time span, features, attrs. None keeps `base`'s when it is
            a GeoStack, else uses the first layer.
        **layers: `name=raster`, on the reference layer's exact geobox. A
            name already in `base` replaces that layer.

    Raises:
        ValueError: No layer across `base` and `layers`, `reference_layer`
            names none, or a layer's geobox differs from the reference's.
        TypeError: A layer isn't a GeoRaster.

    Examples:
        >>> stack = GeoStack(image=s2, dem=dem.reproject_like(s2))
        >>> stack = GeoStack({"image": s2, "dem": dem})
        >>> stack = GeoStack(ingested, dynamicworld=label)
    """

    LAYER_TYPE = GeoRaster

    def __init__(
        self,
        base: Mapping[LayerName, GeoRaster] | None = None,
        /,
        *,
        reference_layer: LayerName | None = None,
        **layers: GeoRaster,
    ) -> None:
        merged: dict[LayerName, GeoRaster] = dict(base) if base is not None else {}
        merged.update(layers)
        if reference_layer is None and isinstance(base, GeoStack):
            reference_layer = base.reference_layer
        super().__init__(merged, reference_layer=reference_layer)

    def _rebuild(self, layers: Mapping[LayerName, GeoRaster], *, reference_layer: LayerName | None = None) -> Self:
        """Build a GeoStack over replacement layers.

        Args:
            layers: Replacement layers, including the reference layer.
            reference_layer: Layer to anchor on. None keeps this stack's own.

        Returns:
            New GeoStack.
        """
        return type(self)(layers, reference_layer=reference_layer or self.reference_layer)

    @classmethod
    def open(cls, path: str | Path, *, reference_layer: LayerName | None = None) -> Self:
        """Lazily open a stack store — one zarr group per layer, as `to_zarr` wrote it.

        Args:
            path: `.zarr` store to open.
            reference_layer: Layer whose anchor is the stack's. None uses the
                first group in name order.

        Returns:
            GeoStack with every group as a layer, in the order `to_zarr`
            recorded, pixels still on disk, each layer's
            `<stem>.<layer>.vector.parquet` sidecar read back as its own
            vector. A store written before layer order was recorded reopens
            in name order.

        Raises:
            ValueError: `path` isn't a `.zarr` store, holds no group, its
                recorded layer order doesn't match its groups, or its groups
                disagree on their geobox.

        Examples:
            >>> stack = GeoStack.open("data/train/sample.zarr")
        """
        import zarr

        path = Path(path)
        if path.suffix != ".zarr":
            raise ValueError(f"Expected a .zarr store, got: {path}")
        root = zarr.open_group(str(path), mode="r")
        groups = set(root.group_keys())
        if not groups:
            raise ValueError(f"no layer groups in {path} — a GeoStack store holds one group per layer")

        stored = dict(root.attrs).get(_STACK_HEADER_KEY)
        header = dict(stored) if isinstance(stored, Mapping) else {}

        recorded = header.get(LAYERS_KEY)
        if isinstance(recorded, (list, tuple)):
            names = [str(name) for name in recorded]
            if set(names) != groups:
                raise ValueError(
                    f"{path} records layers {names} but holds groups {sorted(groups)} — "
                    "the store was written or edited by something that disagreed on its layers"
                )
        else:
            names = sorted(groups)

        if reference_layer is None:
            anchored = header.get(REFERENCE_KEY)
            reference_layer = anchored if isinstance(anchored, str) else None
        return cls(
            {name: GeoRaster.open(path, group=name) for name in names},
            reference_layer=reference_layer,
        )

    # --- Windowing ---

    def crop(self, geobox: GeoBox) -> Self:
        """Cut every layer to a window already on the shared pixel grid.

        Args:
            geobox: Window to keep. Must sit fully inside this stack's own
                geobox and land on its pixel grid.

        Returns:
            New GeoStack on `geobox`, same layers, each layer's features
            filtered to the window and its `tiling` cleared.

        Raises:
            ValueError: `geobox` isn't on this stack's pixel grid, or isn't
                fully inside its extent.
        """
        return self._rebuild({name: raster.crop(geobox) for name, raster in self.items()})

    def to_sample(self) -> GeoTileStack:
        """Read this surface as one window — same layers, headers and features.

        The inverse of `GeoTileStack.to_stack`, and the point at which a caller
        asserts these pixels fit in memory. Nothing here checks that.

        Returns:
            GeoTileStack over the same layers, each as a GeoTile.
        """
        from .tile_stack import GeoTileStack

        return GeoTileStack(
            {name: raster.to_tile() for name, raster in self.items()},
            reference_layer=self.reference_layer,
        )

    def tiles(
        self,
        tile_size_px: int | None = None,
        stride_px: int | None = None,
        overlap: int | float | tuple[int, int] | None = None,
        mode: TilerMode = "reflect",
        vector: bool = True,
        *,
        time: TimeWindow | None = None,
        name: str | None = None,
        context_fn: ContextFn | None = None,
    ) -> Iterator[GeoTileStack]:
        """Cut every layer into matching square windows, one GeoTileStack per position and time window.

        Every layer of one window carries the reference layer's `tiling`
        stamp, so predictions off any of them merge through `from_tiles`.
        Each time window is its own tiling group.

        Args:
            tile_size_px: Window side length in pixels (square), before edge
                handling. None uses the shorter of the two axes.
            stride_px: Distance between consecutive window origins. None = tile_size_px.
            overlap: Forwarded to tiler.Tiler's own overlap kwarg. Wins over
                `stride_px` when both are given.
            mode: How a trailing window's overhang is filled — "reflect"
                mirrors, "edge" repeats, "constant" uses each layer's nodata.
            vector: True gives each sample the reference layer's features
                filtered to its window, kept whole. False yields none.
            time: `(length, stride)` in reference-layer steps, or a bare
                length. Windows are cut on the reference layer alone; every
                other timed layer keeps the steps whose own buckets overlap
                that window, and timeless layers ride along whole. None
                windows nothing.
            name: Extra text folded into each cut's derived `group_id`,
                separating two otherwise identical cuts.
            context_fn: Called once per window with the reference layer's own
                anchor, whose header carries that window's bands and steps; its
                result becomes that sample's `model_context`. None leaves it
                unset, and an encoder derives its own at forward time.

        Yields:
            One GeoTileStack per time window and position, window-major: every
            position of one window before the next window starts, so a
            stitcher holds one output surface at a time. Positions run
            row-major. Lazy in, lazy out — no pixel is read here.

        Raises:
            ValueError: `tile_size_px` isn't positive, `stride_px` isn't
                positive or is wider than the tile, `mode` is invalid,
                mode="constant" and a layer declares no nodata, or `time`
                was given and the reference layer is timeless.

        Examples:
            >>> for sample in stack.tiles(512, time=(4, 1), context_fn=Clay.model_context):
            ...     batch = sample.to_tensor()
        """
        from .tile_stack import GeoTileStack

        names = list(self)
        reference_position = names.index(self.reference_layer)
        for window in self._time_cuts(time):
            # windows of one surface differ only by span, so the group id needs the span to stay distinct
            group_name = name if time is None else f"{name or ''}@{_span_key(window)}"
            cuts = [
                raster.tiles(
                    tile_size_px=tile_size_px,
                    stride_px=stride_px,
                    overlap=overlap,
                    mode=mode,
                    vector=vector if layer == self.reference_layer else False,
                    name=group_name,
                )
                for layer, raster in window.items()
            ]
            for tiles in zip(*cuts, strict=True):
                # the reference's stamp makes every layer one merge group; it already carries it
                stamp = tiles[reference_position].tiling
                layers = {
                    layer: tile
                    if tile.tiling == stamp
                    else GeoTile(data=tile.data, anchor=tile.anchor.rebase(tiling=stamp))
                    for layer, tile in zip(names, tiles, strict=True)
                }
                anchored = layers[self.reference_layer]
                yield GeoTileStack(
                    layers,
                    reference_layer=self.reference_layer,
                    model_context=None if context_fn is None else context_fn(anchored.anchor),
                )

    def _time_cuts(self, time: TimeWindow | None) -> list[Self]:
        """This stack cut into one stack per time window.

        The reference layer's steps define each window; every other timed layer
        keeps the steps whose buckets overlap it. All of them bucket at least as
        coarsely, so a window a layer has nothing in means a real gap in it.

        Args:
            time: `(length, stride)` in reference-layer steps, or a bare
                length. None or empty returns this stack alone.

        Returns:
            One stack per window, in time order.

        Raises:
            ValueError: `time` was given and the reference layer has no time
                dim to window, or a layer has no step over some window.
        """
        if not time:
            return [self]

        reference = self[self.reference_layer]
        if not reference.has_time:
            raise ValueError(
                f"time= windows the reference layer {self.reference_layer!r}, which has no time dim"
            )
        length, stride = (time, time) if isinstance(time, int) else time

        cuts: list[Self] = []
        for window in reference.time_windows(length, stride):
            span = window.timespan
            assert span is not None
            layers: dict[LayerName, GeoRaster] = {}
            for name, layer in self.items():
                if name == self.reference_layer:
                    layers[name] = window
                elif not layer.has_time:
                    layers[name] = layer
                else:
                    try:
                        layers[name] = layer.select(time=span)
                    except ValueError as gap:
                        raise ValueError(
                            f"layer {name!r} has no step over {span[0]}–{span[1]}, a window of "
                            f"reference layer {self.reference_layer!r} — that window would train on "
                            f"a layer that isn't there; fill the gap or narrow the stack's own span"
                        ) from gap
            cuts.append(self._rebuild(layers))
        return cuts

    # --- Persistence ---

    def to_zarr(
        self,
        path: str | Path,
        chunk_px: int | None = 512,
        progress: bool = True,
        **options: Unpack[ZarrOptions],
    ) -> Path:
        """Write one CF-compliant Zarr store, one group per layer.

        Every layer's pixels compute together, so layers sharing a source read
        it once. Each `vector` goes beside the store as `<stem>.<layer>.vector.parquet`,
        and the root records the reference layer and order, so `open` restores both.

        Args:
            path: Output `.zarr` store path.
            chunk_px: Spatial (y/x) chunk side length. `time` is never split.
            progress: Show a dask progress bar while pixels compute.
            **options: Passed to `xarray.Dataset.to_zarr` for every layer —
                see `ZarrOptions`. `group` names the layer and `compute` runs
                the single pass, so neither may be given.

        Returns:
            The written store path.

        Raises:
            ValueError: `group` or `compute` was given, or an option is one
                the writer sets itself.
        """
        import zarr
        from dask.base import compute as dask_compute

        from geosave_engine.geodata.utils.array import progress_bar

        clashing = sorted({"group", "compute"} & set(options))
        if clashing:
            raise ValueError(
                f"to_zarr() sets {clashing} itself — 'group' names each layer's own group, and "
                "'compute' is driven by the single pass that reads shared sources once"
            )

        # the adapter, not GeoRaster.to_zarr: only this write defers, so layers share one pass
        written = Path(path)
        pending = [
            to_zarr(
                written,
                raster._cf_encoded("json"),
                vector=None if raster.vector is None else raster.vector.gdf,
                chunk_px=chunk_px,
                progress=False,
                compute=False,
                # widened: an overload cannot be matched through a TypedDict spread
                **cast("Any", {**options, "group": name}),
            )
            for name, raster in self.items()
        ]
        with progress_bar(progress):
            dask_compute(*pending)

        # reference layer and layer order are stack-level facts, so they ride on the store root, not in any group
        zarr.open_group(str(written), mode="a").attrs[_STACK_HEADER_KEY] = {
            REFERENCE_KEY: self.reference_layer,
            LAYERS_KEY: list(self),
        }
        return written

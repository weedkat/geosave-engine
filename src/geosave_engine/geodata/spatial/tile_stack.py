"""GeoTileStack: named tile layers over one window — one model input. See GeoTileStack for details."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime as dt
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, Unpack, cast, overload

import numpy as np
from dask.base import compute as dask_compute
from numpy.typing import DTypeLike

from ._stack import LayerName, _SpatialStack
from .anchor import GeoAnchor
from .context import ModelContext, numpy_context, tensor_context
from .tile import GeoTile

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from geosave_engine.geodata.viz import Kind, RenderStyle, ViewOptions

    from .stack import GeoStack

__all__ = ["GeoTileStack", "NumpySample", "TensorSample"]


class NumpySample(TypedDict):
    """Materialized NumPy sample envelope."""

    layers: dict[LayerName, np.ndarray]
    anchor: GeoAnchor
    model_context: dict[str, Any]


class TensorSample(TypedDict):
    """Materialized tensor sample envelope."""

    layers: dict[LayerName, torch.Tensor]
    anchor: GeoAnchor
    model_context: dict[str, Any]


class GeoTileStack(_SpatialStack[GeoTile]):
    """Named tile layers over one window — small, whole-load-safe, one model input.

    Comes from `GeoStack.tiles()`/`.to_sample()`, or a pipeline's own
    ingest. Renders to the dict a model reads with `to_numpy`/`to_tensor`.
    Never reconciles grids — align the source rasters before making tiles.

    Args:
        base: Layers this sample starts from — a mapping, or another
            GeoTileStack to extend. None starts empty.
        reference_layer: Layer whose anchor is this sample's own identity.
            None keeps `base`'s when it is a GeoTileStack, else uses the first
            layer.
        model_context: Precomputed model inputs for this window, normally
            `GeoStack.tiles(context_fn=...)`'s own output. Stamped onto the
            reference layer's tile, which owns it. None leaves the tile's own
            in place.
        **layers: `name=tile`, on the reference layer's exact geobox. A name
            already in `base` replaces that layer.

    Raises:
        ValueError: No layer given, a layer's geobox differs from the
            reference's, a `model_context` value isn't array-like, or one was
            given while the reference layer's tile already carries its own.
        TypeError: A layer isn't a GeoTile.

    Examples:
        >>> sample = GeoTileStack(image=tile, label=mask)
        >>> sample = GeoTileStack({"image": tile, "label": mask})
        >>> batch = sample.to_tensor()
    """

    LAYER_TYPE = GeoTile

    def __init__(
        self,
        base: Mapping[LayerName, GeoTile] | None = None,
        /,
        *,
        reference_layer: LayerName | None = None,
        model_context: Mapping[str, object] | None = None,
        **layers: GeoTile,
    ) -> None:
        merged: dict[LayerName, GeoTile] = dict(base) if base is not None else {}
        merged.update(layers)
        if reference_layer is None and isinstance(base, GeoTileStack):
            reference_layer = base.reference_layer
        super().__init__(merged, reference_layer=reference_layer)
        if model_context is not None:
            anchored = self._identity
            if anchored.model_context is not None:
                raise ValueError(
                    f"model_context was given and reference layer {self.reference_layer!r} carries "
                    "one of its own — pass it here or on the tile, not both"
                )
            # context belongs to one window, so it rides on that window's tile, not beside it
            self._layers[self.reference_layer] = replace(anchored, model_context=dict(model_context))

    def _rebuild(self, layers: Mapping[LayerName, GeoTile], *, reference_layer: LayerName | None = None) -> Self:
        """Build a GeoTileStack over replacement layers.

        Args:
            layers: Replacement layers, including the reference layer.
            reference_layer: Layer to anchor on. None keeps this sample's own.

        Returns:
            New GeoTileStack. Its context is whatever the resolved reference
            layer's tile carries.
        """
        return type(self)(layers, reference_layer=reference_layer or self.reference_layer)

    @property
    def model_context(self) -> ModelContext | None:
        """Precomputed model inputs for this window — the reference layer's own.

        Read-only: context derives from an anchor, and this sample's anchor
        cannot change, so there is nothing to re-derive it against.

        Returns:
            Values as the encoder produced them, or None when no
            `context_fn` ran.
        """
        return self._identity.model_context

    def plot(
        self,
        *,
        kind: Kind | None = None,
        style: RenderStyle | None = None,
        band: str | None = None,
        time: dt | None = None,
        vector: bool = True,
        **options: Unpack[ViewOptions],
    ) -> Figure:
        """Draw every layer as one static figure, a panel each.

        Layers declaring the same `render` hints share one color range, so a
        label and a prediction compare honestly. Layers with no hints keep
        their own, since one range across unrelated units would flatten them.

        Args:
            kind: Force one renderer for every layer instead of resolving
                each from its own `render` hints.
            style: Color policy applied to every layer. None takes the default.
            band: Draw this band of every layer that carries it.
            time: Draw this timestamp of every layer that carries it.
            vector: True outlines this sample's own features over each panel.
            **options: Per-view hvplot options — see `ViewOptions`. An
                explicit `clim` here overrides the shared range.

        Returns:
            Matplotlib Figure holding one panel per layer, in this sample's
            own layer order.

        Raises:
            ImportError: The `viz` extra isn't installed.
            KeyError: `band` or `time` names something a layer doesn't carry.

        Examples:
            >>> sample.plot(width=1200)
        """
        import holoviews as hv

        from geosave_engine.geodata.viz import DEFAULT_STYLE, to_static_element

        resolved = style if style is not None else DEFAULT_STYLE
        features = self.vector
        shared = self._shared_limits(resolved)
        panels = []
        for name, tile in self.items():
            panel_options = cast("ViewOptions", dict(options))
            if "clim" not in panel_options and name in shared:
                panel_options["clim"] = shared[name]["clim"]
            panels.append(
                to_static_element(
                    tile.data,
                    render=tile.render,
                    legend=tile.legend,
                    kind=kind,
                    style=resolved,
                    band=band,
                    time=time,
                    vector=features.gdf if vector and features is not None else None,
                    **panel_options,
                ).relabel(name)
            )
        return hv.render(hv.Layout(panels), backend="matplotlib")

    def _shared_limits(self, style: RenderStyle) -> dict[LayerName, dict[str, tuple[float, float]]]:
        """One color range per group of layers declaring the same legend.

        Args:
            style: Policy naming the percentiles the range spans.

        Returns:
            `{layer: {"clim": (low, high)}}` for layers sharing a legend with
            another layer. Layers with no legend, or one nothing else
            shares, are absent and keep their own range.
        """
        from geosave_engine.geodata.viz import shared_limits

        # a legend holds dicts, so layers group by its serialized form rather than by identity
        groups: dict[str, list[LayerName]] = {}
        for name, tile in self.items():
            legend = tile.legend
            if legend is not None:
                groups.setdefault(legend.model_dump_json(), []).append(name)

        limits: dict[LayerName, dict[str, tuple[float, float]]] = {}
        for names in groups.values():
            if len(names) < 2:
                continue
            span = shared_limits([self[name].data for name in names], style)
            limits.update({name: {"clim": span} for name in names})
        return limits

    def to_stack(self) -> GeoStack:
        """Promote this window to a GeoStack — same layers, same headers, same features.

        The disk boundary lives on GeoStack, so this is the route to
        `to_zarr`. `model_context` does not survive: a stack is a surface,
        and context describes one window.

        Returns:
            GeoStack over the same pixels, each layer a GeoRaster.
        """
        from .stack import GeoStack

        return GeoStack(
            {name: tile.to_raster() for name, tile in self.items()},
            reference_layer=self.reference_layer,
        )

    # --- Model input ---

    def to_numpy(
        self,
        bands: Mapping[LayerName, Sequence[str]] | None = None,
        dtype: Mapping[LayerName, DTypeLike] | None = None,
        progress: bool = False,
    ) -> dict[LayerName, np.ndarray]:
        """Read every layer into plain arrays.

        Args:
            bands: Per layer, band names to keep in that order. A layer
                absent here keeps every band. None keeps all of every layer.
            dtype: Per layer, numpy dtype to cast to. A layer absent here
                keeps the dtype its own pixels carry.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            `{layer: array}`, each `(band, y, x)` or `(time, band, y, x)`,
            in this sample's own layer order.

        Raises:
            KeyError: `bands` or `dtype` names a layer that isn't here, or
                `bands` names a band a layer doesn't carry.
        """
        target_dtypes = dtype or {}
        self._validate_layer_keys(target_dtypes, "dtype")
        arrays = self._materialize_layers(bands or {}, progress)
        return {
            name: array if name not in target_dtypes else array.astype(target_dtypes[name])
            for name, array in arrays.items()
        }

    def to_tensor(
        self,
        bands: Mapping[LayerName, Sequence[str]] | None = None,
        dtype: Mapping[LayerName, torch.dtype] | None = None,
        progress: bool = False,
    ) -> dict[LayerName, torch.Tensor]:
        """Read every layer into tensors — same shape rules as `to_numpy`.

        Args:
            bands: Per layer, band names to keep in that order. A layer
                absent here keeps every band.
            dtype: Per layer, torch dtype to cast to. A layer absent here
                keeps the dtype its own pixels carry.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            `{layer: tensor}`, in this sample's own layer order.

        Raises:
            KeyError: `bands` or `dtype` names a layer that isn't here, or
                `bands` names a band a layer doesn't carry.
        """
        target_dtypes = dtype or {}
        self._validate_layer_keys(target_dtypes, "dtype")

        import torch

        arrays = self._materialize_layers(bands or {}, progress)
        layers: dict[LayerName, torch.Tensor] = {}
        for name, array in arrays.items():
            tensor = torch.from_numpy(array)
            target_dtype = target_dtypes.get(name)
            layers[name] = tensor if target_dtype is None else tensor.to(target_dtype)
        return layers

    @overload
    def to_sample(
        self,
        *,
        bands: Mapping[LayerName, Sequence[str]] | None = ...,
        dtype: Mapping[LayerName, torch.dtype] | None = ...,
        mode: Literal["tensor"] = ...,
        progress: bool = ...,
    ) -> TensorSample: ...

    @overload
    def to_sample(
        self,
        *,
        bands: Mapping[LayerName, Sequence[str]] | None = ...,
        dtype: Mapping[LayerName, DTypeLike] | None = ...,
        mode: Literal["numpy"],
        progress: bool = ...,
    ) -> NumpySample: ...

    def to_sample(
        self,
        *,
        bands: Mapping[LayerName, Sequence[str]] | None = None,
        dtype: Mapping[LayerName, Any] | None = None,
        mode: Literal["tensor", "numpy"] = "tensor",
        progress: bool = False,
    ) -> TensorSample | NumpySample:
        """Read this window into one sample — the dict a Dataset serves per item.

        Args:
            bands: Per layer, band names to keep in that order. A layer
                absent here keeps every band.
            dtype: Per layer, dtype to cast to — torch dtypes under
                `mode="tensor"`, numpy ones under `mode="numpy"`. A layer
                absent here keeps the dtype its own pixels carry.
            mode: Runtime to render for. `"tensor"` is what a DataLoader wants.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            {
                "layers": {
                    "<layer>": torch.Tensor | np.ndarray,  # (band, y, x) or (time, band, y, x)
                },
                "anchor": GeoAnchor,  # the grid every layer shares, with this sample's vector
                "model_context": {
                    "<key>": torch.Tensor | np.ndarray | str | None,
                },
            }

        Raises:
            KeyError: `bands` or `dtype` names a layer that isn't here, or
                `bands` names a band a layer doesn't carry.

        Examples:
            >>> sorted(sample.to_sample())
            ['anchor', 'layers', 'model_context']
        """
        if mode == "numpy":
            return {
                "layers": self.to_numpy(bands, dtype, progress),
                "anchor": self.anchor,
                "model_context": numpy_context(self.model_context),
            }
        return {
            "layers": self.to_tensor(bands, dtype, progress),
            "anchor": self.anchor,
            "model_context": tensor_context(self.model_context),
        }

    def _materialize_layers(
        self,
        selected: Mapping[LayerName, Sequence[str]],
        progress: bool,
    ) -> dict[LayerName, np.ndarray]:
        """Compute selected layers together so shared Dask tasks run once."""
        return read_windows([self], selected, progress)[0]

    def _validate_layer_keys(self, values: Mapping[LayerName, object], argument: str) -> None:
        """Reject layer options naming no layer in this sample."""
        missing = set(values) - set(self)
        if missing:
            raise KeyError(f"{argument} names missing layer(s) {sorted(missing)}; available: {list(self)}")


def read_windows(
    windows: Sequence[GeoTileStack],
    bands: Mapping[LayerName, Sequence[str]] | None = None,
    progress: bool = False,
) -> list[dict[LayerName, np.ndarray]]:
    """Read several windows in one Dask pass, so tasks they share run once.

    Windows cut from one surface overlap in the chunks they read: a window
    narrower than a chunk makes that chunk fetch once per window when each is
    computed alone. Handing them to one `dask.compute` collapses those reads,
    which matters most when the pixels come over the network.

    Args:
        windows: Windows to read, in the order results come back. Empty
            returns empty.
        bands: Per layer, band names to keep in that order. A layer absent
            here keeps every band. Applied to every window.
        progress: Show a dask progress bar while pixels compute.

    Returns:
        One `{layer: array}` per window, in the order given, each array
        `(band, y, x)` or `(time, band, y, x)`.

    Raises:
        KeyError: `bands` names a layer some window doesn't carry, or a band
            a layer doesn't carry.

    Examples:
        >>> for group in batched(stack.tiles(512), 16):
        ...     for pixels in read_windows(list(group)):
        ...         ...
    """
    from geosave_engine.geodata.utils.array import progress_bar

    selected = bands or {}
    layout: list[list[LayerName]] = []
    arrays = []
    for window in windows:
        window._validate_layer_keys(selected, "bands")
        names: list[LayerName] = []
        for name, tile in window.items():
            chosen = selected.get(name)
            names.append(name)
            arrays.append(tile.data if chosen is None else tile._select_bands(chosen))
        layout.append(names)

    if not arrays:
        return []
    with progress_bar(progress):
        values = dask_compute(*(array.data for array in arrays))

    read: list[dict[LayerName, np.ndarray]] = []
    cursor = 0
    for names in layout:
        read.append({name: np.asarray(values[cursor + offset]) for offset, name in enumerate(names)})
        cursor += len(names)
    return read

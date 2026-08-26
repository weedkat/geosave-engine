"""Private spatial-stack behavior shared by GeoStack and GeoTileStack."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeVar, Unpack

from geosave_engine.geodata.utils.datetime import bucket_days, is_finer
from geosave_engine.geodata.utils.spatial.geobox import geobox_matches

from ._array import _SpatialArray

if TYPE_CHECKING:
    from datetime import datetime as dt

    from geosave_engine.geodata.utils.datetime import Freq

    import holoviews as hv

    from geosave_engine.geodata.extensions import GeoExtension, TilingInfo, TimeSpec
    from geosave_engine.geodata.utils.datetime import DateRange
    from geosave_engine.geodata.viz import Kind, RenderStyle, ViewOptions

    from .anchor import GeoAnchor
    from .header import GeoHeader
    from .vector import GeoVector

LayerName = str

# Layer name a bare raster or tile is packed under when something needs a stack.
DEFAULT_LAYER: LayerName = "image"
LayerT = TypeVar("LayerT", bound=_SpatialArray)


class _SpatialStack(Mapping[LayerName, LayerT], Generic[LayerT]):
    """Named layers over one shared grid.

    One named reference layer is the stack's own identity — grid, time
    span, features, header. Every other layer contributes only pixels.

    Args:
        layers: `{name: layer}`, at least one, each a `LAYER_TYPE`, every
            one already on the reference layer's exact geobox.
        reference_layer: Layer whose anchor is this stack's. None uses the
            first layer.

    Raises:
        ValueError: No layer was given, `reference_layer` names none, or a
            layer's geobox differs from the reference layer's.
        TypeError: A layer isn't an instance of this class's `LAYER_TYPE`.
    """

    # The one layer type this class accepts — GeoRaster for a stack, GeoTile for a sample.
    LAYER_TYPE: ClassVar[type[_SpatialArray]]

    _layers: dict[LayerName, LayerT]
    _reference: LayerName

    def __init__(
        self,
        layers: Mapping[LayerName, LayerT],
        *,
        reference_layer: LayerName | None = None,
    ) -> None:
        self._reference = self._validate_layers(layers, reference_layer)
        self._layers = dict(layers)

    @classmethod
    def _validate_layers(
        cls,
        layers: Mapping[LayerName, _SpatialArray],
        reference_layer: LayerName | None,
    ) -> LayerName:
        """Check every layer's type, then that every geobox matches the reference's.

        Args:
            layers: Ordered layer mapping about to form a stack.
            reference_layer: Layer whose anchor is the stack's. None uses the first.

        Returns:
            The resolved reference layer name.

        Raises:
            ValueError: No layer was given, `reference_layer` names none, or
                a layer's geobox differs from the reference layer's.
            TypeError: A layer isn't an instance of `LAYER_TYPE`.
        """
        if not layers:
            raise ValueError(f"{cls.__name__} needs at least one layer")

        # every type first — reading .anchor off a non-layer would raise AttributeError instead
        checked: dict[LayerName, _SpatialArray] = {}
        for name, layer in layers.items():
            if not isinstance(layer, cls.LAYER_TYPE):
                raise TypeError(
                    f"layer {name!r} is a {type(layer).__name__}, "
                    f"{cls.__name__} holds {cls.LAYER_TYPE.__name__} layers"
                )
            checked[name] = layer

        reference = reference_layer if reference_layer is not None else next(iter(checked))
        if reference not in checked:
            raise ValueError(f"reference layer {reference!r} isn't present; available: {list(checked)}")

        reference_geobox = checked[reference].anchor.geobox
        for name, layer in checked.items():
            if not geobox_matches(layer.anchor.geobox, reference_geobox):
                raise ValueError(
                    f"layer {name!r} isn't on reference layer {reference!r}'s grid — "
                    f"call {name}.reproject_like({reference}) before stacking"
                )
        cls._validate_nesting(checked, reference)
        return reference

    @staticmethod
    def _bucket_freq(layer: _SpatialArray) -> Freq:
        """The frequency a layer's time axis buckets at.

        Args:
            layer: Layer carrying a time dim.

        Returns:
            Its `timespec`'s own freq, or `"D"` for an axis nothing
            resampled — raw observation instants stand for their own day.
        """
        spec = layer.timespec
        return spec.freq if spec is not None and spec.freq is not None else "D"

    @classmethod
    def _validate_nesting(cls, layers: Mapping[LayerName, _SpatialArray], reference: LayerName) -> None:
        """Check every timed layer buckets at least as coarsely as the reference.

        A coarser bucket contains a whole reference window, so each window draws
        exactly one of that layer's steps and a sample's step counts stay
        predictable. Timeless layers are exempt — they ride along, like a DEM.

        Args:
            layers: Layers about to form a stack, already type-checked.
            reference: Layer whose cadence the others must nest inside.

        Raises:
            ValueError: A layer buckets more finely than the reference,
                which would draw many of its steps per window.
        """
        anchored = layers[reference]
        if not anchored.has_time:
            return

        reference_freq = cls._bucket_freq(anchored)
        for name, layer in layers.items():
            if name == reference or not layer.has_time:
                continue
            layer_freq = cls._bucket_freq(layer)
            if is_finer(layer_freq, than=reference_freq):
                raise ValueError(
                    f"layer {name!r} buckets at {layer_freq} ({bucket_days(layer_freq):.4g} d), finer than "
                    f"reference layer {reference!r}'s {reference_freq} ({bucket_days(reference_freq):.4g} d) — "
                    f"call {name}.resample_time({reference_freq!r}, ...) before stacking"
                )

    # --- Header ---

    @property
    def reference_layer(self) -> LayerName:
        """Name of the layer whose anchor is this stack's own.

        Returns:
            Layer name.
        """
        return self._reference

    @property
    def _identity(self) -> LayerT:
        """The layer this stack takes its identity from.

        Returns:
            The reference layer.
        """
        return self._layers[self._reference]

    @property
    def anchor(self) -> GeoAnchor:
        """Where, when, and over which features — the reference layer's own anchor.

        Returns:
            The reference layer's anchor, unchanged. Other layers keep their
            own time spans and vectors, which are not consulted here.
        """
        return self._identity.anchor

    @property
    def vector(self) -> GeoVector | None:
        """Features over this extent, or None — `anchor`'s own.

        Returns:
            The reference layer's vector.
        """
        return self.anchor.vector

    @property
    def header(self) -> GeoHeader:
        """Every namespace this stack carries — the reference layer's own.

        Returns:
            The reference layer's header. Other layers keep their own,
            readable through `self[name].header`.
        """
        return self._identity.header

    @property
    def extensions(self) -> Mapping[str, GeoExtension]:
        """Every extension this stack carries, `{namespace: extension}`.

        Returns:
            The reference layer's extensions.
        """
        return self.header.extensions

    @property
    def tags(self) -> dict[str, str]:
        """Free-form descriptive strings this stack carries.

        Returns:
            The reference layer's tags. Empty if it has none.
        """
        return self.header.tags

    @property
    def timespec(self) -> TimeSpec | None:
        """How the reference layer's time axis was bucketed.

        Returns:
            Its `timespec`, or None if it was never resampled.
        """
        return self.header.timespec

    @property
    def tiling(self) -> TilingInfo | None:
        """Where this window sits in the grid a `tiles()` call cut.

        Every layer of one cut carries the reference layer's stamp, so this
        is the whole stack's.

        Returns:
            The tiling stamp, or None if this wasn't cut by one.
        """
        return self.header.tiling

    @property
    def group_id(self) -> str | None:
        """Which `tiles()` call this came from, if any (see `tiling`).

        Returns:
            The group id, or None.
        """
        return self.tiling.group_id if self.tiling is not None else None

    @property
    def tile_id(self) -> int | None:
        """This window's own position in its group's grid, if any (see `tiling`).

        Returns:
            The tile id, or None.
        """
        return self.tiling.tile_id if self.tiling is not None else None

    # --- Time ---

    @property
    def timespan(self) -> DateRange | None:
        """The window the reference layer declares — `anchor`'s own.

        Returns:
            Inclusive `(start, end)`, or None when it is timeless.
        """
        return self.anchor.timespan

    @property
    def start(self) -> dt | None:
        """Start of `timespan`.

        Returns:
            First instant, or None when the reference layer is timeless.
        """
        return self.anchor.start

    @property
    def end(self) -> dt | None:
        """End of `timespan`.

        Returns:
            Last instant, or None when the reference layer is timeless.
        """
        return self.anchor.end

    @property
    def has_time(self) -> bool:
        """Whether the reference layer carries a time dim.

        Returns:
            True if it does. Other layers may differ.
        """
        return self._identity.has_time

    @property
    def times(self) -> tuple[dt, ...]:
        """The reference layer's own time labels.

        Returns:
            One datetime per step, in order. Empty when it has no time dim.
        """
        return self._identity.times

    @property
    def time_buckets(self) -> tuple[DateRange, ...]:
        """The period each of the reference layer's steps stands for.

        Returns:
            Inclusive `(start, end)` per step, in `times` order. Empty when
            the reference layer has no time dim.
        """
        return self._identity.time_buckets

    @property
    def observed_time(self) -> DateRange | None:
        """Span the reference layer's own time labels cover.

        The counterpart to `timespan`, which is the window a caller declared
        and only has to contain these labels.

        Returns:
            Inclusive `(start, end)`, or None when it has no time dim.
        """
        return self._identity.observed_time

    # --- Mapping ---

    def __getitem__(self, layer: LayerName) -> LayerT:
        """One layer, as this class's own `LAYER_TYPE`.

        Args:
            layer: Layer name.

        Returns:
            The layer as given at construction.

        Raises:
            KeyError: `layer` names no layer here.
        """
        try:
            return self._layers[layer]
        except KeyError:
            raise KeyError(f"{layer!r} isn't a layer here: {list(self)}") from None

    def __iter__(self) -> Iterator[LayerName]:
        """Layer names, in the order they were given."""
        return iter(self._layers)

    def __len__(self) -> int:
        """Layer count."""
        return len(self._layers)

    def __repr__(self) -> str:
        layers = "\n".join(f"    {name}: {list(layer.bands)}" for name, layer in self.items())
        return f"{type(self).__name__}\n  layers:\n{layers}\n  {self.anchor!r}"

    # --- Layers ---

    def select(self, *names: LayerName) -> Self:
        """Keep only these layers, in this order.

        Args:
            *names: Layer names to keep. Must include the reference layer,
                whose anchor is this stack's identity.

        Returns:
            New instance holding only `names`.

        Raises:
            KeyError: A name isn't a layer here.
            ValueError: The reference layer isn't among `names`, which would
                move this stack's identity onto another layer.

        Examples:
            >>> stack.select("image", "label")
        """
        missing = [name for name in names if name not in self._layers]
        if missing:
            raise KeyError(f"layers {missing} aren't here: {list(self)}")
        if self._reference not in names:
            raise ValueError(
                f"select() must keep reference layer {self._reference!r}, whose anchor is this "
                f"{type(self).__name__}'s own; got {list(names)}"
            )
        return self._rebuild({name: self._layers[name] for name in names})

    def explore(
        self,
        *,
        kind: Kind | None = None,
        style: RenderStyle | None = None,
        rasterize: bool | None = None,
        **options: Unpack[ViewOptions],
    ) -> hv.core.Dimensioned:
        """Open every layer side by side, one panel each.

        Args:
            kind: Force one renderer for every layer instead of resolving
                each from its own `render` hints.
            style: Color policy applied to every layer. None takes the default.
            rasterize: Datashade on the server. None follows each layer's own type.
            **options: Per-view hvplot options — see `ViewOptions`.

        Returns:
            A holoviews Layout in this stack's own layer order, composable
            with `*`, `+` and `.opts()`. `holoviews.save(view, "view.html")`
            writes it as one self-contained page.

        Raises:
            ImportError: The `viz` extra isn't installed.

        Examples:
            >>> holoviews.save(stack.explore(width=400).cols(2), "stack.html")
        """
        import holoviews as hv

        return hv.Layout([
            layer.explore(kind=kind, style=style, rasterize=rasterize, **options).relabel(name)
            for name, layer in self.items()
        ])

    def set_reference(self, layer: LayerName) -> Self:
        """Move this stack's identity onto another layer.

        The reference layer is the stack's own identity — grid, time span,
        features, header — so this changes what the stack means, not just
        which name is marked.

        Args:
            layer: Layer to take the identity from.

        Returns:
            New instance over the same layers, anchored on `layer`.

        Raises:
            KeyError: `layer` isn't a layer here.

        Examples:
            >>> stack.timespan
            (datetime(2024, 1, 1, 0, 0), datetime(2024, 3, 31, 23, 59, 59, 999999))
            >>> stack.set_reference("dem").timespan
            None
        """
        if layer not in self._layers:
            raise KeyError(f"layer {layer!r} isn't here: {list(self)}")
        return self._rebuild(self._layers, reference_layer=layer)

    @abstractmethod
    def _rebuild(self, layers: Mapping[LayerName, LayerT], *, reference_layer: LayerName | None = None) -> Self:
        """Build the same stack type over replacement layers.

        Args:
            layers: Replacement layers, including the reference layer.
            reference_layer: Layer to anchor on. None keeps this stack's own.

        Returns:
            New instance of this class.
        """

"""Private spatial-array behavior shared by GeoRaster and GeoTile."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime as dt
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self, Unpack

import numpy as np
from numpy.typing import DTypeLike

from geosave_engine.geodata.extensions import (
    ArraySpec,
    GeoExtension,
    Legend,
    RenderHints,
    StacItems,
    TimeSpec,
    span_from_times,
)
from geosave_engine.geodata.utils.array import cast_nodata, safe_compute, same_nodata, validate_spatial
from geosave_engine.geodata.utils.datetime import parse_daterange
from geosave_engine.geodata.utils.spatial.geobox import geobox_matches
from geosave_engine.utils.fn import UNSET, Unset

from .anchor import GeoAnchor
from .header import GeoHeader, encode_attrs

if TYPE_CHECKING:
    import holoviews as hv
    import xarray as xr
    from affine import Affine
    from odc.geo.geom import Geometry

    from geosave_engine.geodata.extensions import TilingInfo
    from geosave_engine.geodata.utils.datetime import AnchorDatetime, DateRange
    from geosave_engine.geodata.viz import Kind, RenderStyle, ViewOptions

    from .vector import GeoVector


@dataclass(frozen=True, kw_only=True, eq=False)
class _SpatialArray:
    """One pixel array plus the anchor holding everything else about it.

    Base of `GeoTile` and `GeoRaster`. Every transform here preserves the
    grid, so a `tiling` stamp stays valid across it; grid-changing ones
    live on `GeoRaster`. Never built directly.

    Args:
        data: Canonical pixel array with dimensions `(band, y, x)` or
            `(time, band, y, x)`. Holds the
            geobox, band names, timestamps, dtype and nodata. Its
            `.attrs` mirror `anchor.header` for xarray interoperability.
        anchor: Grid, time span, features and header for `data`. Its geobox
            must be `data`'s own. A None `timespan` is filled in from
            `data`'s own time labels.
    """

    # Whether to trigger datashade
    RASTERIZE: ClassVar[bool] = True

    data: xr.DataArray
    anchor: GeoAnchor

    def __post_init__(self) -> None:
        """Validate canonical pixel data against its anchor.

        Raises:
            ValueError: `data` has invalid canonical dimensions, band or
                time coordinates, or no CRS;
                `anchor`'s geobox isn't `data`'s own; or `anchor`'s time
                span doesn't cover `data`'s own time labels.
        """
        data = validate_spatial(self.data)
        if not geobox_matches(data.odc.geobox, self.anchor.geobox):
            raise ValueError(
                f"anchor's geobox {self.anchor.geobox!r} isn't this data's own {data.odc.geobox!r}"
            )
        self.anchor._validate_time(data)

        # a bare ArraySpec is seeded so its own reconcile can read bands/times/nodata off the pixels
        extensions = {ArraySpec.NAMESPACE: ArraySpec(), **self.anchor.header.extensions}
        revalidated = GeoHeader(extensions, data=data)
        if revalidated != self.anchor.header:
            object.__setattr__(self, "anchor", replace(self.anchor, header=revalidated))

        # a dated array whose anchor carries no span dates the anchor from its own labels
        if self.anchor.timespan is None and "time" in data.dims:
            span = span_from_times(data.time.values, self.anchor.header.timespec)
            object.__setattr__(self, "anchor", self.anchor.rebase(timespan=span))

        # data.attrs always mirrors the current header, stray/stale registered keys included —
        # a caller reading .data directly (interop, plotting, xr.concat outside this library)
        # sees real metadata instead of an empty dict; the header itself stays on self.anchor
        stamped = encode_attrs(data.attrs, self.anchor.header, "json")
        if stamped != data.attrs:
            data = data.copy(deep=False)
            data.attrs = stamped
        object.__setattr__(self, "data", data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}\n  bands: {list(self.bands)}\n  {self.anchor!r}"

    # --- Header ---

    @property
    def header(self) -> GeoHeader:
        """Tags, extensions, tiling and timespec — this anchor's own."""
        return self.anchor.header

    @property
    def vector(self) -> GeoVector | None:
        """Features over this extent, or None — this anchor's own."""
        return self.anchor.vector

    # --- Derived properties ---

    @property
    def bands(self) -> tuple[str, ...]:
        """Band names from `data`'s mandatory `band` coordinate."""
        return tuple(str(band) for band in self.data.band.values)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Pixel dtype."""
        return self.data.dtype

    @property
    def num_bands(self) -> int:
        """Number of named bands."""
        return len(self.bands)

    @property
    def times(self) -> tuple[dt, ...]:
        """Observation datetimes from `data`'s own time coord. Empty when it has no time dim."""
        if "time" not in self.data.dims:
            return ()
        return tuple(value.astype("datetime64[us]").item() for value in self.data.time.values)

    @property
    def time_buckets(self) -> tuple[DateRange, ...]:
        """The period each time step stands for — one `(start, end)` per entry in `times`.

        A label names its bucket, not its whole span: after
        `resample_time("ME", "mean")` a step labelled 2024-01-31 stands for
        all of January. Steps never resampled each stand for their own day.

        Returns:
            Inclusive `(start, end)` per step, in `times` order, bucketed
            by `timespec`. Empty when this array has no time dim.
        """
        if "time" not in self.data.dims:
            return ()
        return tuple((self.timespec or TimeSpec()).bounds(self.data.time.values))

    @property
    def has_time(self) -> bool:
        """True if data has a time dimension."""
        return "time" in self.data.dims

    @property
    def observed_time(self) -> DateRange | None:
        """Span this array's own time labels cover — where the data actually is.

        The counterpart to `timespan`, which is the window a caller declared
        and only has to *contain* these labels. Both name the same span on
        an array nobody declared a window for.

        Returns:
            Inclusive `(start, end)` — earliest label's bucket start to
            latest label's bucket end, bucketed by `timespec`. None when
            this array has no time dim.
        """
        if "time" not in self.data.dims:
            return None
        return span_from_times(self.data.time.values, self.timespec)

    @property
    def nodata(self) -> float | int | None:
        """GDAL-standard nodata value, if set. None means no nodata declared."""
        return self.data.rio.nodata

    @property
    def footprint(self) -> Geometry | None:
        """Union of `vector`'s geometries. None if there's no vector."""
        return self.vector.footprint if self.vector is not None else None

    @property
    def tags(self) -> dict[str, str]:
        """Free-form descriptive strings this carries. Empty if none."""
        return self.header.tags

    @property
    def extensions(self) -> Mapping[str, GeoExtension]:
        """Every extension this carries, `{namespace: extension}`.

        Cut from a raster, a tile inherits the raster's own — `legend`
        included, so a prediction tile keeps its class map.
        """
        return self.header.extensions

    @property
    def timespec(self) -> TimeSpec | None:
        """How `data.time` was bucketed. None if it was never resampled."""
        return self.header.timespec

    @property
    def tiling(self) -> TilingInfo | None:
        """Where this window sits in the grid `GeoRaster.tiles()` cut. None if not cut by one."""
        return self.header.tiling

    @property
    def group_id(self) -> str | None:
        """Which `GeoRaster.tiles()` call this came from, if any (see `tiling`)."""
        return self.tiling.group_id if self.tiling is not None else None

    @property
    def tile_id(self) -> int | None:
        """This window's own position in its group's grid, if any (see `tiling`)."""
        return self.tiling.tile_id if self.tiling is not None else None

    # --- Builtin extension shortcuts ---

    @property
    def render(self) -> RenderHints | None:
        """The `"render"` extension — band display roles. None if this carries none."""
        hints = self.extensions.get(RenderHints.NAMESPACE)
        return hints if isinstance(hints, RenderHints) else None

    @property
    def legend(self) -> Legend | None:
        """The `"legend"` extension — what pixel values mean. None if this carries none."""
        legend = self.extensions.get(Legend.NAMESPACE)
        return legend if isinstance(legend, Legend) else None

    @property
    def stac(self) -> StacItems | None:
        """The `"stac"` extension — every source item, self-dating. None if this carries none."""
        provenance = self.extensions.get(StacItems.NAMESPACE)
        return provenance if isinstance(provenance, StacItems) else None

    # --- Anchor passthroughs — read-only, delegate to self.anchor's own ---

    @property
    def resolution(self) -> float:
        return self.anchor.resolution

    @property
    def affine(self) -> Affine:
        return self.anchor.affine

    @property
    def crs(self) -> str:
        return self.anchor.crs

    @property
    def width(self) -> int:
        return self.anchor.width

    @property
    def height(self) -> int:
        return self.anchor.height

    @property
    def shape(self) -> tuple[int, int]:
        return self.anchor.shape

    @property
    def area_m2(self) -> float:
        return self.anchor.area_m2

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.anchor.bounds

    @property
    def extent(self) -> Geometry:
        return self.anchor.extent

    @property
    def geographic_bounds(self) -> tuple[float, float, float, float]:
        return self.anchor.geographic_bounds

    @property
    def geographic_centroid(self) -> tuple[float, float]:
        return self.anchor.geographic_centroid

    @property
    def timespan(self) -> DateRange | None:
        return self.anchor.timespan

    @property
    def start(self) -> dt | None:
        return self.anchor.start

    @property
    def end(self) -> dt | None:
        return self.anchor.end

    @property
    def stem(self) -> str:
        return self.anchor.stem

    # --- Visualization ---

    def explore(
        self,
        *,
        kind: Kind | None = None,
        style: RenderStyle | None = None,
        rasterize: bool | None = None,
        band: str | None = None,
        time: dt | None = None,
        vector: bool = True,
        **options: Unpack[ViewOptions],
    ) -> hv.core.Dimensioned:
        """Open this array in an interactive holoviews view.

        This array's own `render` hints drive the renderer, and its own
        features are outlined over the pixels.

        Args:
            kind: Force a renderer instead of resolving one from `render`
                and the band count.
            style: Color policy. None takes the default.
            rasterize: Datashade on the server. None follows whether this
                type's pixels are known to fit in memory.
            band: Draw this band alone. None leaves a band widget when more
                than one band remains.
            time: Draw this timestamp alone. None leaves a time widget.
            vector: True outlines this array's own features over the pixels.
                False draws pixels alone.
            **options: Per-view hvplot options — see `ViewOptions`.

        Returns:
            A holoviews object, composable with `*`, `+` and `.opts()`.
            `holoviews.save(view, "view.html")` writes it as one
            self-contained page, no browser or server needed.

        Raises:
            ImportError: The `viz` extra isn't installed.
            KeyError: `band` or `time` names something this array doesn't carry.

        Examples:
            >>> holoviews.save(raster.explore(band="B08", width=800), "nir.html")
        """
        from geosave_engine.geodata.viz import DEFAULT_STYLE, to_element

        features = self.vector
        return to_element(
            self.data,
            render=self.render,
            legend=self.legend,
            kind=kind,
            style=style if style is not None else DEFAULT_STYLE,
            rasterize=self.RASTERIZE if rasterize is None else rasterize,
            band=band,
            time=time,
            vector=features.gdf if vector and features is not None else None,
            **options,
        )

    # --- Metadata ---

    def _rebase(
        self,
        *,
        data: xr.DataArray | Unset = UNSET,
        header: GeoHeader | Unset = UNSET,
        timespan: AnchorDatetime | None | Unset = UNSET,
        vector: GeoVector | None | Unset = UNSET,
        nodata: float | None | Unset = UNSET,
        timespec: TimeSpec | None | Unset = UNSET,
        extensions: Mapping[str, GeoExtension | Mapping[str, Any] | None] | None = None,
    ) -> Self:
        """Build the same spatial-array type with changed pixels or metadata.

        Every argument but `data`, `nodata` and `timespec` is forwarded to
        `GeoAnchor.rebase`. `timespec` is the sanctioned bypass around
        `TimeSpec.SETTABLE = False` — only `resample_time`/`concat` pass it.

        Args:
            data: New pixel array, any grid — the anchor moves onto its
                geobox. Omitted keeps the current pixels.
            header: Complete replacement header. Internal composition path;
                omitted keeps the current header.
            timespan: Datetime string, `(start, end)` pair, or None to clear
                the recorded span. Must still cover the result's own time
                labels. Omitted keeps the current span.
            vector: New features over this extent, or None to clear.
            nodata: GDAL-standard nodata value to set on data, or None to clear.
            timespec: How the time axis was bucketed, or None to clear it.
            extensions: Registered namespace updates, e.g.
                `{"render": {"class_map": {0: "bg", 1: "palm"}}}` or
                `{"tiling": None}`. A field mapping merges onto that
                namespace's current fields; a built instance replaces it
                whole; None drops the namespace.

        Returns:
            New instance of this class.

        Raises:
            ValueError: a field dict fails its extension's validation,
                `vector`'s CRS differs from the result's, a tiling stamp was
                left on while the shape changed, `nodata` can't be
                represented by the result's dtype, or a namespace declares
                `SETTABLE = False`.
            UnknownExtensionError: a keyword names an unregistered namespace.
        """
        pixels = self.data if isinstance(data, Unset) else data
        if not isinstance(nodata, Unset):
            # checked here so an out-of-range sentinel reads as it does from astype(), not as rioxarray's OverflowError
            pixels = pixels.rio.write_nodata(cast_nodata(nodata, pixels.dtype))

        # reading a geobox costs real time, so only re-read it when the pixels actually changed
        base_anchor = self.anchor if isinstance(header, Unset) else replace(self.anchor, header=header)
        anchor = base_anchor.rebase(
            geobox=UNSET if isinstance(data, Unset) else pixels.odc.geobox,
            timespan=timespan,
            vector=vector,
            **(extensions or {}),
        )
        if not isinstance(timespec, Unset):
            anchor = anchor._with_timespec(timespec)
        return replace(self, data=pixels, anchor=anchor)

    # --- Transforms ---

    def _select_bands(self, band: Sequence[str]) -> xr.DataArray:
        """This array's pixels narrowed to `band`, in that order.

        Args:
            band: Band names to keep. Must be non-empty and all present.

        Returns:
            Pixels carrying only `band`, in the order given.

        Raises:
            KeyError: A name isn't one of this array's bands.
            ValueError: `band` is empty.
        """
        if not band:
            raise ValueError(f"select(band=...) needs at least one band; this has {list(self.bands)}")
        missing = [name for name in band if name not in self.bands]
        if missing:
            raise KeyError(f"bands {missing} aren't in this {type(self).__name__}'s {list(self.bands)}")
        return self.data.sel(band=list(band))

    def select(
        self,
        *,
        band: Sequence[str] | None = None,
        time: AnchorDatetime | list[AnchorDatetime] | None = None,
    ) -> Self:
        """Keep these bands and/or the steps over these times.

        The axis counterpart of `crop`, which narrows space. Both axes
        narrow in one call, and either alone leaves the other whole.

        Args:
            band: Band names to keep, in this order. None keeps every band.
            time: A datetime string, an inclusive `(start, end)`, or a list
                of either. Every step whose own bucket overlaps any of them
                is kept — a monthly step is kept by a span touching any part
                of its month, not only one containing the whole month. A
                tuple of two is one range and a list of two is two separate
                spans: `("2024-01", "2024-03")` is January through March,
                `["2024-01", "2024-03"]` is January and March. None keeps
                every step.

        Returns:
            New instance over the same grid, steps in their existing order —
            `time` filters the axis, it never reorders it. A `time`
            selection re-reads the declared span off the kept labels'
            buckets and narrows `stac` provenance to it; a `band` selection
            clears a `render.rgb_bands` it leaves dangling. Self when
            neither was given.

        Raises:
            KeyError: A name in `band` isn't one of this array's bands.
            ValueError: `band` is empty, a `time` string is malformed,
                `time` was given for an array with no time dim, or no step's
                bucket overlaps it.

        Examples:
            >>> raster.select(band=["B08", "B04"]).bands
            ('B08', 'B04')
            >>> summer = raster.select(time=("2024-06", "2024-08"))
            >>> ends = raster.select(time=["2024-01", "2024-12"])
        """
        if band is None and time is None:
            return self

        data = self.data if band is None else self._select_bands(band)
        span: DateRange | Unset = UNSET
        extensions: dict[str, Any] = {}

        if time is not None:
            if not self.has_time:
                raise ValueError(f"select(time=...) needs a time dim; this {type(self).__name__} has none")
            # a list holds several spans; anything else is one AnchorDatetime, tuples included
            wanted = [parse_daterange(one) for one in time] if isinstance(time, list) else [parse_daterange(time)]
            keep = [
                step
                for step, (opens, closes) in enumerate(self.time_buckets)
                if any(opens <= end and start <= closes for start, end in wanted)
            ]
            if not keep:
                listed = ", ".join(f"{start}–{end}" for start, end in wanted)
                raise ValueError(f"no time step's bucket overlaps {listed}")
            data = data.isel(time=keep)
            # the labels only name their buckets, so the span comes off the bucket edges
            span = span_from_times(data.time.values, self.timespec)
            provenance = self.extensions.get(StacItems.NAMESPACE)
            if isinstance(provenance, StacItems):
                extensions[StacItems.NAMESPACE] = {"items": provenance.between(*span).items}

        return self._rebase(data=data, timespan=span, extensions=extensions or None)

    def squeeze_time(self) -> Self:
        """Drop a single-step time axis, leaving `(band, y, x)`.

        What a static layer needs before it rides along with a time series —
        a DEM or a land-cover map arrives with one step because its provider
        dated it, not because it varies over time.

        Returns:
            New instance with no time dim and no `timespec`, keeping its
            declared span. Self when there is no time dim to drop.

        Raises:
            ValueError: this array has more than one time step, which would
                mean choosing one.

        Examples:
            >>> dem.squeeze_time().data.dims
            ('band', 'y', 'x')
        """
        if "time" not in self.data.dims:
            return self
        steps = self.data.sizes["time"]
        if steps != 1:
            raise ValueError(
                f"squeeze_time() needs a single time step, got {steps} — reduce or select one first"
            )
        return self._rebase(data=self.data.isel(time=0, drop=True), timespec=None)


    def remap(self, mapping: dict[int, int]) -> Self:
        """Relabel pixel values — a raw dataset's own class encoding onto your training schema.

        The whole mapping lands at once, read off the original pixels, so
        `{1: 2, 2: 3}` never chains 1s onward to 3 and dict order can't
        change the result. A value outside `mapping` is left alone.

        Args:
            mapping: `{source pixel value: target pixel value}`, e.g.
                `{10: 1, 20: 2}` for a label raster encoded 10/20.

        Returns:
            New instance, same grid, dtype, nodata and header, values
            relabeled. Lazy in, lazy out.

        Raises:
            TypeError: This raster is not integer-valued, or a mapping
                key/value is not an integer.
            ValueError: A key/value is outside this dtype's range, or a
                key remaps the declared nodata sentinel.

        Examples:
            >>> label.remap({1: 2, 2: 1})  # swap two classes
        """
        if not mapping:
            return self
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"remap() needs an integer raster, got {self.dtype}")

        items = list(mapping.items())
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for pair in items
            for value in pair
        ):
            raise TypeError("remap() keys and values must be integers")

        limits = np.iinfo(self.dtype)
        outside = sorted({int(value) for pair in items for value in pair if not limits.min <= value <= limits.max})
        if outside:
            raise ValueError(f"remap() values {outside} are outside {self.dtype}'s {limits.min}..{limits.max} range")
        if self.nodata is not None and any(same_nodata(source, self.nodata) for source in mapping):
            raise ValueError(
                f"remap() cannot use declared nodata {self.nodata!r} as a source value — "
                "clear or replace nodata explicitly first"
            )

        remapped = self.data
        for source, target in mapping.items():
            remapped = remapped.where(self.data != source, other=target)
        return replace(self, data=remapped)

    def astype(
        self,
        dtype: DTypeLike,
        *,
        nodata: float | int | None | Unset = UNSET,
    ) -> Self:
        """Cast pixels to another dtype without changing their grid.

        Args:
            dtype: Target numpy dtype, e.g. `"uint16"` or `np.float32`.
            nodata: Target nodata sentinel. Omitted preserves the current
                sentinel; a different value replaces old nodata pixels
                before casting. None clears the declaration.

        Returns:
            New instance with lazily cast pixels and compatible nodata.

        Raises:
            ValueError: The target dtype cannot represent the resulting nodata.

        Examples:
            >>> labels.astype("uint16").remap({1: 300})
        """
        target = np.dtype(dtype)
        requested = self.nodata if isinstance(nodata, Unset) else nodata
        if (
            np.issubdtype(target, np.integer)
            and self.nodata is not None
            and np.isnan(self.nodata)
            and nodata is None
        ):
            raise ValueError("casting declared NaN nodata to integers needs an explicit integer nodata")
        target_nodata = cast_nodata(requested, target)

        pixels = self.data.rio.write_nodata(None)
        valid: xr.DataArray | None = None
        if not isinstance(nodata, Unset) and nodata is not None and not same_nodata(self.nodata, target_nodata):
            if self.nodata is not None:
                if np.isnan(self.nodata):
                    valid = ~pixels.isnull()
                    pixels = pixels.where(valid, other=0)
                else:
                    valid = pixels != self.nodata

        converted = pixels.astype(target)
        if valid is not None:
            converted = converted.where(valid, other=target_nodata)
        converted = converted.rio.write_nodata(target_nodata)
        return replace(self, data=converted)

    # --- Materialization ---

    def compute(self) -> Self:
        """Force this dask-backed data to compute, retrying past a transient read failure.

        Loads every pixel into memory — safe on a `GeoTile`, not on a
        `GeoRaster` bigger than RAM. Not safe to call from more than one
        thread of the same process at once.

        Returns:
            New instance of this class, same shape, computed (in-memory) data.

        Raises:
            TileDecodeError: GDAL logged a tile decode failure — after 3 retries.
        """
        return replace(self, data=safe_compute(self.data))

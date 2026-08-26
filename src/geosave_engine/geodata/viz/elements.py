"""Turn one georeferenced array into a holoviews element. See to_element.

One description, two backends: bokeh shows it interactively, matplotlib
renders it to a Figure. Nothing here knows about GeoTile or GeoAnchor — it
takes a DataArray plus the hints and features that go with it.
"""
from __future__ import annotations

from datetime import datetime as dt
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Sequence, TypedDict, Unpack

import numpy as np
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg

from geosave_engine.geodata.extensions import Legend, RenderHints
from geosave_engine.geodata.utils.array.nodata import mask_nodata
from geosave_engine.geodata.viz.style import DEFAULT_STYLE, RenderStyle

if TYPE_CHECKING:
    import geopandas as gpd
    import holoviews as hv
    from matplotlib.figure import Figure

# Which renderer draws an array: an RGB composite, a continuous ramp, or a class legend.
Kind = Literal["rgb", "continuous", "categorical"]
# Pixels per inch when a pixel width is translated for the matplotlib backend.
_FIGURE_DPI = 100


class ViewOptions(TypedDict, total=False):
    """Per-view hvplot options every drawing function forwards.

    A curated subset, so a caller keeps completion and a reader keeps a
    traceable signature. Anything set here wins over what `style` and
    `render` derive.
    """

    width: NotRequired[int]
    height: NotRequired[int]
    frame_width: NotRequired[int]
    frame_height: NotRequired[int]
    title: NotRequired[str]
    alpha: NotRequired[float]
    cmap: NotRequired[str | list[str]]
    clim: NotRequired[tuple[float, float]]
    colorbar: NotRequired[bool]


def resolve_kind(
    da: xr.DataArray,
    render: RenderHints | None,
    legend: Legend | None,
    kind: Kind | None,
) -> Kind:
    """Decide which renderer draws an array, without reading a pixel.

    Args:
        da: Array about to be drawn.
        render: Band display roles the array carries, or None.
        legend: Pixel-value meaning the array carries, or None.
        kind: Explicit choice, which always wins.

    Returns:
        The renderer to use. An array is RGB when hints name three bands or
        it carries exactly three, categorical when hints carry a legend, and
        continuous otherwise.
    """
    if kind is not None:
        return kind
    if (render is not None and render.rgb_bands is not None) or ("band" in da.dims and da.sizes["band"] == 3):
        return "rgb"
    if legend is not None and (legend.class_map is not None or legend.color_map is not None):
        return "categorical"
    return "continuous"


def shared_limits(arrays: Sequence[xr.DataArray], style: RenderStyle = DEFAULT_STYLE) -> tuple[float, float]:
    """One color range covering several arrays, so panels compare honestly.

    Args:
        arrays: Arrays to cover, at least one.
        style: Policy naming the percentiles and how many pixels to read.

    Returns:
        `(low, high)` spanning every array's own stretch.

    Raises:
        ValueError: No array was given.
    """
    if not arrays:
        raise ValueError("shared_limits() needs at least one array")
    spans = [_stretch_limits(mask_nodata(array), style) for array in arrays]
    return min(low for low, _ in spans), max(high for _, high in spans)


def to_element(
    da: xr.DataArray,
    *,
    render: RenderHints | None = None,
    legend: Legend | None = None,
    kind: Kind | None = None,
    style: RenderStyle = DEFAULT_STYLE,
    rasterize: bool = True,
    band: str | None = None,
    time: dt | None = None,
    vector: gpd.GeoDataFrame | None = None,
    **options: Unpack[ViewOptions],
) -> hv.core.Dimensioned:
    """Build a holoviews view of one georeferenced array.

    Nodata is masked first so it can't take over the color range, and a
    length-1 `band` or `time` dim is squeezed away rather than becoming a
    widget with one position. Dims that remain become widgets.

    Args:
        da: Array to view, with `y`/`x` dims.
        render: The array's own band display roles. None draws without them.
        legend: The array's own pixel-value meaning — class labels, class
            colors. None draws without them.
        kind: Force a renderer instead of resolving one from `render` and
            the band count.
        style: Color policy — stretch percentiles, default colormaps,
            sample cap, aspect.
        rasterize: Datashade on the server, so an array larger than the
            screen renders without being read. False sends every pixel.
        band: Draw this band alone. None keeps every band, which leaves a
            widget when more than one remains.
        time: Draw this timestamp alone. None keeps every step.
        vector: Features to outline over the pixels, in any CRS the
            geometries declare. None draws none.
        **options: Per-view hvplot options — see `ViewOptions`. Any given
            here wins over what `style` and `render` derive.

    Returns:
        A holoviews object, composable with `*`, `+` and `.opts()`.
        `holoviews.save(view, "view.html")` writes it as a self-contained page.

    Raises:
        KeyError: `band` or `time` names something the array doesn't carry.
        ValueError: An RGB composite was asked of an array whose bands
            don't support it.

    Examples:
        >>> to_element(raster.data, render=raster.render, band="B08")
    """
    import hvplot.xarray  # noqa: F401 — registers the .hvplot accessor

    selected = _select(da, band, time)
    # a length-1 band or time dim would otherwise become a widget with one position to pick
    masked = mask_nodata(selected).squeeze(drop=True)
    resolved = resolve_kind(masked, render, legend, kind)

    element = _draw(masked, render, legend, resolved, style, rasterize, options)
    if resolved == "categorical" and legend is not None and legend.class_map is not None:
        element = element * _legend(legend)
    if vector is not None:
        element = element * outline(vector)
    return element


def to_static_element(
    da: xr.DataArray,
    *,
    render: RenderHints | None = None,
    legend: Legend | None = None,
    kind: Kind | None = None,
    style: RenderStyle = DEFAULT_STYLE,
    band: str | None = None,
    time: dt | None = None,
    vector: gpd.GeoDataFrame | None = None,
    **options: Unpack[ViewOptions],
) -> hv.core.Dimensioned:
    """Build an element that draws as one panel, with no widget dimension left.

    A remaining widget dimension makes the matplotlib backend render an
    animation rather than a figure.

    Args:
        da: Array to draw, `y`/`x` plus at most length-1 extra dims.
        render: The array's own band display roles. None draws without them.
        legend: The array's own pixel-value meaning. None draws without it.
        kind: Force a renderer instead of resolving one.
        style: Color policy.
        band: Draw this band alone, which is how a multi-band array becomes
            one panel.
        time: Draw this timestamp alone.
        vector: Features to outline over the pixels.
        **options: Per-view hvplot options — see `ViewOptions`.

    Returns:
        A holoviews element over `y`/`x` alone, or an RGB composite.

    Raises:
        KeyError: `band` or `time` names something the array doesn't carry.
        ValueError: After selection, `da` still carries a dim beyond `y`/`x`
            with more than one entry, which a static panel cannot show.
    """
    selected = _select(da, band, time)
    extra = {str(dim): int(size) for dim, size in selected.sizes.items() if dim not in ("y", "x") and size > 1}
    if extra.get("band") == 3 or (render is not None and render.rgb_bands is not None):
        extra.pop("band", None)
    if extra:
        raise ValueError(
            f"a static panel draws one frame; {extra} holds more than one — "
            "pass band=/time= to pick one"
        )

    return to_element(
        selected, render=render, legend=legend, kind=kind, style=style, rasterize=False,
        vector=vector, **options
    )


def plot(
    da: xr.DataArray,
    *,
    render: RenderHints | None = None,
    legend: Legend | None = None,
    kind: Kind | None = None,
    style: RenderStyle = DEFAULT_STYLE,
    band: str | None = None,
    time: dt | None = None,
    vector: gpd.GeoDataFrame | None = None,
    **options: Unpack[ViewOptions],
) -> Figure:
    """Render one array to a static matplotlib Figure.

    The same element `to_element` builds, drawn by holoviews' matplotlib
    backend — the one path that needs no browser. Reads every pixel, so it
    belongs to arrays that fit in memory.

    Args:
        da: Array to draw, `y`/`x` plus at most length-1 extra dims.
        render: The array's own band display roles. None draws without them.
        legend: The array's own pixel-value meaning. None draws without it.
        kind: Force a renderer instead of resolving one.
        style: Color policy.
        band: Draw this band alone.
        time: Draw this timestamp alone.
        vector: Features to outline over the pixels.
        **options: Per-view hvplot options — see `ViewOptions`. `width` and
            `height` are pixel sizes, translated to figure inches here
            because the matplotlib backend sizes in inches.

    Returns:
        Matplotlib Figure.

    Raises:
        KeyError: `band` or `time` names something the array doesn't carry.
        ValueError: `da` still carries a widget dim after selection.

    Examples:
        >>> figure = plot(tile.data, render=tile.render, band="B08", width=900)
    """
    import holoviews as hv

    element = to_static_element(
        da, render=render, legend=legend, kind=kind, style=style, band=band, time=time,
        vector=vector, **options
    )
    return hv.render(element.opts(**_figure_size(options), backend="matplotlib"), backend="matplotlib")


def outline(vector: gpd.GeoDataFrame, **options: Unpack[ViewOptions]) -> hv.core.Dimensioned:
    """Draw features as outlines over pixels, in their own CRS.

    Args:
        vector: Features to draw. Their CRS must carry an EPSG code, which
            is what cartopy projects from.
        **options: Per-view hvplot options — see `ViewOptions`.

    Returns:
        A geoviews Path holoviews can overlay on a raster element.

    Raises:
        ImportError: The `viz` extra isn't installed.
        ValueError: The features declare no CRS, or one with no EPSG code.
    """
    import cartopy.crs as ccrs
    import geoviews as gv

    if vector.crs is None:
        raise ValueError("outline() needs features with a CRS — call gdf.set_crs(...) first")
    epsg = vector.crs.to_epsg()
    if epsg is None:
        raise ValueError(f"outline() needs an EPSG-coded CRS to project from, got {vector.crs.to_string()}")
    return gv.Path(vector, crs=ccrs.epsg(epsg)).opts(**options)


def fig_to_array(fig: Figure) -> np.ndarray:
    """Render a matplotlib figure to an `(H, W, 3)` uint8 RGB array.

    For a training logger's `add_image` or a saved PNG. Renders through its
    own `FigureCanvasAgg`, since `fig.canvas` follows whatever backend
    matplotlib is set to and may hold no pixel buffer.

    Args:
        fig: Figure to rasterize.

    Returns:
        `(H, W, 3)` uint8 RGB pixels.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3]


def _select(da: xr.DataArray, band: str | None, time: dt | None) -> xr.DataArray:
    """Narrow an array to one band and one timestamp.

    Args:
        da: Array to narrow.
        band: Band name, or None to keep every band.
        time: Timestamp, or None to keep every step.

    Returns:
        The array, narrowed where asked.

    Raises:
        KeyError: A name isn't present on its own dim.
    """
    if band is not None:
        if "band" not in da.dims:
            raise KeyError("band= was given but this array has no band dim")
        names = [str(name) for name in da.band.values]
        if band not in names:
            raise KeyError(f"band {band!r} isn't here: {names}")
        da = da.sel(band=band)
    if time is not None:
        if "time" not in da.dims:
            raise KeyError("time= was given but this array has no time dim")
        da = da.sel(time=time)
    return da


def _draw(
    da: xr.DataArray,
    render: RenderHints | None,
    legend: Legend | None,
    kind: Kind,
    style: RenderStyle,
    rasterize: bool,
    options: Mapping[str, Any],
) -> hv.core.Dimensioned:
    """Build the pixel element for one resolved renderer.

    Args:
        da: Array with nodata already masked.
        render: Band display roles the array carries, or None.
        legend: Pixel-value meaning the array carries, or None.
        kind: Renderer already resolved.
        style: Color policy.
        rasterize: Whether to datashade.
        options: Caller options, which win over anything derived here.

    Returns:
        The holoviews element for those pixels.
    """
    derived: dict[str, Any] = {"x": "x", "y": "y"}
    if style.aspect is not None:
        derived["data_aspect"] = style.aspect

    match kind:
        case "rgb":
            names = _resolve_rgb_bands(da, None if render is None else render.rgb_bands)
            composite = da.sel(band=list(names))
            low, high = _stretch_limits(composite, style)
            # rgb() takes values in [0, 1]; raw reflectance would clip to near-black
            scaled = ((composite - low) / (high - low)).clip(0.0, 1.0)
            return scaled.hvplot.rgb(bands="band", **{**derived, **options})

        case "categorical":
            derived["colorbar"] = True
            if legend is not None and legend.color_map is not None:
                values = sorted(legend.color_map)
                derived["cmap"] = [legend.color_map[value] for value in values]
                derived["clim"] = (float(values[0]), float(values[-1]))
            # interpolating between class values would invent classes that don't exist
            derived["rasterize"] = False
            return da.hvplot.image(**{**derived, **options})

        case _:
            derived["rasterize"] = rasterize
            if rasterize:
                # datashader rescales per view off pixels it already resampled; a range here would read the surface
                derived["cmap"] = style.sequential_cmap
            else:
                low, high = _stretch_limits(da, style)
                derived["clim"] = (low, high)
                derived["cmap"] = style.diverging_cmap if low < 0 < high else style.sequential_cmap
            return da.hvplot.image(**{**derived, **options})


def _legend(legend: Legend) -> hv.core.Dimensioned:
    """Build a labelled legend for a class raster.

    A colorbar's tick labels are backend-specific, so the legend is drawn as
    an overlay of named, empty markers — which both backends render.

    Args:
        legend: Legend carrying `class_map`, and optionally `color_map`.

    Returns:
        An NdOverlay holding one named marker per class.
    """
    import holoviews as hv

    assert legend.class_map is not None
    colors = legend.color_map or {}
    entries = {
        label: hv.Scatter([]).opts(color=colors.get(value, "#808080"), show_legend=True)
        for value, label in sorted(legend.class_map.items())
    }
    return hv.NdOverlay(entries).opts(show_legend=True)


def _figure_size(options: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a pixel width into the figure inches matplotlib sizes by.

    Args:
        options: Caller options, read for `width`/`height`.

    Returns:
        `{"fig_inches": ...}`, or empty when no size was asked for.
    """
    width, height = options.get("width"), options.get("height")
    if width is None and height is None:
        return {}
    if width is not None and height is not None:
        return {"fig_inches": (width / _FIGURE_DPI, height / _FIGURE_DPI)}
    return {"fig_inches": (width or height) / _FIGURE_DPI}


def _resolve_rgb_bands(da: xr.DataArray, rgb_bands: tuple[str, str, str] | None) -> tuple[str, str, str]:
    """Resolve which three band names are red, green and blue.

    Args:
        da: Array carrying a `band` dim.
        rgb_bands: Explicit names, or None to take a three-band array's own.

    Returns:
        `(red, green, blue)` band names.

    Raises:
        ValueError: `da` has no `band` dim, `rgb_bands` is None and `da`
            doesn't carry exactly three bands, or a named band is absent.
    """
    if "band" not in da.dims:
        raise ValueError("an RGB composite needs a band dim; this array has none")
    names = tuple(str(name) for name in da.coords["band"].values)
    if rgb_bands is None:
        if len(names) != 3:
            raise ValueError(
                f"{len(names)} bands {list(names)} — which three are R/G/B is ambiguous. "
                "Pass render=RenderHints(rgb_bands=(red, green, blue))."
            )
        return names[0], names[1], names[2]
    missing = [name for name in rgb_bands if name not in names]
    if len(rgb_bands) != 3 or missing:
        raise ValueError(f"rgb_bands must name three of this array's bands {list(names)}, got {list(rgb_bands)}")
    return rgb_bands


def _stretch_limits(da: xr.DataArray, style: RenderStyle) -> tuple[float, float]:
    """Percentile limits for the color range, read off a decimated sample.

    Args:
        da: Array to sample, nodata already masked.
        style: Policy naming the percentiles and how many pixels to read.

    Returns:
        `(low, high)`. `(0.0, 1.0)` when every sampled value is nodata, and
        a unit-wide range when they are all equal.
    """
    step = max(1, int(np.sqrt(da.y.size * da.x.size / style.sample_cap)))
    sample = da.isel(y=slice(None, None, step), x=slice(None, None, step)).values
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = (float(value) for value in np.percentile(finite, style.stretch))
    return (low, low + 1.0) if high <= low else (low, high)

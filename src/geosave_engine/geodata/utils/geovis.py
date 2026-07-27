"""Auto-typed plotting for GeoTile.

One entry point, ``plot()``. No caller-picked layer type (there's no more
``RGBLayer``/``ContinuousLayer``/``LabelLayer`` to choose between) — each
tile's own band count, dtype, and value range decide how it's rendered:

- Exactly 3 bands -> RGB, 2–98 percentile stretched, band order as stored.
- More than 3 bands -> RGB too, but which 3 count as R/G/B is genuinely
  ambiguous from shape alone — pass ``rgb_bands=(r_name, g_name, b_name)``
  or this raises, naming the tile and its available bands, rather than
  guess. Resolved per panel by band *name*, not position, so one shared
  ``rgb_bands`` value works correctly across several differently-ordered
  multiband tiles in the same call — each panel looks its own names up in
  its own ``tile.bands``.
- 1 band, floating point -> continuous colormap.
- 1 band, integer with a small number of distinct values -> categorical
  (a label map), auto-paletted unless ``class_map``/``color_map`` is given.
- Anything else (2 bands, or an integer band with too many distinct values
  to plausibly be a label map) -> band 0 shown continuous, flagged in the
  panel caption.

Every panel also outlines ``tile.polygon`` (the exact AOI footprint, when
set) on top of the image, at ``polygon_alpha`` opacity — ``0`` hides it, a
tile with no polygon draws nothing regardless of the value.

Multiple tiles are grouped by ``(name, bands, date)`` first, then split into
connected components by actual bbox adjacency/overlap (not just centroid
proximity — an intentional tiling grid has *different* centroids by
design) — only tiles that are genuinely part of one contiguous area *and*
the same named layer mosaic together (via ``geosave_engine.geodata.tile.mosaic``);
anything else facets as its own panel, even if it happens to share bands
and date. Name comes from the input: a ``dict[str, GeoTile]``/``GeoStack``
gives each tile its own layer name (so two different layers, e.g. imagery
and a cloud mask, never mosaic into each other even if they happen to share
band names and footprint); a bare ``GeoTile``/sequence has no name, so every
tile shares the same empty one. Any tile carrying its own time dimension
(even a leftover length-1 one from a single-scene pull) is split into one
panel per timestep first, regardless of how many tiles were passed in.

Each panel's title is its layer name (empty for unnamed input) — never
derived from a tile's ``metadata`` (an unenforced, general-purpose bag, not
a schema). Each panel's own time/place caption sits at the bottom instead
— there's no one global caption, since panels in a facet grid can
legitimately differ on both. Pass ``show_metadata=True`` to also print a
panel's raw ``tile.metadata`` there.
"""
from __future__ import annotations

import dataclasses
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal, Mapping, Sequence, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from odc.geo.math import apply_affine
from typing_extensions import Unpack

from geosave_engine.geodata.tile import GeoTile, mosaic
from geosave_engine.utils.colorize import Palette, colorize

# DejaVu Sans (matplotlib's default) has no CJK glyphs — a reverse-geocoded
# address in a script it can't render (e.g. "岭脚村" in a panel caption) is a
# real, already-visible limitation (renders as a missing-glyph box), not
# something bundling a CJK font here would fix. This just silences the
# duplicate warning-per-glyph noise on top of that, nothing else.
warnings.filterwarnings("ignore", message=r"Glyph \d+ .*missing from font", category=UserWarning)

_CATEGORICAL_MAX_CLASSES = 32  # beyond this, an integer band is probably not a label map
_NODATA_FACECOLOR = "#c8c8c8"  # shows through transparent nodata pixels — distinct from white page bg
_LEGEND_COLUMN_LEFT = 0.82  # figure-fraction x: where the shared legend column starts

PanelKind = Literal["rgb", "continuous", "categorical", "fallback"]


@dataclass
class _Panel:
    kind: PanelKind
    tile: GeoTile
    title: str
    rgb_bands: tuple[str, str, str] = ("", "", "")
    cmap: str = "viridis"
    class_map: dict[int, str] | None = None
    color_map: Palette | None = None
    polygon_alpha: float = 0.8


def _resolve_rgb_bands(tile: GeoTile, title: str, rgb_bands: tuple[str, str, str] | None) -> tuple[str, str, str]:
    if tile.num_bands == 3 and rgb_bands is None:
        return cast("tuple[str, str, str]", tile.bands)
    if rgb_bands is None:
        raise ValueError(
            f"{title!r}: {tile.num_bands} bands {tile.bands} — which 3 are R/G/B is ambiguous. "
            f"Pass rgb_bands=(r_name, g_name, b_name) to plot(), or set it once via "
            f"tile.with_plot_meta(rgb_bands=(r_name, g_name, b_name))."
        )
    missing = [b for b in rgb_bands if b not in tile.bands]
    if missing:
        raise ValueError(f"{title!r}: rgb_bands {missing} not in this tile's bands {tile.bands}")
    return rgb_bands


def _detect_panel(
    tile: GeoTile,
    title: str,
    *,
    cmap: str,
    class_map: dict[int, str] | None,
    color_map: Palette | None,
    rgb_bands: tuple[str, str, str] | None,
    polygon_alpha: float,
) -> _Panel:
    """Pick a renderer for `tile`, preferring its own `plot_meta` over the call-level fallback."""
    rgb_bands = tile.plot_meta.rgb_bands or rgb_bands
    class_map = tile.plot_meta.class_map or class_map
    color_map = tile.plot_meta.color_map or color_map
    if tile.num_bands >= 3:
        return _Panel(
            "rgb", tile, title, rgb_bands=_resolve_rgb_bands(tile, title, rgb_bands), polygon_alpha=polygon_alpha
        )
    if tile.num_bands == 1:
        if np.issubdtype(tile.data.dtype, np.floating):
            return _Panel("continuous", tile, title, cmap=cmap, polygon_alpha=polygon_alpha)
        n_unique = int(np.unique(tile.data.values).size)
        if n_unique <= _CATEGORICAL_MAX_CLASSES:
            if class_map is None and color_map is None:
                warnings.warn(
                    f"{title!r}: categorical tile has no class_map/color_map — auto-palette used, "
                    f"values shown as raw ints. Set via tile.with_plot_meta(class_map=..., color_map=...) "
                    f"for readable labels.",
                    stacklevel=2,
                )
            return _Panel(
                "categorical", tile, title, class_map=class_map, color_map=color_map, polygon_alpha=polygon_alpha
            )
        return _Panel("continuous", tile, title, cmap=cmap, polygon_alpha=polygon_alpha)
    return _Panel("fallback", tile, title, cmap=cmap, polygon_alpha=polygon_alpha)  # 2 bands, no clean mapping


def _as_mpl_color(color: tuple[int, int, int] | str) -> tuple[float, float, float] | str:
    if isinstance(color, str):
        return color
    r, g, b = color
    return (r / 255, g / 255, b / 255)


def _default_palette(classes: list[int]) -> dict[int, tuple[int, int, int]]:
    tab20 = plt.get_cmap("tab20")
    palette: dict[int, tuple[int, int, int]] = {}
    for i, c in enumerate(classes):
        r, g, b = tab20(i % 20)[:3]
        palette[c] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def _stretch(channel: np.ndarray) -> np.ma.MaskedArray:
    """2-98 percentile stretch to [0, 1] — nodata pixels masked, not colored.

    Clipping the *masked* array (not the raw one) keeps the mask through to
    the caller — plain `np.clip` on `channel` itself would leave nodata as
    bare `NaN`, which `imshow` has no sensible color for and ends up
    rendering as blank white, indistinguishable from an unfilled axes.
    """
    masked = np.ma.masked_invalid(channel)
    try:
        lo, hi = np.percentile(masked.compressed(), [2, 98])
    except ValueError:
        lo, hi = float(np.nanmin(channel)), float(np.nanmax(channel))
    return np.ma.clip((masked - lo) / (hi - lo + 1e-8), 0, 1)


def _time_str(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


_CAPTION_WRAP_WIDTH = 50


def _panel_caption(tile: GeoTile) -> str:
    """Per-panel time + place caption — each panel may genuinely differ on both."""
    time_str = _time_str(tile.start) if tile.start == tile.end else f"{_time_str(tile.start)} → {_time_str(tile.end)}"
    address = tile.location.get("address")
    geo_str = f"{address}  ·  {tile.coordinate_str}" if address else tile.coordinate_str
    return f"{time_str}\n{textwrap.fill(geo_str, width=_CAPTION_WRAP_WIDTH)}"


def _draw_polygon(ax: Axes, tile: GeoTile, alpha: float) -> None:
    """Outline `tile.polygon` (exact AOI footprint) on top of the plotted image, if set.

    Converts to pixel space via the tile's own affine transform (inverted),
    the same coordinate system `imshow` already draws the array in — no
    `extent=` needed on the image itself.

    Uses `.exterior.points` (raw native-CRS coordinates), not `.geojson()`
    — the latter always reprojects to WGS84 regardless of the geometry's
    actual CRS (correct per the GeoJSON spec, wrong for pixel conversion).
    """
    if tile.polygon is None or alpha <= 0:
        return
    poly = tile.polygon.to_crs(tile.crs) if tile.crs else tile.polygon
    xs, ys = zip(*poly.exterior.points)
    # affine.Affine subclasses namedtuple("Affine", ...) — same name as the outer
    # class — a known pyright false positive: __invert__'s untyped
    # tuple.__new__(self.__class__, ...) return doesn't collapse back to the
    # same Affine identity apply_affine's param expects, though it's the same
    # runtime class (verified numerically identical to the old per-point loop).
    px, py = apply_affine(~tile.affine, np.asarray(xs), np.asarray(ys))  # pyright: ignore[reportArgumentType]
    ax.plot(px, py, color="red", linewidth=1.5, alpha=alpha)


def _render_rgb(ax: Axes, panel: _Panel) -> list[Patch] | None:
    """Stretch each requested band 2-98 percentile and stack as an RGBA image.

    ``to_numpy(bands=...)`` returns exactly one array per name in
    ``panel.rgb_bands`` in that order — iterate that count, not a hardcoded
    3, so a mismatch between the two would surface as a clear shape error
    instead of a wrong picture.

    A 4th (alpha) channel is added from the per-band nodata masks — a pixel
    nodata in *any* band renders fully transparent (``_NODATA_FACECOLOR``
    showing through, set on the axes in ``_render``) instead of opaque
    `(NaN, NaN, NaN)`, which `imshow` can't draw a color for and ends up
    blank white — indistinguishable from an unfilled axes.
    """
    arr = panel.tile.to_numpy(bands=list(panel.rgb_bands)).astype("float32")
    channels = [_stretch(arr[i]) for i in range(len(panel.rgb_bands))]
    rgb = np.ma.stack(channels, axis=-1)
    opaque = ~np.ma.getmaskarray(rgb).any(axis=-1)
    rgba = np.dstack([rgb.filled(0.0), opaque.astype("float32")])
    ax.imshow(rgba)
    return None


def _render_single_band(ax: Axes, panel: _Panel) -> list[Patch] | None:
    """Continuous colormap for one band — also the fallback for an unclassifiable one."""
    arr = np.ma.masked_invalid(panel.tile.to_numpy()[0])
    im = ax.imshow(arr, cmap=panel.cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return None


def _render_categorical(ax: Axes, panel: _Panel) -> list[Patch] | None:
    """Palette-colorized label map. Returns legend handles — plot() draws them
    stacked in one figure-level column instead of per-panel, so a panel's
    own legend doesn't shrink that panel's image."""
    arr = panel.tile.to_numpy()[0].astype(int)
    classes = sorted(np.unique(arr).tolist())
    palette: Palette = panel.color_map or _default_palette(classes)
    ax.imshow(colorize(arr, palette))
    labels = panel.class_map or {}
    return [Patch(color=_as_mpl_color(palette[c]), label=labels.get(c, str(c))) for c in classes]


_PANEL_RENDERERS: dict[PanelKind, Callable[[Axes, _Panel], list[Patch] | None]] = {
    "rgb": _render_rgb,
    "continuous": _render_single_band,
    "fallback": _render_single_band,
    "categorical": _render_categorical,
}


def _panel_caption_text(panel: _Panel, show_metadata: bool) -> str:
    caption = _panel_caption(panel.tile)
    if panel.kind == "fallback":
        caption = f"band 0 of {panel.tile.num_bands} — no RGB/label mapping\n{caption}"
    if show_metadata and panel.tile.metadata:
        meta_str = ", ".join(f"{k}={v!r}" for k, v in panel.tile.metadata.items())
        caption = f"{caption}\n{meta_str}"
    return caption


def _render(ax: Axes, panel: _Panel, show_metadata: bool) -> list[Patch] | None:
    ax.set_facecolor(_NODATA_FACECOLOR)
    handles = _PANEL_RENDERERS[panel.kind](ax, panel)
    # imshow sets a tight view limit; ax.plot() below defaults to a 5% autoscale
    # margin around its own points and wins over imshow's tight one, leaving a
    # blank gap past the raster's real edge — pin the view to the raster before
    # the polygon line can touch it, so the line only ever overlays, never resizes.
    raster_xlim, raster_ylim = ax.get_xlim(), ax.get_ylim()
    _draw_polygon(ax, panel.tile, panel.polygon_alpha)
    ax.set_xlim(raster_xlim)
    ax.set_ylim(raster_ylim)
    ax.set_title(panel.title, fontsize=10, fontweight="bold")
    ax.set_xlabel(_panel_caption_text(panel, show_metadata), fontsize=7, color="#666666", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    return handles


def _split_time(tile: GeoTile) -> list[GeoTile]:
    """Flatten a tile's own time dim (if any) into standalone single-timestep tiles.

    `has_time` is True the moment a "time" dim exists at all, even a
    length-1 one left over from a single-scene STAC pull — every ingested
    tile needs this, not just a lone tile passed alone, or a leftover time
    axis silently survives into `to_numpy()`'s output shape and throws off
    every downstream index that assumes `(band, y, x)`.

    `with_data()` alone would leave `.datetime` as the original (start !=
    end) range — every split-off tile would then collide into one bogus
    group — so `.datetime` is rewritten to the split timestep's own real
    time too.
    """
    if not tile.has_time:
        return [tile]
    return [
        dataclasses.replace(tile, data=tile.data.isel(time=i), datetime=tile.times[i]) for i in range(len(tile.times))
    ]


def _group_key(name: str, tile: GeoTile) -> tuple[str, tuple[str, ...], str]:
    return (name, tile.bands, tile.start.date().isoformat())


def _adjacent_components(named: list[tuple[str, GeoTile]]) -> list[list[tuple[str, GeoTile]]]:
    """Split (name, tile) pairs into groups that actually touch/overlap — not just share bands+date.

    A tiling grid's adjacent tiles have different centroids by design, so
    centroid proximity can't tell "meant to mosaic" apart from "unrelated
    location that coincidentally shares bands and date." Real bbox
    adjacency can. Every pair here already shares one `_group_key` (name
    included), so within one component there's nothing left to mismatch on.
    """
    n = len(named)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = named[i][1], named[j][1]
            if ti.crs == tj.crs and ti.bbox_polygon.intersects(tj.bbox_polygon):
                union(i, j)

    components: dict[int, list[tuple[str, GeoTile]]] = {}
    for i, pair in enumerate(named):
        components.setdefault(find(i), []).append(pair)
    return list(components.values())


class PlotKwargs(TypedDict, total=False):
    """Keyword args accepted by `plot()` — also what `GeoTile.plot()` forwards."""

    cmap: str
    class_map: dict[int, str] | None
    color_map: Palette | None
    rgb_bands: tuple[str, str, str] | None
    polygon_alpha: float
    title: str
    show_metadata: bool


def plot(
    tiles: GeoTile | Sequence[GeoTile] | Mapping[str, GeoTile], cols: int | None = None, **kwargs: Unpack[PlotKwargs]
) -> tuple[Figure, np.ndarray]:
    """Plot one or more GeoTiles, auto-picking a renderer per tile.

    Grouping, one tile pair at a time — same rule regardless of how many
    tiles are passed:

    | Same name? | Same date? | Same/adjacent location? | Result |
    | --- | --- | --- | --- |
    | Yes | Yes | Yes (touch or overlap) | One mosaicked panel (via ``mosaic()``) |
    | No | — | — | Two separate panels — different layers never mosaic |
    | Yes | Yes | No (unrelated location) | Two separate panels — same date alone isn't enough |
    | Yes | No | Yes (same/overlapping spot, different day) | Two separate panels — a time series, not a mosaic |

    Concretely: an ingested tiling grid (adjacent chips, same layer, same
    acquisition day) mosaics into one continuous image. A handful of
    anchors sampled from across a whole training set (different places,
    maybe different days) facets one panel per anchor. A single location
    revisited over time facets one panel per date. Two unrelated locations
    that happen to share an acquisition date do **not** silently merge into
    one panel just because the group key matches — adjacency is checked
    for real (bbox intersects), not guessed from date/bands alone. Two
    *different* layers (e.g. imagery and a cloud mask) never mosaic into
    each other either, even if they happen to share bands/date/footprint —
    name is part of the grouping key.

    Any tile carrying its own time dimension — one tile alone, or one of
    several passed in together — is split into one panel per timestep
    first, then run through the same rule (so a multi-date tile still
    facets by date exactly like separate single-date tiles would). Each
    resulting panel carries its own time + place caption at the bottom —
    there's no one global caption, since panels can legitimately differ on
    both.

    Args:
        tiles: One GeoTile, a sequence of them (no layer names — every
            panel titles empty), or a `dict[str, GeoTile]`/`GeoStack` (each
            tile's dict key becomes its panel's title, and its own
            grouping-key name — see above).
        cmap: Colormap for single-band float (continuous) tiles. Default
            `"viridis"` — only used for a tile whose own
            `plot_meta` doesn't already set one.
        class_map: ``{value: name}`` for single-band integer (categorical)
            tiles — fallback for a tile whose own `plot_meta.class_map`
            isn't set.
        color_map: ``{value: hex_or_rgb}`` for the same — fallback for a
            tile whose own `plot_meta.color_map` isn't set; auto-generated
            from a fixed palette if neither is given.
        rgb_bands: ``(r_name, g_name, b_name)`` for tiles with more than 3
            bands — fallback for a tile whose own `plot_meta.rgb_bands`
            isn't set; required (from either source) in that case, since
            which 3 count as color is ambiguous from shape alone. Ignored
            for exactly-3-band tiles (band order as stored).
        cols: Facet grid column count. Auto-sized (max 4) if omitted.
        polygon_alpha: Opacity of a panel's ``tile.polygon`` outline (the
            exact AOI footprint, when set — e.g. from ``from_polygon``/
            ``from_geojson``), ``0`` hides it. Default `0.8`. A tile with no
            polygon (built from a bbox/coordinate) draws nothing regardless.
        title: Optional figure suptitle — distinct from each panel's own
            title (the layer name).
        show_metadata: Also print a panel's raw ``tile.metadata`` at the
            bottom, alongside its time/place caption. Default `False`.

    Returns:
        ``(Figure, ndarray of Axes)`` — same shape whether one panel or several.

    Raises:
        ValueError: ``tiles`` is empty; a tile has more than 3 bands and
            no ``rgb_bands`` was resolved from either its own `plot_meta`
            or the call-level kwarg; ``rgb_bands`` names a band a tile
            doesn't have; or adjacent tiles can't mosaic (mismatched CRS
            without reconciliation, etc — see ``mosaic()``).
    """
    cmap = kwargs.get("cmap", "viridis")
    class_map = kwargs.get("class_map")
    color_map = kwargs.get("color_map")
    rgb_bands = kwargs.get("rgb_bands")
    polygon_alpha = kwargs.get("polygon_alpha", 0.8)
    title = kwargs.get("title", "")
    show_metadata = kwargs.get("show_metadata", False)

    entries: list[tuple[str, GeoTile]] = []
    if isinstance(tiles, GeoTile):
        entries.append(("", tiles))
    elif isinstance(tiles, Mapping):
        for name, tile in tiles.items():
            if not isinstance(name, str) or not isinstance(tile, GeoTile):
                raise TypeError(
                    f"plot() Mapping input must be dict[str, GeoTile], got key {name!r} "
                    f"({type(name).__name__}) -> value {tile!r} ({type(tile).__name__})"
                )
            entries.append((name, tile))
    else:
        for tile in tiles:
            if not isinstance(tile, GeoTile):
                raise TypeError(f"plot() sequence input must contain only GeoTile, got {tile!r} ({type(tile).__name__})")
            entries.append(("", tile))
    if not entries:
        raise ValueError("plot() needs at least one GeoTile")

    named: list[tuple[str, GeoTile]] = []
    for name, tile in entries:
        for split in _split_time(tile):
            named.append((name, split))

    groups: dict[tuple[str, tuple[str, ...], str], list[tuple[str, GeoTile]]] = {}
    for name, t in named:
        groups.setdefault(_group_key(name, t), []).append((name, t))

    components: list[list[tuple[str, GeoTile]]] = []
    for group in groups.values():
        components.extend(_adjacent_components(group))

    panels = [
        _detect_panel(
            mosaic([t for _, t in component]) if len(component) > 1 else component[0][1],
            title=component[0][0],
            cmap=cmap,
            class_map=class_map,
            color_map=color_map,
            rgb_bands=rgb_bands,
            polygon_alpha=polygon_alpha,
        )
        for component in components
    ]

    n = len(panels)
    cols = cols or min(4, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.6 * rows), squeeze=False)
    axes_flat = axes.flatten()
    legend_groups: list[tuple[str, list[Patch]]] = []
    for ax, panel in zip(axes_flat, panels):
        handles = _render(ax, panel, show_metadata)
        if handles:
            legend_groups.append((panel.title, handles))
    for ax in axes_flat[n:]:
        ax.axis("off")

    right = _LEGEND_COLUMN_LEFT if legend_groups else 1.0
    top = 0.96 if title else 1.0
    fig.tight_layout(rect=(0, 0, right, top))
    if title:
        fig.suptitle(title, fontsize=11, y=0.99)
    # Pass 1: draw each legend to learn its real rendered height (only known
    # after a draw) — position doesn't matter yet, only the measurement.
    _LEGEND_GAP = 0.02
    legend_x = right + 0.02
    legends: list[tuple[Legend, float]] = []
    for legend_title, handles in legend_groups:
        leg = fig.legend(
            handles=handles,
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(legend_x, 0.95),
            fontsize=8,
            title_fontsize=9,
        )
        fig.canvas.draw()
        renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
        bbox_fig = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        legends.append((leg, bbox_fig.height))

    # Pass 2: reposition as one block, vertically centered in the column —
    # evenly dividing figure height by legend count ignored each legend's
    # actual size, leaving a big gap next to a short one; stacking from the
    # top left the same gap pushed to the bottom instead. Centering the
    # whole (tightly-spaced) block fixes both.
    total_height = sum(h for _, h in legends) + _LEGEND_GAP * (len(legends) - 1)
    legend_y = min(0.95, 0.5 + total_height / 2)
    for leg, height in legends:
        leg.set_bbox_to_anchor((legend_x, legend_y), transform=fig.transFigure)
        legend_y -= height + _LEGEND_GAP
    fig.canvas.draw()
    return fig, axes


def fig_to_array(fig: Figure) -> np.ndarray:
    """Render a matplotlib figure to an ``(H, W, 3)`` uint8 RGB array.

    For handing a rendered figure to something that wants a plain array —
    a training logger's ``add_image``, a saved PNG — rather than an
    interactive ``Figure``. Renders through its own ``FigureCanvasAgg``,
    not ``fig.canvas`` — that's typed ``FigureCanvasBase``, which has no
    pixel buffer, and its actual runtime type follows whatever backend
    matplotlib is set to, not necessarily Agg.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3]

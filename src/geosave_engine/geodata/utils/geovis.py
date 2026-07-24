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

Multiple tiles are grouped by ``(bands, date)`` first, then split into
connected components by actual bbox adjacency/overlap (not just centroid
proximity — an intentional tiling grid has *different* centroids by
design) — only tiles that are genuinely part of one contiguous area mosaic
together (via ``geosave_engine.geodata.tile.mosaic``); anything else facets
as its own panel, even if it happens to share bands and date. A single tile
with its own time dimension is split into one panel per timestep the same
way. Each panel gets its own time/place caption — there's no one global
caption, since panels in a facet grid can legitimately differ on both.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
from typing_extensions import Unpack

from geosave_engine.geodata.tile import GeoTile, mosaic
from geosave_engine.utils.colorize import Palette, colorize

_CATEGORICAL_MAX_CLASSES = 32  # beyond this, an integer band is probably not a label map

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
            f"Pass rgb_bands=(r_name, g_name, b_name) to plot() explicitly."
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
    if tile.num_bands >= 3:
        return _Panel(
            "rgb", tile, title, rgb_bands=_resolve_rgb_bands(tile, title, rgb_bands), polygon_alpha=polygon_alpha
        )
    if tile.num_bands == 1:
        if np.issubdtype(tile.data.dtype, np.floating):
            return _Panel("continuous", tile, title, cmap=cmap, polygon_alpha=polygon_alpha)
        n_unique = int(np.unique(tile.data.values).size)
        if n_unique <= _CATEGORICAL_MAX_CLASSES:
            return _Panel(
                "categorical", tile, title, class_map=class_map, color_map=color_map, polygon_alpha=polygon_alpha
            )
        return _Panel("continuous", tile, title, cmap=cmap, polygon_alpha=polygon_alpha)
    return _Panel("fallback", tile, title, cmap=cmap, polygon_alpha=polygon_alpha)  # 2 bands, no clean mapping


def _as_mpl_color(color: tuple[int, int, int] | str) -> tuple[float, float, float] | str:
    if isinstance(color, str):
        return color
    return tuple(c / 255 for c in color)


def _default_palette(classes: list[int]) -> dict[int, tuple[int, int, int]]:
    return {c: tuple(int(x * 255) for x in plt.cm.tab20(i % 20)[:3]) for i, c in enumerate(classes)}


def _stretch(channel: np.ndarray) -> np.ndarray:
    masked = np.ma.masked_invalid(channel)
    try:
        lo, hi = np.percentile(masked.compressed(), [2, 98])
    except ValueError:
        lo, hi = float(np.nanmin(channel)), float(np.nanmax(channel))
    return np.clip((channel - lo) / (hi - lo + 1e-8), 0, 1)


def _time_str(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _panel_caption(tile: GeoTile) -> str:
    """Per-panel time + place caption — each panel may genuinely differ on both."""
    time_str = _time_str(tile.start) if tile.start == tile.end else f"{_time_str(tile.start)} → {_time_str(tile.end)}"
    address = tile.location.get("address")
    geo_str = f"{address}  ·  {tile.coordinate_str}" if address else tile.coordinate_str
    return f"{time_str}\n{geo_str}"


def _draw_polygon(ax: plt.Axes, tile: GeoTile, alpha: float) -> None:
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
    inv = ~tile.affine
    px, py = zip(*(inv * (x, y) for x, y in poly.exterior.points))
    ax.plot(px, py, color="red", linewidth=1.5, alpha=alpha)


def _render(ax: plt.Axes, panel: _Panel) -> None:
    if panel.kind == "rgb":
        arr = panel.tile.to_numpy(bands=list(panel.rgb_bands)).astype("float32")
        rgb = np.stack([_stretch(arr[i]) for i in range(3)], axis=-1)
        ax.imshow(rgb)
    elif panel.kind in ("continuous", "fallback"):
        arr = np.ma.masked_invalid(panel.tile.to_numpy()[0])
        im = ax.imshow(arr, cmap=panel.cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:  # categorical
        arr = panel.tile.to_numpy()[0].astype(int)
        classes = sorted(np.unique(arr).tolist())
        palette: Palette = panel.color_map or _default_palette(classes)
        ax.imshow(colorize(arr, palette))
        labels = panel.class_map or {}
        handles = [Patch(color=_as_mpl_color(palette[c]), label=labels.get(c, str(c))) for c in classes]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    _draw_polygon(ax, panel.tile, panel.polygon_alpha)
    ax.set_title(panel.title, fontsize=10, fontweight="bold")
    caption = _panel_caption(panel.tile)
    if panel.kind == "fallback":
        caption = f"band 0 of {panel.tile.num_bands} — no RGB/label mapping\n{caption}"
    ax.set_xlabel(caption, fontsize=7, color="#666666", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])


def _group_key(tile: GeoTile) -> tuple[tuple[str, ...], str]:
    return (tile.bands, tile.start.date().isoformat())


def _adjacent_components(tiles: list[GeoTile]) -> list[list[GeoTile]]:
    """Split tiles into groups that actually touch/overlap — not just share bands+date.

    A tiling grid's adjacent tiles have different centroids by design, so
    centroid proximity can't tell "meant to mosaic" apart from "unrelated
    location that coincidentally shares bands and date." Real bbox
    adjacency can.
    """
    n = len(tiles)
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
            if tiles[i].crs == tiles[j].crs and tiles[i].bbox_polygon.intersects(tiles[j].bbox_polygon):
                union(i, j)

    components: dict[int, list[GeoTile]] = {}
    for i, t in enumerate(tiles):
        components.setdefault(find(i), []).append(t)
    return list(components.values())


def _panel_title(component: list[GeoTile]) -> str:
    description = component[0].metadata.get("description")
    if description:
        return description
    return f"{len(component)} tiles mosaicked" if len(component) > 1 else ""


class PlotKwargs(TypedDict, total=False):
    """Keyword args accepted by `plot()` — also what `GeoTile.plot()` forwards."""

    cmap: str
    class_map: dict[int, str] | None
    color_map: Palette | None
    rgb_bands: tuple[str, str, str] | None
    polygon_alpha: float
    cols: int | None
    title: str


def plot(tiles: GeoTile | Sequence[GeoTile], **kwargs: Unpack[PlotKwargs]) -> tuple[plt.Figure, np.ndarray]:
    """Plot one or more GeoTiles, auto-picking a renderer per tile.

    Grouping, one tile pair at a time — same rule regardless of how many
    tiles are passed:

    | Same date? | Same/adjacent location? | Result |
    | --- | --- | --- |
    | Yes | Yes (touch or overlap) | One mosaicked panel (via ``mosaic()``) |
    | Yes | No (unrelated location) | Two separate panels — same date alone isn't enough |
    | No | Yes (same/overlapping spot, different day) | Two separate panels — a time series, not a mosaic |
    | No | No | Two separate panels |

    Concretely: an ingested tiling grid (adjacent chips, same acquisition
    day) mosaics into one continuous image. A handful of anchors sampled
    from across a whole training set (different places, maybe different
    days) facets one panel per anchor. A single location revisited over
    time facets one panel per date. Two unrelated locations that happen to
    share an acquisition date do **not** silently merge into one panel just
    because the group key matches — adjacency is checked for real (bbox
    intersects), not guessed from date/bands alone.

    A single tile with its own time dimension is split into one panel per
    timestep first, then run through the same rule (so a multi-date single
    tile still facets by date exactly like a list of single-date tiles
    would). Each resulting panel carries its own time + place caption —
    there's no one global caption, since panels can legitimately differ on
    both.

    Args:
        tiles: One GeoTile, or several to mosaic/facet together.
        cmap: Colormap for single-band float (continuous) tiles. Default `"viridis"`.
        class_map: ``{value: name}`` for single-band integer (categorical)
            tiles — applies to every categorical panel in this call.
        color_map: ``{value: hex_or_rgb}`` for the same — auto-generated
            from a fixed palette if omitted.
        rgb_bands: ``(r_name, g_name, b_name)`` for tiles with more than 3
            bands — required in that case, since which 3 count as color is
            ambiguous from shape alone. Resolved per panel by name against
            that panel's own ``tile.bands``, so one shared value works even
            across several differently-ordered multiband tiles in one call.
            Ignored for exactly-3-band tiles (band order as stored).
        cols: Facet grid column count. Auto-sized (max 4) if omitted.
        polygon_alpha: Opacity of a panel's ``tile.polygon`` outline (the
            exact AOI footprint, when set — e.g. from ``from_polygon``/
            ``from_geojson``), ``0`` hides it. Default `0.8`. A tile with no
            polygon (built from a bbox/coordinate) draws nothing regardless.
        title: Optional figure suptitle.

    Returns:
        ``(Figure, ndarray of Axes)`` — same shape whether one panel or several.

    Raises:
        ValueError: ``tiles`` is empty; a tile has more than 3 bands and
            ``rgb_bands`` wasn't given; ``rgb_bands`` names a band a tile
            doesn't have; or adjacent tiles can't mosaic (mismatched CRS
            without reconciliation, etc — see ``mosaic()``).
    """
    cmap = kwargs.get("cmap", "viridis")
    class_map = kwargs.get("class_map")
    color_map = kwargs.get("color_map")
    rgb_bands = kwargs.get("rgb_bands")
    cols = kwargs.get("cols")
    polygon_alpha = kwargs.get("polygon_alpha", 0.8)
    title = kwargs.get("title", "")

    tile_list = [tiles] if isinstance(tiles, GeoTile) else list(tiles)
    if not tile_list:
        raise ValueError("plot() needs at least one GeoTile")

    if len(tile_list) == 1 and tile_list[0].has_time:
        base = tile_list[0]
        # with_data() alone would leave .datetime as the original (start != end)
        # range — every split-off tile would then collide into one bogus group.
        tile_list = [
            dataclasses.replace(base, data=base.data.isel(time=i), datetime=base.times[i])
            for i in range(len(base.times))
        ]

    groups: dict[tuple[tuple[str, ...], str], list[GeoTile]] = {}
    for t in tile_list:
        groups.setdefault(_group_key(t), []).append(t)

    components = [component for group in groups.values() for component in _adjacent_components(group)]

    panels = [
        _detect_panel(
            mosaic(component) if len(component) > 1 else component[0],
            title=_panel_title(component),
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
    for ax, panel in zip(axes_flat, panels):
        _render(ax, panel)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    if title:
        fig.suptitle(title, fontsize=11, y=0.99)
    return fig, axes


def fig_to_array(fig: plt.Figure) -> np.ndarray:
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

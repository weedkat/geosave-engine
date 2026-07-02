"""Geospatial EDA visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from geosave_engine.geodata.core import GeoTile
from geosave_engine.utils.geolocator import Place

# Colorbar dimensions in inches — must match make_axes_locatable args in ContinuousLayer.plot().
# plot_eda_grid uses these to pre-allocate extra column width so the image area stays at cell_w.
_CB_WIDTH: float = 0.05
_CB_PAD: float = 0.05


def _first_frame(array: np.ndarray) -> np.ndarray:
    """Drop a leading time axis for single-scene visualisation."""
    if array.ndim == 4:
        return array[0]
    return array


@dataclass
class RGBLayer:
    """Multi-band GeoTile rendered as a 2–98 percentile-stretched RGB composite.

    Args:
        title: Subplot title.
        geotile: GeoTile with shape (bands, H, W).
        bands: 0-based band indices for R, G, B. Defaults to (0, 1, 2).
    """

    title: str
    geotile: GeoTile
    bands: tuple[int, int, int] = (0, 1, 2)

    def plot(self, ax: plt.Axes) -> None:
        if self.geotile.data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        arr = _first_frame(self.geotile.to_numpy())

        if arr.ndim < 3 or arr.shape[0] <= max(self.bands):
            ax.text(0.5, 0.5, "RGB needs ≥3 bands", ha="center", va="center", transform=ax.transAxes)
            return

        rgb = arr[list(self.bands)].transpose(1, 2, 0).astype(np.float32)
        out = np.zeros_like(rgb)
        for i in range(3):
            channel = np.ma.masked_invalid(rgb[..., i])
            try:
                lo, hi = np.percentile(channel.compressed(), [2, 98])
            except ValueError:
                lo, hi = float(np.nanmin(channel)), float(np.nanmax(channel))
            out[..., i] = np.clip((rgb[..., i] - lo) / (hi - lo + 1e-8), 0, 1)

        ax.imshow(out, origin="upper")
        ax.set_title(self.title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])


@dataclass
class ContinuousLayer:
    """Single-band continuous GeoTile (e.g. NDVI, elevation).

    Args:
        title: Subplot title.
        geotile: GeoTile with shape (1, H, W) or (H, W).
        cmap: Matplotlib colormap name or object.
    """

    title: str
    geotile: GeoTile
    cmap: str = "RdYlGn"

    def plot(self, ax: plt.Axes) -> None:
        if self.geotile.data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        arr = np.ma.masked_invalid(_first_frame(self.geotile.to_numpy()))
        if arr.ndim == 3:
            arr = arr[0]
        img = ax.imshow(arr, cmap=self.cmap, origin="upper")

        if fig := ax.get_figure():
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=_CB_WIDTH, pad=_CB_PAD)
            cb = fig.colorbar(img, cax=cax)
            cb.ax.tick_params(labelsize=6)

        ax.set_title(self.title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])


@dataclass
class LabelLayer:
    """Discrete integer label GeoTile rendered with a caller-supplied color map.

    Args:
        title: Subplot title, also used as the legend group title.
        geotile: GeoTile with integer class values.
        color_map: ``{class_id: hex_color}`` for imshow rendering.
        class_map: ``{class_id: name}`` for legend labels. Falls back to str(id) if None.
    """

    title: str
    geotile: GeoTile
    color_map: dict[int, str] | dict[str, str] = field(default_factory=dict)
    class_map: dict[int, str] | dict[str, str] | None = None
    _cmap: ListedColormap = field(init=False, repr=False)
    _norm: BoundaryNorm = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = [int(v) for v in sorted(self.color_map)]
        colors = [self.color_map[v] for v in sorted(self.color_map)]
        self._cmap = ListedColormap(colors)
        boundaries = [v - 0.5 for v in values] + [values[-1] + 0.5]
        self._norm = BoundaryNorm(boundaries, self._cmap.N)

    def label(self, class_id: int | str) -> str:
        if self.class_map and class_id in self.class_map:
            return self.class_map[class_id]
        return str(class_id)

    def legend_handles(self) -> tuple[list[Patch], list[str]]:
        """Return handles and labels filtered to class IDs present in the data."""
        if self.geotile.data is None:
            return [], []
        unique = set(np.unique(_first_frame(self.geotile.to_numpy())).astype(int))
        handles = [Patch(facecolor=self.color_map[cid], edgecolor="black") for cid in sorted(self.color_map) if cid in unique]
        labels = [self.label(cid) for cid in sorted(self.color_map) if cid in unique]
        return handles, labels

    def plot(self, ax: plt.Axes) -> None:
        if self.geotile.data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        arr = np.ma.masked_invalid(_first_frame(self.geotile.to_numpy()).astype(float))
        if arr.ndim == 3:
            arr = arr[0]
        ax.imshow(arr, cmap=self._cmap, norm=self._norm, origin="upper")
        ax.set_title(self.title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])


def _cell_dims(layers: list[RGBLayer | ContinuousLayer | LabelLayer]) -> tuple[float, float]:
    """Return (cell_w, cell_h) in inches derived from the first layer with data."""
    dpi = plt.rcParams["figure.dpi"]
    for layer in layers:
        if layer.geotile.data is not None:
            data = _first_frame(layer.geotile.to_numpy())
            return data.shape[-1] / dpi, data.shape[-2] / dpi
    return 3.0, 3.0  # fallback when all layers have no data


def plot_eda_grid(
    layers: list[RGBLayer | ContinuousLayer | LabelLayer],
    title: str = "EDA Grid",
    cols: int | None = None,
    gap: float = 0.1,
    legend_width: float = 1.0,
    legend_gap: float = 0.03,
) -> tuple[plt.Figure, np.ndarray]:
    if not layers:
        raise ValueError("layers list cannot be empty")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Liberation Sans", "Noto Sans", "Helvetica", "Arial", "DejaVu Sans"]

    n = len(layers)
    cols = min(cols or n, n)
    rows = int(np.ceil(n / cols))

    label_layers = [ll for ll in layers if isinstance(ll, LabelLayer) and ll.legend_handles()[0]]

    cell_w, cell_h = _cell_dims(layers)

    has_cb = [
        any(isinstance(layers[r * cols + c], ContinuousLayer) for r in range(rows) if r * cols + c < n)
        for c in range(cols)
    ]
    col_widths = [cell_w + _CB_WIDTH + _CB_PAD if hc else cell_w for hc in has_cb]
    plot_width = sum(col_widths)
    fig_width = plot_width + (legend_width if label_layers else 0)
    right = plot_width / fig_width if label_layers else 1.0

    ref = layers[0].geotile
    lon, lat = ref.centroid
    coord_str = f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}"
    place = Place.from_coordinate(lat, lon)
    address = place.to_address() if place else None
    geo_label = f"{address}  ·  {coord_str}" if address else coord_str

    times = ref.times
    if times:
        t0 = str(times[0])[:10]
        date_str = f"{t0} → {str(times[-1])[:10]}" if len(times) > 1 else t0
    else:
        date_str = ref.datetime.strftime("%Y-%m-%d")
    geo_label = f"{date_str}  ·  {geo_label}"

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_width, rows * cell_h + 0.2),
        gridspec_kw={'wspace': gap, 'hspace': gap, 'width_ratios': col_widths},
    )
    axes_flat = np.array(axes).flatten()

    for layer, ax in zip(layers, axes_flat):
        layer.plot(ax)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    cx = (min(ax.get_position().x0 for ax in axes_flat[:n]) +
          max(ax.get_position().x1 for ax in axes_flat[:n])) / 2
    fig.suptitle(title, fontsize=12, weight='bold', x=cx, y=0.97)

    if label_layers:
        panel_cx = right + (legend_width / fig_width) / 2
        legends = []
        for ll in label_layers:
            handles, labels = ll.legend_handles()
            leg = fig.legend(
                handles, labels, title=ll.title,
                loc="upper center",
                bbox_to_anchor=(panel_cx, 1.0), bbox_transform=fig.transFigure,
                fontsize=9, frameon=False,
                title_fontproperties={"weight": "bold", "size": 9},
            )
            legends.append(leg)

        fig.canvas.draw()
        fig_h = fig.get_window_extent().height
        total_h = sum(leg.get_window_extent().height for leg in legends) / fig_h + legend_gap * (len(legends) - 1)
        y = 0.5 + total_h / 2
        for i, leg in enumerate(legends):
            leg.set_bbox_to_anchor((panel_cx, y), transform=fig.transFigure)
            y -= leg.get_window_extent().height / fig_h + (legend_gap if i < len(legends) - 1 else 0)

        fig.canvas.draw()
        fig_ext = fig.get_window_extent()
        exts = [leg.get_window_extent() for leg in legends]
        pad = 6
        bx0 = (min(e.x0 for e in exts) - pad) / fig_ext.width
        by0 = (min(e.y0 for e in exts) - pad) / fig_ext.height
        bw = (max(e.x1 for e in exts) + pad) / fig_ext.width - bx0
        bh = (max(e.y1 for e in exts) + pad) / fig_ext.height - by0
        fig.add_artist(Rectangle((bx0, by0), bw, bh, transform=fig.transFigure,
                                  fill=False, edgecolor="grey", linewidth=0.8, clip_on=False))

    fig.text(cx, 0.015, geo_label, ha="center", va="bottom", fontsize=9, color="#666666", style="italic")

    return fig, axes_flat

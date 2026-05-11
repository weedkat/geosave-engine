"""Geospatial EDA visualization utilities.

Provides composable layer descriptors for plotting xarray DataArrays in a
grid layout. Each descriptor controls its own render type and parameters —
no auto-detection, full caller control.

Example usage::

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from geosave_engine.utils.geovis import plot_eda_grid, ContinuousLayer, RGBLayer, LabelLayer

    # Define label colormap in the notebook (domain knowledge stays with caller)
    DW_LABELS = [
        (0, "No data", "#000000"),
        (1, "Water", "#419bdf"),
        (2, "Trees", "#397d49"),
    ]
    dw_cmap = ListedColormap([c for _, _, c in DW_LABELS])
    dw_norm = BoundaryNorm(range(len(DW_LABELS) + 1), dw_cmap.N)

    layers = [
        ContinuousLayer("ndvi", result["ndvi"].squeeze()),
        RGBLayer("image", result["image"].squeeze(), bands=[3, 2, 1]),
        LabelLayer("label", result["label"].squeeze(), label_entries=DW_LABELS, cmap=dw_cmap, norm=dw_norm),
    ]

    fig, axes = plot_eda_grid(layers, cols=3)
    plt.show()
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from geosave_engine.geodata.pipeline.io import ClassMeta, MaskMeta


def _to_entry(e: tuple[int, str, str] | ClassMeta | MaskMeta) -> tuple[int, str, str]:
    if not isinstance(e, tuple):
        return (e.id, e.name, e.color or "#000000")
    return cast(tuple[int, str, str], e)

# ---------------------------------------------------------------------------
# Geocoding (Nominatim via geopy)
# Cached by rounded center coordinate (~1 km grid) to avoid repeat requests.
# Graceful fallback if network unavailable or geopy not installed.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _reverse_geocode(lat: float, lon: float) -> str:
    """Return a human-readable place name for the given WGS84 coordinate.

    Results are cached per (lat, lon) rounded to 2 decimal places (~1 km).
    Returns empty string on failure — never raises.
    """
    try:
        from geopy.geocoders import Nominatim  # optional dep

        geolocator = Nominatim(user_agent="geosave-engine")
        location = geolocator.reverse((lat, lon), timeout=5, language="en")
        if location is None:
            return ""
        addr = location.raw.get("address", {})
        parts = [
            addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county"),
            addr.get("state") or addr.get("region"),
            addr.get("country"),
        ]
        return ", ".join(p for p in parts if p)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Extent helpers
# ---------------------------------------------------------------------------


def _get_extent(data: xr.DataArray) -> tuple[float, float, float, float]:
    """Extract (left, right, bottom, top) WGS84 extent from a DataArray.

    Reads attrs['bbox'] (set by Pipeline.run(), always WGS84) when available.
    """
    bbox = data.attrs.get("bbox")
    if bbox is not None:
        left, bottom, right, top = bbox  # rasterio: (left, bottom, right, top)
        return (left, right, bottom, top)  # matplotlib: (left, right, bottom, top)
    raise ValueError("DataArray missing 'bbox' attribute for extent")


def _format_extent(extent: tuple[float, float, float, float]) -> str:
    """Format extent as informative label with place name, center, and bounds.

    Expects WGS84 (left, right, bottom, top) — as returned by _get_extent() from attrs['bbox'].

    Example output::

        Bandung, West Java, Indonesia | center: 6.90°S, 107.62°E
    """
    if extent is None:
        return ""
    left, right, bottom, top = extent

    center_lat = (bottom + top) / 2
    center_lon = (left + right) / 2

    lat_str = f"{abs(center_lat):.2f}°{'N' if center_lat >= 0 else 'S'}"
    lon_str = f"{abs(center_lon):.2f}°{'E' if center_lon >= 0 else 'W'}"
    center = f"center: {lat_str}, {lon_str}"

    place = _reverse_geocode(round(center_lat, 2), round(center_lon, 2))
    if place:
        return f"{place} | {center}"
    return f"{center}"


# ---------------------------------------------------------------------------
# Shared axis setup
# ---------------------------------------------------------------------------


def _setup_ax(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Layer descriptors
# ---------------------------------------------------------------------------


@dataclass
class ContinuousLayer:
    """Single-band continuous value layer (e.g. NDVI, elevation).

    Args:
        title: Subplot title.
        data: 2-D DataArray (extra dims squeezed automatically).
        cmap: Matplotlib colormap name or object. Defaults to ``"RdYlGn"``.

    Example::

        ContinuousLayer("ndvi", result["ndvi"].squeeze())
        ContinuousLayer("elevation", dem, cmap="terrain")
    """

    title: str
    data: xr.DataArray
    cmap: str = "RdYlGn"

    def plot(self, ax: plt.Axes) -> None:
        arr = np.ma.masked_invalid(np.asarray(self.data.squeeze()))
        extent = _get_extent(self.data)
        image = ax.imshow(arr, cmap=self.cmap, extent=extent, origin="upper")
        _setup_ax(ax, self.title)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


@dataclass
class RGBLayer:
    """Multi-band layer rendered as RGB composite with 2–98 percentile stretch.

    Args:
        title: Subplot title.
        data: 3-D DataArray with shape (bands, height, width).
        bands: Band names (from 'band' coordinate) to composite as R, G, B.
               If None, uses first three bands in coordinate order.

    Example::

        RGBLayer("true color", result["image"].squeeze(), bands=["B4", "B3", "B2"])
    """

    title: str
    data: xr.DataArray
    bands: list[str] | None = None

    def plot(self, ax: plt.Axes) -> None:
        arr = np.asarray(self.data.squeeze())
        extent = _get_extent(self.data)
        _setup_ax(ax, self.title)

        # Resolve band names to integer indices
        band_indices: list[int]
        if "band" in self.data.coords:
            band_names_in_data = [str(b) for b in self.data["band"].values]
            if self.bands is None:
                band_indices = list(range(min(3, len(band_names_in_data))))
            else:
                try:
                    band_indices = [band_names_in_data.index(b) for b in self.bands]
                except ValueError as e:
                    ax.text(0.5, 0.5, f"Band not found: {e}", ha="center", va="center", transform=ax.transAxes)
                    return
        else:
            if self.bands is not None:
                ax.text(0.5, 0.5, "Band names require 'band' coordinate", ha="center", va="center", transform=ax.transAxes)
                return
            band_indices = [0, 1, 2]

        if arr.ndim < 3 or arr.shape[0] <= max(band_indices):
            ax.text(0.5, 0.5, "RGB needs ≥3 bands", ha="center", va="center", transform=ax.transAxes)
            return

        rgb = arr[band_indices].transpose(1, 2, 0).astype(np.float32)
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):
            band = np.ma.masked_invalid(rgb[..., i])
            try:
                vmin, vmax = np.percentile(band.compressed(), [2, 98])
            except ValueError:
                vmin, vmax = float(np.nanmin(band)), float(np.nanmax(band))
            rgb_norm[..., i] = np.clip((rgb[..., i] - vmin) / (vmax - vmin + 1e-8), 0, 1)

        ax.imshow(rgb_norm, extent=extent, origin="upper")


@dataclass
class LabelLayer:
    """Discrete label layer rendered with auto-generated colormap and legend.

    Builds colormap and norm from label_entries automatically, so caller
    only needs to provide the label definitions.

    Args:
        title: Subplot title.
        data: 2-D integer DataArray.
        label_entries: List of ``(value, name, hex_color)`` tuples defining
                       the class registry. Used to build colormap and legend.
        cmap: Optional ``ListedColormap``. Auto-generated from label_entries
              if not provided.
        norm: Optional ``BoundaryNorm``. Auto-generated from label_entries
              if not provided.

    Example::

        LABELS = [(0, "No data", "#000"), (1, "Water", "#419bdf"), ...]
        LabelLayer("labels", result["label"].squeeze(), label_entries=LABELS)

        # Or with custom colormap:
        custom_cmap = ListedColormap([c for _, _, c in LABELS])
        LabelLayer("labels", data, label_entries=LABELS, cmap=custom_cmap)
    """

    title: str
    data: xr.DataArray
    label_entries: list[tuple[int, str, str]] | list[ClassMeta] | list[MaskMeta]
    cmap: ListedColormap | None = None
    norm: BoundaryNorm | None = None

    def __post_init__(self) -> None:
        if self.cmap is None:
            colors = [_to_entry(e)[2] for e in self.label_entries]
            object.__setattr__(self, "cmap", ListedColormap(colors))
        if self.norm is None:
            boundaries = np.arange(-0.5, len(self.label_entries) + 0.5, 1)
            cmap = cast(ListedColormap, self.cmap)
            object.__setattr__(self, "norm", BoundaryNorm(boundaries, cmap.N))

    def plot(self, ax: plt.Axes) -> None:
        arr = np.asarray(self.data.squeeze())
        arr_masked = np.ma.masked_invalid(arr.astype(float))
        extent = _get_extent(self.data)

        ax.imshow(arr_masked, cmap=self.cmap, norm=self.norm, extent=extent, origin="upper")
        _setup_ax(ax, self.title)

        unique_vals = set(np.unique(arr[~np.isnan(arr.astype(float))]).astype(int))
        handles = [
            Patch(facecolor=color, edgecolor="black", label=f"{value} {name}")
            for value, name, color in (_to_entry(e) for e in self.label_entries)
            if value in unique_vals
        ]
        if handles:
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# Grid entry point
# ---------------------------------------------------------------------------


def plot_eda_grid(
    layers: list[ContinuousLayer | RGBLayer | LabelLayer],
    cols: int = 3,
    figsize_per_subplot: tuple[float, float] = (3.5, 2.8),
    dpi: int = 100,
) -> tuple[plt.Figure, np.ndarray]:
    """Render a grid of EDA subplots from a list of layer descriptors.

    The figure title shows the geographic location of the first layer's
    extent, resolved via Nominatim reverse geocoding (cached, with fallback).

    Args:
        layers: Ordered list of layer descriptors.
        cols: Columns in the grid; rows computed automatically.
        figsize_per_subplot: (width, height) in inches per cell.
        dpi: Figure resolution.

    Returns:
        ``(fig, axes)`` where axes is a 2-D numpy array of Axes objects.

    Raises:
        ValueError: If ``layers`` is empty.

    Example::

        layers = [
            ContinuousLayer("ndvi", result["ndvi"].squeeze()),
            RGBLayer("image", result["image"].squeeze(), bands=["B04", "B03", "B02"]),
            LabelLayer("label", result["label"].squeeze(),
                       label_entries=DW_LABELS, cmap=dw_cmap, norm=dw_norm),
        ]
        fig, axes = plot_eda_grid(layers, cols=3)
        plt.show()
    """
    if not layers:
        raise ValueError("layers list cannot be empty")

    n = len(layers)
    rows = int(np.ceil(n / cols))
    figsize = (cols * figsize_per_subplot[0], rows * figsize_per_subplot[1])

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    extent_label = _format_extent(_get_extent(layers[0].data))
    if extent_label:
        fig.subplots_adjust(bottom=0.15)
        fig.text(0.5, -0.05, extent_label, ha="center", va="center", fontsize=9, wrap=True)

    for layer, ax in zip(layers, axes.flat):
        layer.plot(ax)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    return fig, axes

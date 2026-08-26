"""GeoTile: one small window of pixels plus its own where/when. See GeoTile for details."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict, Unpack, overload

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
from numpy.typing import DTypeLike

from ._array import _SpatialArray
from .anchor import GeoAnchor
from .context import ModelContext, numpy_context, tensor_context, validate_context

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from geosave_engine.geodata.viz import Kind, RenderStyle, ViewOptions

    from .raster import GeoRaster


__all__ = ["GeoTile", "NumpyTile", "TensorTile"]


class NumpyTile(TypedDict):
    """Materialized single-raster NumPy item."""

    data: np.ndarray
    anchor: GeoAnchor
    model_context: dict[str, Any]


class TensorTile(TypedDict):
    """Materialized single-raster tensor item."""

    data: torch.Tensor
    anchor: GeoAnchor
    model_context: dict[str, Any]


@dataclass(frozen=True, kw_only=True, eq=False)
class GeoTile(_SpatialArray):
    """Raster pixel data plus its own where/when — small, consumable, whole-load-safe.

    Comes from `GeoRaster.tiles()`/`.to_tile()`, or a fresh fetch not yet
    promoted to a `GeoRaster`.

    Args:
        model_context: Precomputed model inputs for this window, normally
            `GeoRaster.tiles(context_fn=...)`'s own output. Kept as the
            encoder produced them. None leaves an encoder to derive its own
            from `anchor`.

    Raises:
        ValueError: A `model_context` value isn't array-like.
    """

    # pixels already fit in memory, so a view sends them rather than datashading
    RASTERIZE: ClassVar[bool] = False

    model_context: ModelContext | None = None

    def __post_init__(self) -> None:
        """Validate pixels against the anchor, then the model context."""
        super().__post_init__()
        if self.model_context is not None:
            # frozen dataclass, so the validated copy has to be written past the descriptor
            object.__setattr__(self, "model_context", validate_context(self.model_context))

    # --- Visualization ---

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
        """Draw this tile as one static panel.

        The bounded counterpart of `explore`: same element, rendered through
        holoviews' matplotlib backend, the one path needing no browser.

        Args:
            kind: Force a renderer instead of resolving one from `render`.
            style: Color policy. None takes the default.
            band: Draw this band alone. A tile carrying several bands needs
                one, since a static panel has no widget to pick with.
            time: Draw this timestamp alone, for the same reason.
            vector: True outlines this tile's own features over the pixels.
            **options: Per-view hvplot options — see `ViewOptions`. The
                title defaults to this tile's own time and place, and
                `width`/`height` are pixels.

        Returns:
            Matplotlib Figure.

        Raises:
            ImportError: The `viz` extra isn't installed.
            KeyError: `band` or `time` names something this tile doesn't carry.
            ValueError: Several bands or steps remain and none was named.

        Examples:
            >>> figure = tile.plot(band="B08", width=900)
        """
        from geosave_engine.geodata.viz import DEFAULT_STYLE, plot

        options.setdefault("title", self._caption())
        features = self.vector
        return plot(
            self.data,
            render=self.render,
            legend=self.legend,
            kind=kind,
            style=style if style is not None else DEFAULT_STYLE,
            band=band,
            time=time,
            vector=features.gdf if vector and features is not None else None,
            **options,
        )

    def _caption(self) -> str:
        """One-line "when and where" for this tile's own plot panel.

        Returns:
            Time range (or "timeless") followed by the WGS84 centroid.
        """
        lon, lat = self.geographic_centroid
        if self.start is None:
            when = "timeless"
        elif self.start == self.end:
            when = str(self.start)
        else:
            when = f"{self.start} – {self.end}"
        return f"{when}  |  {lat:.4f}, {lon:.4f}"

    # --- Materialization ---

    def to_numpy(
        self,
        bands: Sequence[str] | None = None,
        dtype: DTypeLike | None = None,
        progress: bool = False,
    ) -> np.ndarray:
        """Read pixels into a numpy array.

        Args:
            bands: Band names to keep, in this order. None keeps every band as-is.
            dtype: Numpy dtype to cast to. None keeps the stored dtype.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            Array shaped `(band, y, x)`, or `(time, band, y, x)` if this
            tile has a time dim.

        Raises:
            KeyError: a name in `bands` isn't one of this tile's bands.
        """
        from geosave_engine.geodata.utils.array import progress_bar

        da = self.data if bands is None else self._select_bands(bands)
        with progress_bar(progress):
            array = np.asarray(da.values)
        return array if dtype is None else array.astype(dtype)

    def to_tensor(
        self,
        bands: Sequence[str] | None = None,
        dtype: torch.dtype | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Read pixels into a torch tensor — same shape rules as `to_numpy`.

        Args:
            bands: Band names to keep, in this order. None keeps every band as-is.
            dtype: Torch dtype to cast to. None keeps the stored dtype.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            Tensor shaped `(band, y, x)`, or `(time, band, y, x)`.

        Raises:
            KeyError: a name in `bands` isn't one of this tile's bands.
        """
        import torch

        tensor = torch.from_numpy(self.to_numpy(bands, progress=progress))
        return tensor if dtype is None else tensor.to(dtype)

    @overload
    def to_sample(
        self,
        *,
        bands: Sequence[str] | None = ...,
        dtype: torch.dtype | None = ...,
        mode: Literal["tensor"] = ...,
        progress: bool = ...,
    ) -> TensorTile: ...

    @overload
    def to_sample(
        self,
        *,
        bands: Sequence[str] | None = ...,
        dtype: DTypeLike | None = ...,
        mode: Literal["numpy"],
        progress: bool = ...,
    ) -> NumpyTile: ...

    def to_sample(
        self,
        *,
        bands: Sequence[str] | None = None,
        dtype: Any = None,
        mode: Literal["tensor", "numpy"] = "tensor",
        progress: bool = False,
    ) -> TensorTile | NumpyTile:
        """Read this window into one sample — the dict a Dataset serves per item.

        Args:
            bands: Band names to keep, in this order. None keeps every band as-is.
            dtype: Dtype to cast pixels to — a torch dtype under
                `mode="tensor"`, a numpy one under `mode="numpy"`. None keeps
                the stored dtype.
            mode: Runtime to render for. `"tensor"` is what a DataLoader wants.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            {
                "data": torch.Tensor | np.ndarray,  # (band, y, x) or (time, band, y, x)
                "anchor": GeoAnchor,  # this window's own grid, with its vector
                "model_context": {
                    "<key>": torch.Tensor | np.ndarray | str | None,
                },
            }

        Raises:
            KeyError: a name in `bands` isn't one of this tile's bands.

        Examples:
            >>> tile.to_sample()["data"].shape
            torch.Size([2, 256, 256])
        """
        if mode == "numpy":
            return {
                "data": self.to_numpy(bands, dtype, progress),
                "anchor": self.anchor,
                "model_context": numpy_context(self.model_context),
            }
        return {
            "data": self.to_tensor(bands, dtype, progress),
            "anchor": self.anchor,
            "model_context": tensor_context(self.model_context),
        }

    def to_raster(self) -> GeoRaster:
        """Promote this window to a GeoRaster — same pixels, same header, same vector.

        The disk boundary lives on `GeoRaster`: a file holds a surface, a
        tile is a window into one. Use this to write anything but a
        GeoTIFF, and `GeoRaster.open` to read.

        Returns:
            GeoRaster over the same window. `model_context` is dropped —
            it describes one bounded window, not a surface.
        """
        from .raster import GeoRaster

        return GeoRaster(data=self.data, anchor=self.anchor)

    # --- Persistence ---

    def to_geotiff(
        self,
        path: str | Path,
        time: dt | np.datetime64 | str | None = None,
        chunk_px: int | None = 512,
        progress: bool = True,
    ) -> Path:
        """Write this tile as one GeoTIFF.

        Attrs become flat GDAL tags; CF-only `flag_*` attrs are dropped.
        A `vector` is written beside it as `<stem>.vector.parquet`. Reading
        it back gives a `GeoRaster` — a file holds a surface, not a window.

        Args:
            path: Output `.tif` path.
            time: Which step to write, for a tile with a `time` dim — one
                of its own `times`, as a datetime or a string xarray can
                match (`"2024-01-15"`). Must name exactly one step. None is
                only valid for a tile with no `time` dim.
            chunk_px: On-disk block side length. None leaves GTiff untiled.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            The written path.

        Raises:
            ValueError: this tile has a `time` dim and `time` wasn't given,
                `time` was given for a tile with no `time` dim, or `time`
                names no step or more than one.
        """
        return self.to_raster().to_geotiff(path, time, chunk_px=chunk_px, progress=progress)

    def to_cog(
        self,
        path: str | Path,
        time: dt | np.datetime64 | str | None = None,
        chunk_px: int | None = 512,
        progress: bool = True,
    ) -> Path:
        """Write this tile as one cloud-optimized GeoTIFF — same contract as `to_geotiff`.

        Args:
            path: Output `.tif` path.
            time: Which step to write, for a tile with a `time` dim — one
                of its own `times`, as a datetime or a string xarray can
                match (`"2024-01-15"`). Must name exactly one step. None is
                only valid for a tile with no `time` dim.
            chunk_px: On-disk block side length. None leaves the COG
                driver's own default alone.
            progress: Show a dask progress bar while pixels compute.

        Returns:
            The written path.

        Raises:
            ValueError: this tile has a `time` dim and `time` wasn't given,
                `time` was given for a tile with no `time` dim, or `time`
                names no step or more than one.
        """
        return self.to_raster().to_cog(path, time, chunk_px=chunk_px, progress=progress)

"""Spatial and attrs properties shared by every `gs` accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from numpy.typing import DTypeLike
    from odc.geo import CRS, BoundingBox, Resolution

    from geosave_engine.geodata.attrs import AttrsHeader


def tensor(
    pixels: Callable[[DTypeLike], np.ndarray], dtype: torch.dtype | None
) -> torch.Tensor:
    """Build a model-input tensor from pixels read in a torch-compatible dtype.

    Args:
        pixels: Reads the pixels in the numpy dtype it is given.
        dtype: Tensor dtype, which the pixels are also read in. None casts to
            `torch.float32`, which keeps unsigned imagery off `torch.uint16` —
            a dtype torch accepts and carries no arithmetic kernels for. A
            dtype numpy cannot hold, such as `torch.bfloat16`, is read as
            float32 and narrowed on the way out.

    Returns:
        Tensor over contiguous pixels, in `dtype`.
    """
    import torch

    target = dtype or torch.float32
    try:
        reading_dtype = torch.empty(0, dtype=target).numpy().dtype
    except TypeError:
        reading_dtype = np.dtype(np.float32)
    return torch.as_tensor(np.ascontiguousarray(pixels(reading_dtype)), dtype=target)


class GeoAccessor[DataT: xr.Dataset | xr.DataArray | xr.DataTree]:
    """Spatial and attrs properties every `gs` accessor reads the same way.

    Concrete accessors bind their xarray object to `_data`; this class is
    never built directly.
    """

    _data: DataT

    @property
    def _grid_source(self) -> xr.Dataset | xr.DataArray:
        """Return the object odc-geo reads a geobox off directly.

        A DataTree carries no `.odc` accessor of its own, so its root
        Dataset stands in for it.

        Returns:
            `_data` itself, or its root Dataset for a DataTree.
        """
        return self._data.dataset if isinstance(self._data, xr.DataTree) else self._data

    @property
    def geobox(self) -> GeoBox:
        """Read the pixel grid, which places this object on the ground.

        Every ground-referenced member reads the grid through here, so an
        object carrying no CRS refuses them all rather than answering in
        pixel coordinates that mean nothing.

        Returns:
            Grid including CRS, transform, bounds, and shape.

        Raises:
            ValueError: The object carries no CRS, carries no spatial dims,
                or describes its grid by ground control points rather than a
                transform.
        """
        kind = type(self._data).__name__
        grid = self._grid_source.odc.geobox
        if grid is None:
            raise ValueError(
                f"{kind} carries no locatable grid; it has no CRS or spatial dims"
            )
        if not isinstance(grid, GeoBox):
            raise ValueError(
                f"{kind} grid is a {type(grid).__name__}; expected a regular GeoBox"
            )
        if grid.crs is None:
            raise ValueError(
                f"{kind} grid carries no CRS, so it is indexed in pixels rather "
                f"than placed on the ground; assign one with odc.geo.xr.assign_crs "
                f"before asking where it is"
            )
        return grid

    @property
    def is_georeferenced(self) -> bool:
        """Whether this object carries a locatable grid.

        Returns:
            True when `geobox` resolves; False where it would raise.
        """
        try:
            self.geobox
        except ValueError:
            return False
        return True

    @property
    def crs(self) -> CRS:
        """Read the coordinate reference system.

        Returns:
            CRS the grid mapping coordinate names.

        Raises:
            ValueError: The object carries no CRS.
        """
        grid = self.geobox
        assert grid.crs is not None  # geobox already refused a CRS-less grid
        return grid.crs

    @property
    def bounds(self) -> BoundingBox:
        """Read the grid's extent in its own CRS.

        Returns:
            Bounding box covering every pixel.

        Raises:
            ValueError: The object carries no locatable grid.
        """
        return self.geobox.boundingbox

    @property
    def resolution(self) -> Resolution:
        """Read the pixel size in CRS units.

        Returns:
            Signed resolution along x and y.

        Raises:
            ValueError: The object carries no locatable grid.
        """
        return self.geobox.resolution

    @property
    def attrs(self) -> AttrsHeader:
        """Read the typed attrs this object carries.

        Returns:
            Detached header, reread on every access because xarray attrs are
            mutable in place.
        """
        return attrs.read(self._data)

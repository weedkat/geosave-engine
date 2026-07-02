from __future__ import annotations

from typing import Literal

import pystac
import xarray as xr
from typing_extensions import Self

from geosave_engine.utils.geodata import extract_raster_scale_offset
from geosave_engine.utils.stac_query import CQL2

from .base import Source


class Sentinel2Source(Source):
    """Source with Sentinel-2 specific query filter helpers.

    Filter methods are sugar over ``with_filter(CQL2.*(...))``.
    Only valid for collections that carry the corresponding STAC properties
    (``eo:cloud_cover``, ``view:sun_elevation``, etc.).
    """

    def preprocess(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        """Apply radiometric scaling using per-item scale/offset from STAC metadata."""
        scale, offset = extract_raster_scale_offset(items[0])
        return ds * scale + offset

    def max_cloud_cover(self, max_cover: float) -> Self:
        """Filter scenes with cloud cover above ``max_cover`` percent.

        Args:
            max_cover: Maximum allowed cloud cover (0–100).
        """
        return self.with_filter(CQL2.lte("eo:cloud_cover", max_cover))

    def max_snow_cover(self, max_cover: float) -> Self:
        """Filter scenes with snow cover above ``max_cover`` percent.

        Args:
            max_cover: Maximum allowed snow cover (0–100).
        """
        return self.with_filter(CQL2.lte("eo:snow_cover", max_cover))

    def min_sun_elevation(self, min_elev: float) -> Self:
        """Filter scenes with sun elevation below ``min_elev`` degrees.

        Args:
            min_elev: Minimum sun elevation angle in degrees. Low values → long shadows.
        """
        return self.with_filter(CQL2.gte("view:sun_elevation", min_elev))

    def platform(self, name: Literal["sentinel-2a", "sentinel-2b"]) -> Self:
        """Filter to a specific Sentinel-2 platform.

        Args:
            name: ``'sentinel-2a'`` or ``'sentinel-2b'``.
        """
        return self.with_filter(CQL2.eq("platform", name))

    def relative_orbit(self, orbit: int) -> Self:
        """Filter by relative orbit number for consistent viewing geometry.

        Args:
            orbit: Relative orbit number (1–143 for Sentinel-2).
        """
        return self.with_filter(CQL2.eq("sat:relative_orbit", orbit))

    def sortby_cloud_cover(self, direction: str = "asc") -> Self:
        """Sort results by cloud cover to prioritize clearer scenes.

        Args:
            direction: ``'asc'`` (clearest first) or ``'desc'``.
        """
        self.query.with_sortby("eo:cloud_cover", direction)
        return self

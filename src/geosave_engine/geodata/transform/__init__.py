"""Explicit transformations of a raster Dataset.

Each transform runs the underlying xarray or odc operation, then rebases the
attrs its result earns. Nothing is aligned, promoted, reprojected, or
resampled implicitly.
"""

from . import grid, tiles, time, variables

__all__ = [
    "grid",
    "tiles",
    "time",
    "variables",
]

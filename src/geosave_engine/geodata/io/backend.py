"""Let xarray open GeoSave rasters and stacks by the engine name `geosave`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import xarray as xr
from xarray.backends import BackendEntrypoint

from .readers import read_raster, read_stack


class GeoSaveBackendEntrypoint(BackendEntrypoint):
    """Open rasters with `read_raster` and stacks with `read_stack`.

    The engine runs only when named. It claims no file on its own, so
    `xr.open_dataset("scene.tif")` keeps choosing among xarray's own engines.

    Examples:
        >>> xr.open_dataset("scene.zarr", engine="geosave").gs.variables
        ('red', 'nir')
        >>> xr.open_datatree("prepared/s1", engine="geosave").gs.groups
        ('label', 'sentinel_2_l2a')
    """

    description = "GeoSave rasters and raster stacks (COG, Zarr, NetCDF)"
    url = "https://github.com/weedkat/geosave-engine"
    open_dataset_parameters = ("filename_or_obj", "drop_variables")

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **options: Any,
    ) -> xr.Dataset:
        """Open what `read_raster` reads.

        Args:
            filename_or_obj: One raster file or store, a folder of COGs, or
                several files holding one raster.
            drop_variables: Variable name or names to leave out.
            **options: Read options the format's reader accepts, as
                `mask_and_scale=True`.

        Returns:
            Raster Dataset, lazily indexed; xarray applies `chunks` itself.
        """
        raster = read_raster(filename_or_obj, **options)
        if drop_variables is None:
            return raster
        return raster.drop_vars(drop_variables)

    def open_datatree(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        **options: Any,
    ) -> xr.DataTree:
        """Open what `read_stack` reads.

        Args:
            filename_or_obj: Folder holding one raster per group, or group
                names mapped to what `read_raster` reads.
            drop_variables: Unused; drop variables from the groups instead.
            **options: Read options passed to every group's reader.

        Returns:
            DataTree holding one group per raster.
        """
        return read_stack(filename_or_obj, **options)

"""Build raster stacks and read them through the `gs` DataTree accessor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Unpack, cast

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

from .raster import GeoRaster

# odc names the grid mapping variable, and every group shares the one root holds.
_GRID_MAPPING_COORDINATE = "spatial_ref"

if TYPE_CHECKING:
    from collections.abc import Sequence
    from dask.delayed import Delayed
    from os import PathLike
    import holoviews as hv
    import numpy as np
    import torch
    from numpy.typing import DTypeLike

    from geosave_engine.geodata.utils.io.netcdf import (
        NetCDFEngine,
        NetCDFWriteOptions,
    )
    from geosave_engine.geodata.attrs import TilingMode
    from geosave_engine.geodata import DataTree
    from geosave_engine.geodata.utils.io.zarr import ZarrWriteOptions


def stack(rasters: Mapping[str, xr.Dataset]) -> DataTree:
    """Build one raster stack from named rasters sharing an exact grid.

    Every raster becomes a group under a root that carries the shared grid, so
    xarray refuses a later group that does not align with it. A group carrying
    `time` is held to the same standard — resample or broadcast a mismatched one first.

    Args:
        rasters: Group name mapped to a raster Dataset. Names become on-disk
            group names, so they are supplied rather than inferred.

    Returns:
        Flat DataTree whose root carries the shared grid (and shared time
        axis, when any group carries one) and whose children are the
        supplied rasters, unchanged.

    Raises:
        ValueError: `rasters` is empty, a raster carries no locatable grid,
            the rasters do not share one exact GeoBox, or a raster's time
            axis disagrees with another's.

    Examples:
        >>> stack({"sentinel-2-l2a": optical, "dem": dem}).gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    if not rasters:
        raise ValueError("a raster stack needs at least one named raster")

    grids = {name: GeoRaster(raster).geobox for name, raster in rasters.items()}
    reference_name, reference_grid = next(iter(grids.items()))
    mismatched = sorted(name for name, grid in grids.items() if grid != reference_grid)
    if mismatched:
        raise ValueError(
            f"rasters {mismatched} are on a different grid from "
            f"{reference_name!r}; a stack shares one exact GeoBox, so align or resample "
            f"them onto it first"
        )

    anchor = rasters[reference_name]
    root_coords = {
        name: anchor.coords[name]
        for name in (*reference_grid.dimensions, _GRID_MAPPING_COORDINATE)
    }

    # A raster with no 'time' coordinate has nothing to disagree with, so it is exempt.
    carries_time = {
        name: rasters[name].coords["time"]
        for name in rasters
        if "time" in rasters[name].coords
    }
    if carries_time:
        reference_time_name, reference_time = next(iter(carries_time.items()))
        time_mismatched = sorted(
            name
            for name, coord in carries_time.items()
            if not coord.equals(reference_time)
        )
        if time_mismatched:
            raise ValueError(
                f"rasters {time_mismatched} carry a different time axis from "
                f"{reference_time_name!r}; resample or broadcast them onto one "
                f"shared axis first"
            )

        root_coords["time"] = reference_time
        # The reference may be a broadcast() result with no time_bnds of its own.
        bounded = next(
            (
                rasters[name]
                for name in carries_time
                if "time_bnds" in rasters[name].coords
            ),
            None,
        )
        if bounded is not None:
            root_coords["time_bnds"] = bounded.coords["time_bnds"]

    root = xr.Dataset(coords=root_coords)
    return cast(
        "DataTree",
        xr.DataTree.from_dict(
            {"/": root, **{f"/{name}": raster for name, raster in rasters.items()}}
        ),
    )


@xr.register_datatree_accessor("gs")
class GeoStack:
    """Read and persist one raster-stack DataTree.

    Args:
        data: Flat DataTree whose direct children are raster Datasets on one
            exact GeoBox.
    """

    def __init__(self, data: xr.DataTree) -> None:
        """Bind the DataTree.

        Args:
            data: DataTree to read through this accessor.
        """
        self._data: DataTree = cast("DataTree", data)

    @property
    def groups(self) -> tuple[str, ...]:
        """Return raster group names in stack order.

        Returns:
            Direct child group names.
        """
        return tuple(self._data.children)

    @property
    def geobox(self) -> GeoBox:
        """Read the exact grid every raster group shares.

        Returns:
            Grid carried by the stack root.

        Raises:
            ValueError: The root carries no locatable grid, so this DataTree
                is not a raster stack.
        """
        return GeoRaster(self._data.dataset).geobox

    def plot(self, *, cols: int = 1) -> hv.Layout:
        """Draw every group down the page, composed with `+`. Needs the `viz` extra.

        Each group draws through `Dataset.gs.plot`, so it keeps its own kind,
        captioned with the group name, and the panels stack into one column. A
        group spanning `time` contributes one panel per step.

        Args:
            cols: Panels per row. Defaults to one, stacking them in a column.

        Returns:
            Layout of every panel, or the sole panel when the stack holds one
            group with no time axis.

        Raises:
            ValueError: A group names nothing to draw; declare `RenderHints`
                on it or plot it alone with
                `stack["<group>"].to_dataset().gs.plot()`.

        Examples:
            >>> hv.save(scene.gs.plot(), "scene.png")
        """
        import holoviews as hv

        # A time facet is an NdLayout; mpl won't nest it, so take its panels.
        panels: list[hv.Element] = []
        for name in self.groups:
            drawn = self._data[name].to_dataset().gs.plot(title=name)
            panels.extend(drawn.values() if isinstance(drawn, hv.NdLayout) else [drawn])

        return hv.Layout(panels).cols(cols)

    def insert(self, name: str, raster: xr.Dataset) -> DataTree:
        """Add one raster to this stack under a new group name.

        Args:
            name: Group name, absent from this stack.
            raster: Raster Dataset on the stack's exact grid.

        Returns:
            New DataTree carrying the existing groups and `name`.

        Raises:
            ValueError: `name` is already a group, `raster` carries no
                locatable grid, or its grid differs from the stack's.

        Examples:
            >>> stack.gs.insert("ndvi", ndvi).gs.groups
            ('sentinel-2-l2a', 'dem', 'ndvi')
        """
        if name in self._data.children:
            raise ValueError(
                f"group {name!r} is already in this stack; drop it first or "
                f"choose another name"
            )
        grid = GeoRaster(raster).geobox
        stack_grid = self.geobox
        if grid != stack_grid:
            raise ValueError(
                f"raster {name!r} is on {grid} but the stack is on "
                f"{stack_grid}; align or resample it onto the stack grid first"
            )
        grown = self._data.copy()
        grown[name] = xr.DataTree(raster)
        return grown

    def tile(
        self,
        shape: tuple[int, int],
        *,
        group_id: str,
        overlap: int | float | tuple[int, int] = 0,
        mode: TilingMode = "reflect",
    ) -> list[DataTree]:
        """Cut every group of this stack on one shared layout.

        This stack holds one exact grid, so each group is cut at the same
        positions and a tile pairs the groups covering the same ground.

        Args:
            shape: Tile height and width in pixels.
            group_id: Identifier every tile of this operation shares, used to
                route tiles back to this stack when stitching.
            overlap: Shared pixels as a count, a fraction in `[0, 1)`, or
                row-column counts.
            mode: How the trailing edges are padded. `"constant"` extends every
                variable with its own declared fill value.

        Returns:
            Stacks in row-major order, each holding every group cut to one tile
            and stamped with the shared `Tiling`.

        Raises:
            ValueError: This stack holds no group, a group carries no locatable
                grid or sits on a different one, `shape` exceeds the stack's own
                shape, or `mode` is `"constant"` while a variable declares no
                fill value.

        Examples:
            >>> tiles = scene.gs.tile((256, 256), group_id="scene-001")
            >>> tiles[3].gs.groups
            ('sentinel-2-l2a', 'dem')
        """
        from geosave_engine.geodata.transform.tiles import tile_stack

        return cast(
            "list[DataTree]",
            tile_stack(
                self._data, shape, group_id=group_id, overlap=overlap, mode=mode
            ),
        )

    def time_window(
        self,
        slot: Mapping[str, int],
        *,
        stride: int,
    ) -> list[DataTree]:
        """Cut every group of this stack into fixed-length windows along time.

        Every group already shares one exact time axis, so the walk advances
        by one shared `stride` and each group contributes its own `slot`
        length of context around the same calendar position.

        Args:
            slot: Window length in buckets, keyed by group name.
            stride: Buckets between consecutive window starts, shared across
                every group.

        Returns:
            Windowed stacks in walk order, each holding every group cut to
            its own `slot` length around the same calendar position.

        Raises:
            ValueError: This stack holds no group, a group's name is missing
                from `slot`, a group's time axis is not calendar-contiguous,
                or a `slot` value exceeds the shared axis length.

        Examples:
            >>> windows = scene.gs.time_window({"sentinel-2-l2a": 3, "dem": 1}, stride=1)
            >>> windows[0].gs.groups
            ('sentinel-2-l2a', 'dem')
        """
        from geosave_engine.geodata.transform.time import time_window_stack

        return cast(
            "list[DataTree]", time_window_stack(self._data, slot, stride=stride)
        )

    def merge(self, other: xr.DataTree) -> DataTree:
        """Combine this stack with another on the same grid.

        Args:
            other: Raster stack sharing this stack's exact grid and sharing no
                group name with it.

        Returns:
            New DataTree carrying both stacks' groups.

        Raises:
            ValueError: The stacks share a group name or sit on different
                grids.

        Examples:
            >>> optical_stack.gs.merge(label_stack).gs.groups
            ('sentinel-2-l2a', 'dem', 'labels')
        """
        collisions = sorted(set(self.groups) & set(other.gs.groups))
        if collisions:
            raise ValueError(
                f"both stacks carry the groups {collisions}; rename one side "
                f"before merging"
            )
        other_grid = other.gs.geobox
        stack_grid = self.geobox
        if other_grid != stack_grid:
            raise ValueError(
                f"the other stack is on {other_grid} but this one is on "
                f"{stack_grid}; align or resample them onto the same grid first"
            )
        merged = self._data.copy()
        for name, node in other.children.items():
            merged[name] = node
        return merged

    def to_numpy(
        self,
        variables: Mapping[str, Sequence[str] | None] | None = None,
        *,
        dtype: DTypeLike | Mapping[str, DTypeLike] | None = None,
    ) -> dict[str, np.ndarray]:
        """Stack each group's variables into one model-input array.

        Groups share a grid but not their variables, non-spatial axes, or
        dtypes, so each one keeps its own array rather than being joined.

        Args:
            variables: Groups to read, mapped to their variables in the order
                the model expects. A group is dropped by leaving it out, and
                reads every variable in Dataset order when mapped to None.
                None reads every group.
            dtype: Cast applied to every group's array, or one cast per group.
                A group left out of a mapping keeps its source dtype, as does
                every group when None.

        Returns:
            {
                group name: array shaped `(*axes, band, y, x)`,
            }

        Raises:
            KeyError: A named group or variable is absent.
            ValueError: A group's selected variables carry different
                non-spatial dimensions, or they differ in dtype while no cast
                covers that group.

        Examples:
            >>> {name: array.shape for name, array in scene.gs.to_numpy().items()}
            {'sentinel-2-l2a': (2, 4, 256, 256), 'dem': (1, 256, 256)}
        """
        selected = self._select(variables)
        arrays: dict[str, np.ndarray] = {}
        for name, names in selected.items():
            group = self._data[name].to_dataset()
            arrays[name] = group.gs.to_numpy(names, dtype=_cast_for(dtype, name))
        return arrays

    def to_tensor(
        self,
        variables: Mapping[str, Sequence[str] | None] | None = None,
        *,
        dtype: torch.dtype | Mapping[str, torch.dtype] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stack each group's variables into one model-input tensor.

        Args:
            variables: Groups to read, mapped to their variables in the order
                the model expects. A group is dropped by leaving it out, and
                reads every variable in Dataset order when mapped to None.
                None reads every group.
            dtype: Tensor dtype for every group, or one dtype per group. A
                group no mapping covers casts to `torch.float32`, as does
                every group when None.

        Returns:
            {
                group name: tensor shaped `(*axes, band, y, x)`,
            }

        Raises:
            KeyError: A named group or variable is absent.
            ValueError: A group's selected variables carry different
                non-spatial dimensions.

        Examples:
            >>> batch = scene.gs.to_tensor(dtype={"label": torch.int64})
            >>> batch["label"].dtype
            torch.int64
        """
        selected = self._select(variables)
        tensors: dict[str, torch.Tensor] = {}
        for name, names in selected.items():
            group = self._data[name].to_dataset()
            tensors[name] = group.gs.to_tensor(names, dtype=_cast_for(dtype, name))
        return tensors

    def _select(
        self, variables: Mapping[str, Sequence[str] | None] | None
    ) -> dict[str, Sequence[str] | None]:
        """Resolve which groups to read and which variables each contributes.

        Args:
            variables: Groups mapped to their variables, or None for every
                group and every variable.

        Returns:
            {
                group name: variables to read, or None for all of them,
            }
            in this stack's own group order.

        Raises:
            KeyError: A named group is absent.
        """
        groups = self.groups
        if variables is None:
            return dict.fromkeys(groups)

        absent = sorted(set(variables) - set(groups))
        if absent:
            raise KeyError(
                f"{absent} are not groups of this stack; it carries {groups}"
            )

        selected: dict[str, Sequence[str] | None] = {}
        for name in groups:
            if name in variables:
                selected[name] = variables[name]
        return selected

    def to_zarr(
        self,
        destination: str | PathLike[str],
        *,
        overwrite: bool = False,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> Path | Delayed:
        """Write this raster stack to Zarr.

        Args:
            destination: Output path ending in `.zarr`.
            overwrite: Replace an existing destination when true.
            **write_options: Supported xarray Zarr write options.

        Returns:
            Destination path, or xarray's delayed write when `compute=False`.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.
        """
        from geosave_engine.geodata.utils.io import zarr

        return zarr.write(
            self._data,
            destination,
            overwrite=overwrite,
            **write_options,
        )

    def to_netcdf(
        self,
        destination: str | PathLike[str],
        *,
        engine: NetCDFEngine = "netcdf4",
        overwrite: bool = False,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> Path | Delayed:
        """Write this raster stack to netCDF.

        Args:
            destination: Output path ending in `.nc`, `.nc4`, or `.cdf`.
            engine: Concrete xarray netCDF writing engine.
            overwrite: Replace an existing destination when true.
            **write_options: Supported xarray netCDF write options.

        Returns:
            Destination path, or xarray's delayed write when `compute=False`.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.
        """
        from geosave_engine.geodata.utils.io import netcdf

        return netcdf.write(
            self._data,
            destination,
            engine=engine,
            overwrite=overwrite,
            **write_options,
        )


def _cast_for[T](dtype: T | Mapping[str, T] | None, group: str) -> T | None:
    """Read the cast one group is given.

    Args:
        dtype: One cast covering every group, casts named per group, or None.
        group: Group being read.

    Returns:
        Cast to apply to that group, or None when nothing covers it.
    """
    if isinstance(dtype, Mapping):
        return dtype.get(group)
    return dtype

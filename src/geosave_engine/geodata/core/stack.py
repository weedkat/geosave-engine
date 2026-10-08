"""Build raster stacks and read them through the `gs` DataTree accessor.

A stack is a flat DataTree holding one raster per group, a group being as many
variables as one grid can hold. What the root carries says whether the groups
are co-registered, and `GeoStack.align` is what puts them there.

Examples:
    Co-registered, so the grid sits at the root and the groups inherit it::

        >>> scene = stack({"sentinel-2-l2a": optical, "dem": dem})
        /                       y, x, spatial_ref   shared by both groups
        ├── /sentinel-2-l2a     red, nir            (time, y, x)
        └── /dem                elevation           (y, x)

    Native resolutions, so the root holds nothing and each group keeps its
    own grid::

        >>> sentinel = stack({"r10": bands_10m, "r20": bands_20m})
        >>> sentinel.gs.geobox is None
        True
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Unpack, cast, overload

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import warp

from .base import GeoRasterAccessor
from geosave_engine.geodata.conventions import CRS_COORDINATE, SPATIAL_DIMENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from dask.delayed import Delayed
    from os import PathLike
    import holoviews as hv
    import numpy as np
    from numpy.typing import DTypeLike

    from geosave_engine.geodata.io.raster.netcdf import (
        NetCDFEngine,
        NetCDFWriteOptions,
    )
    from geosave_engine.geodata.transform.warp import Resampling
    from geosave_engine.geodata.utils.datetime import DateRange

    from odc.geo import SomeResolution
    from geosave_engine.geodata import DataTree, Dataset
    from geosave_engine.geodata.io.raster.geotiff import COGWriteOptions
    from geosave_engine.geodata.io.raster.zarr import ZarrWriteOptions
    from geosave_engine.geodata.io.storage import StorageOptions

    from .anchor import GeoAnchor


def stack(rasters: Mapping[str, xr.Dataset]) -> DataTree:
    """Build one raster stack from named rasters.

    A group holds as many variables as one grid can, so a product delivering
    several grids names one group per grid. Groups agree on nothing but what
    the root publishes, which only rasters on one exact GeoBox earn.

    Args:
        rasters: Group name mapped to its raster Dataset. The names become
            on-disk group names, and carry no `"/"`, a stack being flat. Where
            the rasters share a grid and dimension names, the first supplies the root's spatial
            coordinates, so its coordinate attrs are the ones every group reads.

    Returns:
        Flat DataTree whose children are the given rasters, unchanged, over a
        root carrying the spatial coordinates they share, or no coordinates
        where they share none. The caller retains ownership of opened rasters;
        this factory does not transfer their file-close callbacks.

    Raises:
        ValueError: `rasters` is empty, or a name spells a path.

    Examples:
        >>> stack({"sentinel-2-l2a": optical, "dem": dem}).gs.groups
        ('sentinel-2-l2a', 'dem')

        One product's resolutions are one group each, since no grid holds two
        of them, and the stack carries no grid of its own until it is aligned:

        >>> sentinel = stack({"s2-l2a-10m": bands_10m, "s2-l2a-20m": bands_20m})
        >>> sentinel.gs.geobox is None
        True
    """
    if not rasters:
        raise ValueError("a raster stack needs at least one named raster")

    pathed = sorted(name for name in rasters if "/" in name)
    if pathed:
        raise ValueError(
            f"{pathed} spell paths rather than group names, which would nest "
            f"nodes a stack does not read as rasters; a stack is flat, so name "
            f"one group per grid, as 'sentinel-2-l2a-10m'"
        )

    reference = next(iter(rasters.values()))
    shared = reference.gs.geobox
    if not isinstance(shared, GeoBox) or any(
        raster.gs.geobox != shared for raster in rasters.values()
    ):
        root = xr.Dataset()
    else:
        root = xr.Dataset(
            coords={
                name: reference.coords[name]
                for name in (*SPATIAL_DIMENSIONS, CRS_COORDINATE)
            }
        )
    return cast(
        "DataTree",
        xr.DataTree.from_dict(
            {"/": root, **{f"/{name}": raster for name, raster in rasters.items()}}
        ),
    )


def map_groups(
    tree: xr.DataTree, change: Callable[[xr.Dataset], xr.Dataset]
) -> DataTree:
    """Apply one change to every group, keeping the stack's own attrs.

    The stack is rebuilt rather than mapped over, so a change that moves the
    grid still leaves a root that matches its groups.

    Args:
        tree: Stack to change.
        change: Returns the changed raster for one group.

    Returns:
        New stack of the changed groups, in the same order, carrying the
        root attrs `tree` carried.

    Examples:
        >>> map_groups(scene, lambda raster: raster.isel(time=0)).gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    changed = stack({name: change(raster) for name, raster in tree.gs.rasters.items()})
    return attrs.rebase(changed, tree.gs.attrs.root)


@xr.register_datatree_accessor("gs")
class GeoStack(GeoRasterAccessor["DataTree"]):
    """Read and persist one raster-stack DataTree.

    `attrs` reads only the root's own attrs; groups carry their own, read
    through `rasters["<group>"].gs.attrs`.

    Args:
        data: Flat DataTree whose direct children are raster Datasets.

    Examples:
        >>> scene.gs.groups
        ('sentinel-2-l2a', 'dem')
        >>> scene.gs.geobox.shape
        (512, 512)
    """

    @property
    def groups(self) -> tuple[str, ...]:
        """Return raster group names in stack order.

        Returns:
            Direct child group names.
        """
        return tuple(self._data.children)

    @property
    def variables(self) -> tuple[str, ...]:
        """Name every grouped data variable in stack order."""
        return tuple(
            f"{group}/{variable}"
            for group, raster in self.rasters.items()
            for variable in raster.gs.variables
        )

    @property
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage across every group.

        Groups need not share a cadence, so the stack runs from the earliest
        instant any group reaches to the latest, a timeless group widening
        nothing. Read one group's own span with `rasters["<group>"].gs.timespan`.

        Returns:
            First and last covered instant, or None where no group is dated.

        Examples:
            >>> scene.gs.rasters["dem"].gs.timespan is None
            True
            >>> scene.gs.timespan == scene.gs.rasters["sentinel-2-l2a"].gs.timespan
            True
        """
        spans = [
            span
            for span in (raster.gs.timespan for raster in self.rasters.values())
            if span is not None
        ]
        if not spans:
            return None
        return (min(start for start, _ in spans), max(end for _, end in spans))

    @property
    def times(self) -> pd.DatetimeIndex | None:
        """Read every instant any group reaches, in order.

        Returns:
            Sorted labels across the groups, each instant once, or None where
            no group is dated.
        """
        dated = [
            times
            for times in (raster.gs.times for raster in self.rasters.values())
            if times is not None
        ]
        if not dated:
            return None
        return pd.DatetimeIndex(dated[0].append(dated[1:]).unique().sort_values())

    @property
    def anchor(self) -> GeoAnchor:
        """Read exact spatial and temporal coverage.

        Returns:
            Anchor over the shared grid and everything the groups jointly
            cover in time, which names the stack's centroid and place and
            formats both into a filename.

        Raises:
            ValueError: The groups do not share a grid.

        Examples:
            >>> scene.gs.anchor.format("{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}")
            '50.90N_26.40E_20250601'
        """
        from .anchor import GeoAnchor

        geobox = self.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(
                f"{type(self._data).__name__} publishes no grid at its root, so "
                f"nothing names the ground it covers; align its groups onto one "
                f"grid with gs.align first"
            )
        return GeoAnchor(geobox, timespan=self.timespan)

    @property
    def rasters(self) -> dict[str, Dataset]:
        """Read every group as a raster Dataset.

        Returns:
            {
                group name: that group's raster, carrying the grid and time
                    coordinates it inherits from the root,
            }

        Examples:
            Rebuild a stack to add a group, or to join two of them:

            >>> stack({**scene.gs.rasters, "ndvi": ndvi}).gs.groups
            ('sentinel-2-l2a', 'dem', 'ndvi')
            >>> stack({**optical.gs.rasters, **labels.gs.rasters}).gs.groups
            ('sentinel-2-l2a', 'dem', 'labels')
        """
        # .dataset would return a DatasetView whose attrs still write through here.
        return {
            name: cast(
                "Dataset", self._data.children[name].to_dataset(inherit="all_coords")
            )
            for name in self.groups
        }

    @overload
    def align(
        self,
        *,
        target: warp.Target | None = None,
        extent: warp.Extent = "union",
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | warp.Native | None = None,
        inplace: Literal[False] = False,
    ) -> DataTree: ...

    @overload
    def align(
        self,
        *,
        target: warp.Target | None = None,
        extent: warp.Extent = "union",
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | warp.Native | None = None,
        inplace: Literal[True],
    ) -> None: ...

    def align(
        self,
        *,
        target: warp.Target | None = None,
        extent: warp.Extent = "union",
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | warp.Native | None = None,
        inplace: bool = False,
    ) -> DataTree | None:
        """Warp every group onto one grid, which the root then publishes.

        A stack whose groups sit on their own grids names no shared axes, so
        nothing can index them together. This is what brings them onto one,
        and `Tiles` needs it before it will cut a stack.

        Args:
            target: Grid to align onto, a raster already on one, or a CRS.
                None takes the finest group's grid as it stands.
            extent: Ground the aligned groups cover. `"union"` reaches every
                group, holding fill where one does not; `"intersection"` drops
                every edge outside the ground they all cover; `"match"` adopts
                the reference's own.
            resampling: One GDAL kernel for every variable, or a mapping naming
                each variable's own as `"<group>/<variable>"`, which `"*"`
                answers the rest of. Pixels move only where a group does not
                already sit on the grid they align onto.
            resolution: Pixel size every group lands at. None takes the
                reference's own, so the stack comes out co-registered and
                publishes its grid. `"native"` keeps each group's instead, so
                the groups cover one extent on nesting grids and the root stays
                bare — which `Tiles` cuts group by group.
            inplace: Warp into this stack rather than returning a new one.

        Returns:
            New DataTree whose groups share one exact GeoBox, or cover one
            extent at their own pixel sizes under `"native"`, or None when
            `inplace` is set.

        Raises:
            KeyError: A `resampling` mapping names neither a variable nor `"*"`.
            ValueError: A group sits on no locatable grid, `target` names no
                grid or CRS, `extent` names no ground, `"intersection"` leaves
                none in common, or `resampling` would blend class codes.

        Examples:
            A group names its grid by the raster it is, since a bare string is
            read as a CRS:

            >>> sentinel.gs.geobox is None
            True
            >>> onto = sentinel.gs.rasters["r10"]
            >>> sentinel.gs.align(target=onto, extent="intersection").gs.resolution.x
            10.0

            Imagery and labels in one stack warp with different kernels:

            >>> scene.gs.align(resampling={"*": "bilinear", "labels/mask": "mode"})

            Aligning in place leaves the stack ready to cut:

            >>> sentinel.gs.align(inplace=True)
            >>> Tiles([sentinel], (256, 256))
        """
        # Only the tree knows its qualified names, so a mapping can name a group.
        geobox = warp.common_grid(
            list(self.rasters.values()),
            target=target,
            extent=extent,
            resolution=resolution,
        )
        aligned = warp.reproject(
            self._data,
            geobox,
            resampling=resampling,
            resolution=warp.NATIVE if resolution == warp.NATIVE else None,
        )
        if not inplace:
            return aligned

        # A child disagreeing with the root is refused, so the root's grid goes first.
        self._data.dataset = xr.Dataset(attrs=dict(self._data.attrs))
        for name in aligned.gs.groups:
            self._data[name] = aligned[name]
        self._data.dataset = xr.Dataset(
            coords=aligned.dataset.coords, attrs=dict(self._data.attrs)
        )
        return None

    def plot(self, *, cols: int = 1) -> hv.Layout:
        """Draw every group down the page. Needs the `viz` extra.

        Each group draws through `Dataset.gs.plot`. Its group name, time, and
        location appear below the axes. A group spanning `time` contributes
        one panel per step.

        Args:
            cols: Panels per row. Defaults to one, stacking them in a column.

        Returns:
            Layout of every panel, or the sole panel when the stack holds one
            group with no time axis.

        Raises:
            ValueError: A group names nothing to draw; name its channels with
                `gs.write_rgb` first or plot it alone with
                `stack["<group>"].to_dataset().gs.plot()`.

        Examples:
            >>> hv.save(scene.gs.plot(), "scene.png")
        """
        import holoviews as hv

        # A time facet is an NdLayout; mpl won't nest it, so take its panels.
        panels: list[hv.Element] = []
        for name in self.groups:
            raster = self._data.children[name].to_dataset(inherit="all_coords")
            place = raster.gs.anchor.locate()
            caption = name if place is None else f"{name}\n{place.to_address()}"
            drawn = raster.gs.plot(xlabel=caption)
            panels.extend(drawn.values() if isinstance(drawn, hv.NdLayout) else [drawn])

        return hv.Layout(panels).cols(cols)

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> dict[str, np.ndarray]:
        """Stack each group's variables into one model-input array.

        Groups share a grid but not their variables, non-spatial axes, or
        dtypes, so each keeps its own array rather than being joined. Cast a
        single group afterwards, which is one call on the array it returns.

        Args:
            dtype: Cast applied to every group. None keeps each group's source
                dtype, which every variable of that group must then share.

        Returns:
            {
                group name: array shaped `(*axes, band, y, x)`,
            }

        Raises:
            ValueError: A group's variables carry different non-spatial
                dimensions, or differ in dtype while `dtype` is None.

        Examples:
            >>> {name: a.shape for name, a in scene.gs.to_numpy().items()}
            {'sentinel-2-l2a': (2, 4, 256, 256), 'dem': (1, 256, 256)}
        """
        return {
            name: raster.gs.to_numpy(dtype=dtype)
            for name, raster in self.rasters.items()
        }

    def to_cog(
        self,
        destination: str | PathLike[str],
        *,
        split_bands: bool = False,
        map_scale: float | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> dict[str, tuple[Path | str, ...]]:
        """Write every group as Cloud Optimized GeoTIFFs named after the group.

        Args:
            destination: Directory or fsspec URL the groups are written into.
            split_bands: Give each variable its own single-band file, rather
                than keeping them as bands of one file per instant.
            map_scale: Map denominator used to write pixels per centimetre in
                every file.
            overwrite: Replace files that already exist.
            storage_options: Options for the filesystem a URL names.
            **options: COG creation options passed to every file.

        Returns:
            Group names mapped to exactly the files written for each group,
            which `read_stack` reads back.

        Raises:
            FileExistsError: A file exists and `overwrite` is false.
            ValueError: A group spans a non-spatial axis other than time.

        Examples:
            >>> paths = scene.gs.to_cog("scene")
            >>> paths["dem"]
            (PosixPath('scene/dem.tif'),)
        """
        from geosave_engine.geodata.io import cogs

        return {
            name: cogs.write(
                raster,
                f"{str(destination).rstrip('/')}/{name}",
                split_bands=split_bands,
                map_scale=map_scale,
                overwrite=overwrite,
                storage_options=storage_options,
                **options,
            )
            for name, raster in self.rasters.items()
        }

    def to_zarr(
        self,
        destination: str | PathLike[str],
        *,
        compute: bool = True,
        overwrite: bool = False,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> dict[str, Path | str] | Delayed:
        """Write every group as its own Zarr store named after the group.

        Args:
            destination: Directory or fsspec URL the stores are written into.
            compute: False defers writing pixels.
            overwrite: Replace stores that already exist.
            **write_options: Supported xarray Zarr write options, applied to
                every group.

        Returns:
            Group names mapped to the store written for each, which
            `read_stack` reads back, or a delayed task returning that mapping
            when `compute=False`.

        Raises:
            FileExistsError: A store exists and `overwrite` is false.
            TypeError: An option is unsupported.

        Examples:
            >>> sample.gs.to_zarr("samples/s0")
            {'optical': PosixPath('samples/s0/optical.zarr'),
             'label': PosixPath('samples/s0/label.zarr')}
        """
        from dask.delayed import delayed

        from geosave_engine.geodata.io import zarr

        written = {}
        for name, raster in self.rasters.items():
            written[name] = zarr.write(
                raster,
                f"{str(destination).rstrip('/')}/{name}.zarr",
                compute=compute,
                overwrite=overwrite,
                **write_options,
            )
        if not compute:
            return delayed(dict)(written)
        return written

    def to_netcdf(
        self,
        destination: str | PathLike[str],
        *,
        compute: bool = True,
        engine: NetCDFEngine = "netcdf4",
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> dict[str, Path | str] | Delayed:
        """Write every group as its own netCDF file named after the group.

        Args:
            destination: Directory or fsspec URL the files are written into.
            compute: False defers writing pixels.
            engine: Concrete xarray netCDF writing engine.
            overwrite: Replace files that already exist.
            storage_options: Options for the filesystem a URL names.
            **write_options: Supported xarray netCDF write options, applied to
                every group.

        Returns:
            Group names mapped to the file written for each, which
            `read_stack` reads back, or a delayed task returning that mapping
            when `compute=False`.

        Raises:
            FileExistsError: A file exists and `overwrite` is false.
            TypeError: An option is unsupported.
            ValueError: A URL is asked to defer its write.

        Examples:
            >>> sample.gs.to_netcdf("samples/s0")
            {'optical': PosixPath('samples/s0/optical.nc'),
             'label': PosixPath('samples/s0/label.nc')}
        """
        from dask.delayed import delayed

        from geosave_engine.geodata.io import netcdf

        written = {}
        for name, raster in self.rasters.items():
            written[name] = netcdf.write(
                raster,
                f"{str(destination).rstrip('/')}/{name}.nc",
                compute=compute,
                overwrite=overwrite,
                engine=engine,
                storage_options=storage_options,
                **write_options,
            )
        if not compute:
            return delayed(dict)(written)
        return written

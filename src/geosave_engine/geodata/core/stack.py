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
from typing import TYPE_CHECKING, Any, Literal, Unpack, cast, overload

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import warp

from .base import GeoAccessor
from .convention import CRS_COORDINATE

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
    from geosave_engine.geodata.attrs import AttrsModel
    from geosave_engine.geodata.transform.warp import Resampling
    from geosave_engine.geodata.utils.datetime import DateRange
    from odc.geo import SomeResolution
    from geosave_engine.geodata import DataTree, Dataset
    from geosave_engine.geodata.utils.io.geotiff import COGWriteOptions
    from geosave_engine.geodata.utils.io.layout import Layout
    from geosave_engine.geodata.utils.io.zarr import ZarrWriteOptions

    from .anchor import GeoAnchor


def stack(rasters: Mapping[str, xr.Dataset]) -> DataTree:
    """Build one raster stack from named rasters.

    A group holds as many variables as one grid can, so a product delivering
    several grids names one group per grid. Groups agree on nothing but what
    the root publishes, which only rasters on one exact GeoBox earn.

    Args:
        rasters: Group name mapped to its raster Dataset. The names become
            on-disk group names, and carry no `"/"`, a stack being flat. Where
            the rasters share a grid, the first supplies the root's spatial
            coordinates, so its coordinate attrs are the ones every group reads.

    Returns:
        Flat DataTree whose children are the given rasters, unchanged, over a
        root carrying the spatial coordinates they share, or no coordinates
        where they share none.

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
                for name in (*shared.dimensions, CRS_COORDINATE)
            }
        )
    return cast(
        "DataTree",
        xr.DataTree.from_dict(
            {"/": root, **{f"/{name}": raster for name, raster in rasters.items()}}
        ),
    )


@xr.register_datatree_accessor("gs")
class GeoStack(GeoAccessor["DataTree"]):
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

    def __init__(self, data: xr.DataTree) -> None:
        """Bind the DataTree.

        Args:
            data: DataTree to read through this accessor.
        """
        self._data = cast("DataTree", data)

    @property
    def groups(self) -> tuple[str, ...]:
        """Return raster group names in stack order.

        Returns:
            Direct child group names.
        """
        return tuple(self._data.children)

    @property
    def grid_dims(self) -> tuple[str, str]:
        """Name the two dimensions the shared grid spans.

        Unlike `GeoRaster.grid_dims`, this refuses a stack whose groups sit on
        their own grids rather than falling back to `("y", "x")`, since no one
        pair of axes spans them.

        Returns:
            `("y", "x")` for a projected CRS, `("latitude", "longitude")` for a
            geographic one.

        Raises:
            ValueError: The groups do not share a grid.
        """
        geobox = self.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(
                f"{type(self._data).__name__} publishes no grid at its root, so "
                f"its groups name no shared axes; align them onto one grid with "
                f"gs.align first"
            )
        return geobox.dimensions

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
    def anchor(self) -> GeoAnchor:
        """Read exact spatial and temporal coverage.

        Returns:
            Anchor over the shared grid and everything the groups jointly
            cover in time, which names the stack's centroid, filename stem,
            and place.

        Raises:
            ValueError: The groups do not share a grid.

        Examples:
            >>> scene.gs.anchor.stem
            '26.3970E_50.8960N_800mx800m_202506-20250611T000000.000000_100m'
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
            name: cast("Dataset", self._data[name].to_dataset()) for name in self.groups
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
        aligned.attrs.update(self._data.attrs)
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

    @overload
    def rebase(
        self,
        *models: AttrsModel,
        target: str | Sequence[str] | None = None,
        inplace: Literal[False] = False,
        **model_kwargs: Mapping[str, Any] | None,
    ) -> DataTree: ...

    @overload
    def rebase(
        self,
        *models: AttrsModel,
        target: str | Sequence[str] | None = None,
        inplace: Literal[True],
        **model_kwargs: Mapping[str, Any] | None,
    ) -> None: ...

    def rebase(
        self,
        *models: AttrsModel,
        target: str | Sequence[str] | None = None,
        inplace: bool = False,
        **model_kwargs: Mapping[str, Any] | None,
    ) -> DataTree | None:
        """Return a copy of this stack whose root carries the supplied attrs.

        Only the root is written; groups keep their own, rebased through
        `rasters["<group>"].gs.rebase`.

        Args:
            *models: Model instances to apply to `target`.
            target: Shared coordinate name the models describe, or several of
                them. None writes to the root's own attrs.
            inplace: Write into this stack rather than returning a new one.
            **model_kwargs: Model name mapped to its field values, or to None
                to drop that model.

        Returns:
            New DataTree carrying the attrs without copying pixel data, or None
            when `inplace` is set.

        Raises:
            KeyError: A keyword names no registered model.
            ValueError: `target` names no coordinate of the root.
            ValidationError: A supplied value does not satisfy its field.

        Examples:
            >>> scene.gs.rebase(ACDD(title="Training scene"))
            >>> scene.gs.rebase(coordinate={"axis": "Y"}, target="y")
        """
        if inplace:
            attrs.rebase(
                self._data, *models, target=target, inplace=True, **model_kwargs
            )
            return None
        return attrs.rebase(
            self._data, *models, target=target, inplace=False, **model_kwargs
        )

    def plot(self, *, cols: int = 1) -> hv.Layout:
        """Draw every group down the page. Needs the `viz` extra.

        Each group draws through `Dataset.gs.plot`, captioned with the group
        name. A group spanning `time` contributes one panel per step.

        Args:
            cols: Panels per row. Defaults to one, stacking them in a column.

        Returns:
            Layout of every panel, or the sole panel when the stack holds one
            group with no time axis.

        Raises:
            ValueError: A group names nothing to draw; name each channel's
                `GDALVariable.colorinterp` on it or plot it alone with
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

    def to_tensor(self, *, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
        """Stack each group's variables into one model-input tensor.

        Args:
            dtype: Tensor dtype for every group. None casts each to
                `torch.float32`, which keeps unsigned imagery off dtypes torch
                carries no arithmetic kernels for.

        Returns:
            {
                group name: tensor shaped `(*axes, band, y, x)`,
            }

        Raises:
            ValueError: A group's variables carry different non-spatial
                dimensions.

        Examples:
            >>> {name: t.shape for name, t in scene.gs.to_tensor().items()}
            {'sentinel-2-l2a': (2, 4, 256, 256), 'dem': (1, 256, 256)}
        """
        return {
            name: raster.gs.to_tensor(dtype=dtype)
            for name, raster in self.rasters.items()
        }

    def to_cog(
        self,
        destination: str | PathLike[str],
        *,
        layout: Layout | Literal["nested", "flat"] = "nested",
        overwrite: bool = False,
        **options: Unpack[COGWriteOptions],
    ) -> None:
        """Write every group as a tree of Cloud Optimized GeoTIFFs.

        Each group writes into its own directory named after the group, so the
        groups stay separable on disk.

        Args:
            destination: Directory the groups are written into.
            layout: `"nested"` or `"flat"` on their defaults, or a configured
                `NestedLayout` or `FlatLayout`, applied to every group.
            overwrite: Replace leaves that already exist.
            **options: COG creation options passed to every leaf.

        Raises:
            FileExistsError: A leaf exists and `overwrite` is false.
            KeyError: `layout` names neither `"nested"` nor `"flat"`.
            ValueError: A group spans a non-spatial axis other than time.

        Examples:
            >>> scene.gs.to_cog("scene")  # scene/sentinel-2-l2a/..., scene/dem/...
        """
        root = Path(destination)
        for name, raster in self.rasters.items():
            raster.gs.to_cog(root / name, layout=layout, overwrite=overwrite, **options)

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

"""Build raster Datasets and read them through the `gs` Dataset accessor.

Georeferencing is odc-geo's. GeoSave adds only the CF axis semantics odc
omits, which `write_crs` writes.

Examples:
    A raster carrying no CRS holds pixels without claiming ground position, so
    `GeoRaster.geobox` is None, and members needing ground position refuse.
    Variable access still works:

    >>> png
    <xarray.Dataset> Size: 262kB
    Dimensions:  (y: 512, x: 512)
    Dimensions without coordinates: y, x
    Data variables:
        band     (y, x) uint8 ...

    A georeferenced one carries `spatial_ref`, which odc reads a GeoBox off:

    >>> opened
    <xarray.Dataset> Size: 2MB
    Dimensions:      (time: 2, y: 512, x: 512)
    Coordinates:
      * time         (time) datetime64[ns] 2025-06-01 2025-06-11
      * y            (y) float64 5.0051e+06 5.0051e+06 ... 5.0000e+06
      * x            (x) float64 3.0000e+05 3.0000e+05 ... 3.0511e+05
        spatial_ref  int32 32633
    Data variables:
        red          (time, y, x) uint16 ...
    >>> opened.y.attrs
    {'units': 'metre', 'resolution': -10.0, 'crs': 'EPSG:32633'}

    `write_crs` adds what those axes measure, which odc leaves out:

    >>> opened.gs.write_crs().y.attrs
    {'units': 'metre', 'resolution': -10.0, 'crs': 'EPSG:32633',
     'standard_name': 'projection_y_coordinate', 'axis': 'Y'}

    A geographic CRS names the axes `latitude` and `longitude` instead,
    following odc-geo:

    >>> wgs84.gs.grid_dims
    ('latitude', 'longitude')
    >>> wgs84.latitude.attrs["standard_name"], wgs84.latitude.attrs["units"]
    ('latitude', 'degrees_north')
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Unpack, cast

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.attrs.headers.geobox import (
    create_header as create_geobox_header,
)
from geosave_engine.geodata.transform import nodata

from .array import tensor
from .base import GeoRasterAccessor
from .profile import (
    BAND_DIMENSION,
    CRS_COORDINATE,
    NOT_GEOREFERENCED_DIMENSIONS,
    TIME_COORDINATE,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from dask.delayed import Delayed
    from os import PathLike
    from numpy.typing import DTypeLike
    from odc.geo import SomeCRS

    import torch

    from geosave_engine.geodata.io.netcdf import (
        NetCDFEngine,
        NetCDFWriteOptions,
    )
    from geosave_engine.geodata.io.geotiff import COGWriteOptions
    from geosave_engine.geodata.io.layout import LeafPath
    from geosave_engine.geodata.io.zarr import ZarrWriteOptions

    import holoviews as hv

    from geosave_engine.geodata import DataArray, Dataset

# One variable's pixels, alone or paired with the axes it carries.
type RasterVariable = np.ndarray | tuple[np.ndarray, Sequence[str]]


class AxisExtent(NamedTuple):
    """How long one axis is, and which variable said so.

    Args:
        sized_by: Name of the first variable that gave the axis its length.
        length: Positions along the axis.
    """

    sized_by: str
    length: int


def raster(
    variables: Mapping[str, RasterVariable],
    geobox: GeoBox | None = None,
    /,
    *,
    nodata: float | int | None = None,
    **coords: Sequence[Any] | np.ndarray | None,
) -> Dataset:
    """Build one raster from named arrays sharing a grid.

    Arrays end in the two spatial axes, which the geobox names, and carry the
    axes `coords` names ahead of them. A bare array carries every axis
    `coords` names; pair it with its own names to carry only some.

    Args:
        variables: Data variable name mapped to its pixels, or to a
            `(pixels, axes)` pair naming the axes it carries.
        geobox: Grid placing the trailing two axes. None leaves the raster
            unreferenced, so it holds pixels without claiming ground position.
        nodata: Value standing for nodata pixels, written on every variable
            as `Nodata.fill_value`. None writes no fill value.
        **coords: Leading axis name mapped to its labels, in array order. None
            labels an axis carrying none. Pass a name Python reserves as
            `**{"class": labels}`.

    Returns:
        Georeferenced Dataset when `geobox` is given, its spatial coordinates
        naming what they measure, otherwise a Dataset carrying no grid, whose
        spatial dims are named `y` and `x`.

    Raises:
        ValueError: `variables` is empty, a variable names an axis no keyword gives,
            its rank does not match the axes it carries, its trailing axes do
            not match `geobox`, two variables size one axis differently, an
            axis is labelled with the wrong number of values, an axis no
            variable carries is named, or an axis name collides with a
            coordinate the grid supplies.

    Examples:
        A labelled axis becomes a coordinate every variable carrying it shares:

        >>> raster({"ndvi": cube}, geobox, time=labels)
        <xarray.Dataset> Size: 2MB
        Dimensions:      (y: 512, x: 512, time: 2)
        Coordinates:
          * y            (y) float64 5.005e+06 5.005e+06 ... 5e+06
          * x            (x) float64 3e+05 3e+05 ... 3.051e+05
          * time         (time) datetime64[ns] 2025-06-01 2025-06-11
            spatial_ref  int32 32633
        Data variables:
            ndvi         (time, y, x) float32 ...

        Pairing pixels with their own axes lets variables differ in rank:

        >>> raster({"ndvi": cube, "dem": (flat, ())}, geobox, time=labels)
        <xarray.Dataset> Size: 3MB
        Dimensions:      (y: 512, x: 512, time: 2)
        Coordinates:
          * y            (y) float64 5.005e+06 5.005e+06 ... 5e+06
          * x            (x) float64 3e+05 3e+05 ... 3.051e+05
          * time         (time) datetime64[ns] 2025-06-01 2025-06-11
            spatial_ref  int32 32633
        Data variables:
            ndvi         (time, y, x) float32 ...
            dem          (y, x) float32 ...
    """
    if not variables:
        raise ValueError("a raster needs at least one named array")

    from odc.geo.xr import wrap_xr

    spatial_dims = NOT_GEOREFERENCED_DIMENSIONS if geobox is None else geobox.dimensions
    grid_coords = (*spatial_dims, CRS_COORDINATE)
    raster_axes = tuple(coords)

    collisions = sorted(set(raster_axes) & set(grid_coords))
    if collisions:
        raise ValueError(
            f"{collisions} name coordinates this grid already supplies "
            f"{list(grid_coords)}; name the leading axes something else"
        )

    # An axis is as long as the variables along it say, and they must agree.
    axis_extent: dict[str, AxisExtent] = {}
    data_vars: dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray]] = {}

    for name, value in variables.items():
        if isinstance(value, tuple):
            pixels, variable_axes = value[0], tuple(value[1])
        else:
            pixels, variable_axes = value, raster_axes

        unknown = [axis for axis in variable_axes if axis not in raster_axes]
        if unknown:
            raise ValueError(
                f"{name!r} lies along {unknown}, which no keyword names; this "
                f"raster names {list(raster_axes)}"
            )
        if pixels.ndim != len(variable_axes) + 2:
            raise ValueError(
                f"{name!r} is {pixels.ndim}-dimensional but lies along "
                f"{variable_axes} ahead of the spatial pair; a raster's arrays end "
                f"in the two spatial axes"
            )
        if geobox is not None and pixels.shape[len(variable_axes) :] != geobox.shape:
            raise ValueError(
                f"{name!r}'s trailing axes are {pixels.shape[len(variable_axes) :]} "
                f"but the geobox is {tuple(geobox.shape)}; place it on a matching "
                f"grid first"
            )

        variable_dims = (*variable_axes, *spatial_dims)
        for position, axis in enumerate(variable_dims):
            extent = axis_extent.setdefault(
                axis, AxisExtent(name, pixels.shape[position])
            )
            if extent.length != pixels.shape[position]:
                raise ValueError(
                    f"{name!r} makes axis {axis!r} {pixels.shape[position]} long but "
                    f"{extent.sized_by!r} makes it {extent.length}; one axis is one length"
                )

        if geobox is None:
            data_vars[name] = (variable_dims, pixels)
        else:
            data_vars[name] = wrap_xr(
                pixels, geobox, dims=variable_dims, axis=len(variable_axes)
            )

    empty_axes = [axis for axis in raster_axes if axis not in axis_extent]
    if empty_axes:
        raise ValueError(
            f"{empty_axes} are named but no variable lies along them; drop them "
            f"or name them on a variable"
        )

    miscounted = []
    for axis, labels in coords.items():
        if labels is None:
            continue
        length = axis_extent[axis].length
        if len(labels) != length:
            miscounted.append(
                f"{axis!r} got {len(labels)} labels for an axis of {length}"
            )
    if miscounted:
        raise ValueError(
            f"{'; '.join(miscounted)}; label an axis once per value along it"
        )

    built = xr.Dataset(data_vars)

    axis_labels = {axis: given for axis, given in coords.items() if given is not None}
    if axis_labels:
        built = built.assign_coords(axis_labels)

    if nodata is not None:
        built = built.gs.write_nodata(nodata)

    if geobox is not None:
        # odc places the axes; GeoSave writes what they measure.
        built = attrs.rebase(built, create_geobox_header(geobox))
    return cast("Dataset", built)


@xr.register_dataset_accessor("gs")
class GeoRaster(GeoRasterAccessor["Dataset"]):
    """Read and transform one raster Dataset.

    Each member reads only what it needs, so a Dataset mid-computation still
    answers for its grid.

    Args:
        data: Dataset holding one raster.

    Examples:
        >>> ds = source.load(anchor)
        >>> ds.gs.anchor.geobox.shape
        (512, 512)
    """

    def __init__(self, data: xr.Dataset) -> None:
        """Bind the Dataset.

        Args:
            data: Dataset to read through this accessor.
        """
        self._data = cast("Dataset", data)

    @property
    def grid_dims(self) -> tuple[str, str]:
        """Name the two dimensions the grid spans.

        Returns:
            `("y", "x")` for a projected CRS, `("latitude", "longitude")` for a
            geographic one, and `("y", "x")` for a raster carrying no grid,
            whose pixels span those axes without claiming ground position.
        """
        grid = self._data.odc.geobox
        return NOT_GEOREFERENCED_DIMENSIONS if grid is None else grid.dimensions

    @property
    def variables(self) -> tuple[str, ...]:
        """Name the data variables.

        Returns:
            Variable names, in Dataset order.
        """
        return tuple(str(name) for name in self._data.data_vars)

    def _require_variables(self, names: Sequence[str]) -> None:
        """Refuse names this raster carries no data variable for.

        Args:
            names: Data variable names the caller supplied.

        Raises:
            ValueError: A name is not a data variable of this raster.
        """
        unknown = sorted(set(names) - set(self.variables))
        if unknown:
            raise ValueError(
                f"{unknown} are not data variables of this raster; it carries "
                f"{list(self.variables)}"
            )

    @property
    def times(self) -> np.ndarray | None:
        """Read the time coordinate labels.

        Returns:
            Labels in axis order, or None for timeless data.
        """
        if TIME_COORDINATE not in self._data.coords:
            return None
        return self._data.variables[TIME_COORDINATE].values

    def write_crs(self, crs: SomeCRS | None = None) -> Dataset:
        """Write onto the spatial coordinates what they measure.

        The grid decides the answer: a projected raster names
        `projection_y_coordinate` and `projection_x_coordinate`, a geographic
        one `latitude` and `longitude`. No pixel moves; `reproject` does that.

        Args:
            crs: CRS the existing coordinates are in, for a raster that
                carries none or carries the wrong one. None reads the CRS
                the raster already carries.

        Returns:
            New Dataset whose spatial coordinates carry what the grid says of
            them: odc's units, resolution, and CRS, and CF's standard name and
            axis. Their other attrs are replaced.

        Raises:
            ValueError: The raster carries no locatable grid and `crs` names
                none, `crs` names no known reference system, or the raster is
                placed by ground control points rather than a regular grid.

        Examples:
            >>> ds.gs.write_crs("EPSG:32633").y.attrs["standard_name"]
            'projection_y_coordinate'
        """
        result = self._data if crs is None else self._data.odc.assign_crs(crs)
        geobox = result.gs.geobox
        if geobox is None:
            raise ValueError(
                "raster carries no locatable grid and no crs was given; pass "
                "one or assign a CRS first"
            )
        if not isinstance(geobox, GeoBox):
            raise ValueError(
                f"raster is placed by {type(geobox).__name__}, not a regular "
                f"grid, so its axes measure no CF coordinate"
            )
        return cast("Dataset", attrs.rebase(result, create_geobox_header(geobox)))

    def write_nodata(
        self,
        value: float | int | None,
        *,
        target: str | Sequence[str] | None = None,
    ) -> Dataset:
        """Write the stored value standing for nodata pixels.

        `Nodata` mirrors the value across both `_FillValue` and odc's own
        `nodata`, so no reader can find the two naming different pixels.

        Args:
            value: Stored value marking nodata. None clears it.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables carry `value` as their fill.

        Raises:
            ValueError: `target` names something that is not a data variable.
            ValidationError: `value` is not a stored fill value.

        Examples:
            >>> ds.gs.write_nodata(0).red.attrs["_FillValue"]
            0
            >>> ds.gs.write_nodata(-9999, target="elevation")
        """
        if target is None:
            var_names: Sequence[str] = self.variables
        elif isinstance(target, str):
            var_names = (target,)
        else:
            var_names = tuple(target)

        self._require_variables(var_names)

        if value is not None:
            for name in var_names:
                nodata.check_fill_fits(value, self._data[name].dtype, name)

        # Nodata owns both spellings of a fill value, and mirrors them.
        return attrs.rebase(
            self._data, attrs.Nodata(fill_value=value), target=var_names
        )

    def write_rgb(self, red: str, green: str, blue: str) -> Dataset:
        """Name the three variables a true-colour composite draws.

        What a band measures does not say which channel draws it, a false
        colour composite drawing near-infrared as red. Recording the choice is
        also the only way a `GeoStack` panel can name its own bands.

        Args:
            red: Variable drawn as the red channel.
            green: Variable drawn as the green channel.
            blue: Variable drawn as the blue channel.

        Returns:
            New Dataset whose three named variables carry their channel, every
            other variable left uninterpreted.

        Raises:
            ValueError: A name is not a data variable, or one variable is named
                for two channels.

        Examples:
            >>> ds.gs.write_rgb("B04", "B03", "B02").gs.plot()
        """
        channels = dict(zip(("red", "green", "blue"), (red, green, blue), strict=True))

        self._require_variables(tuple(channels.values()))
        if len(set(channels.values())) < len(channels):
            raise ValueError(
                f"one variable cannot draw two channels, but these name "
                f"{channels}; give each channel its own variable"
            )

        # A band left interpreted would compose a second, stale colour.
        written = attrs.rebase(
            self._data,
            attrs.GDALVariable(colorinterp=None),
            target=self.variables,
        )
        for colour, name in channels.items():
            written = attrs.rebase(
                written, attrs.GDALVariable(colorinterp=colour), target=name
            )
        return written

    def to_array(self, *, dtype: DTypeLike | None = None) -> DataArray:
        """Stack every variable this raster carries into one array.

        No packing or display scaling is applied. A multi-band array keeps
        names on `band`, and `GeoArray.to_raster` stacks it back apart.

        Args:
            dtype: Cast applied to the stacked array. None keeps the source
                dtype, which every stacked variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`. Its own attrs are the ones
            every band carries with one value, so it reads as one variable.
            Each band's remaining attrs and this raster's own attrs ride on
            the `band` coordinate as `StackedAttrs`.

        Raises:
            ValueError: This raster carries no variables or already spans the
                stacked axis, the variables carry different non-spatial
                dimensions, or they differ in dtype while `dtype` is None.
            TypeError: An attr has no JSON spelling.

        Examples:
            >>> ds[["ndvi"]].gs.to_array()
            >>> ds[["B04", "B03", "B02"]].gs.to_array()
        """
        array = self._stacked(dtype)
        band_attrs = {
            name: dict(self._data.variables[name].attrs) for name in self.variables
        }
        # A key every band carries with one value describes the stacked array.
        shared = attrs.common_attrs(list(band_attrs.values()))
        array.attrs = shared
        stacked = attrs.StackedAttrs(
            variable_attrs={
                name: {key: value for key, value in held.items() if key not in shared}
                for name, held in band_attrs.items()
            },
            dataset_attrs=dict(self._data.attrs),
        )
        attrs.rebase(array, stacked, target=BAND_DIMENSION, inplace=True)
        return cast("DataArray", array)

    def plot(
        self,
        variable: str | tuple[str, str, str] | None = None,
        *,
        cmap: str | Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        cols: int = 4,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> hv.Element | hv.NdLayout:
        """Draw this raster the way `variable` names it. Needs the `viz` extra.

        Three names give a composite, one gives an Image, and None falls back
        to a sole variable, then to the ones measuring red, green, and blue.
        A `Legend` the variable carries supplies the class and colour maps.

        Args:
            variable: One data variable, three ordered red, green, blue, or
                None to read what this raster carries.
            cmap: Colormap. Refused for a composite, which draws no colormap.
            clim: Bounds mapped onto the full display range: colour limits for
                an Image, the composite's stretch onto `[0, 1]`.
            cols: Columns a `time` axis lays panels into. Ignored otherwise.
            title: Panel title above the axes. None suppresses automatic
                titles.
            xlabel: Location caption below the axes. None uses the raster's
                reverse-geocoded location; an empty string suppresses it.

        Returns:
            Image or RGB element over this raster's grid, or a Layout of one
            panel per `time` value, `cols` wide.

        Raises:
            ValueError: A named variable is absent, `variable` is None while
                this raster carries more than one variable and measures no
                red, green, and blue among them, two variables measure one
                colour, or the drawing refuses the raster.

        Examples:
            >>> hv.save(ds.gs.plot("ndvi"), "ndvi.png")
            >>> ds.gs.plot("ndvi", cmap="RdYlGn") + ds.gs.plot(clim=(0.0, 0.3))
        """
        # viz/__init__ re-exports a function named plot, shadowing the submodule.
        from geosave_engine.geodata.viz.plot import plot as draw

        if isinstance(variable, str):
            selected: tuple[str, ...] = (variable,)
        elif variable is not None:
            selected = tuple(variable)
        elif len(self.variables) == 1:
            selected = self.variables
        else:
            names = self.variables
            selected = tuple(
                names[band] for band in attrs.GDALVariable.rgb_indices(self._data)
            )

        self._require_variables(selected)
        array = self._data[list(selected)].gs.to_array()

        # A composite carries colour rather than class codes, so it draws no legend.
        legend = None
        if len(selected) == 1:
            legend = attrs.Legend.from_attrs(self._data.variables[selected[0]].attrs)

        place = self.anchor.location if xlabel is None else None
        caption = place.to_address() if place is not None else xlabel
        return draw(
            array,
            cmap=cmap,
            clim=clim,
            class_map=legend.class_map if legend else None,
            color_map=legend.color_map if legend else None,
            cols=cols,
            title=title,
            xlabel=caption,
        )

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> np.ndarray:
        """Stack every variable this raster carries into one model-input array.

        Every axis besides the grid leads, then the variables, then the grid
        pair — torchgeo's `[T, C, H, W]`. No batch axis is added, so one raster
        is one sample. Order the variables with xarray before stacking them.

        Args:
            dtype: Cast applied to the stacked array. None keeps the source
                dtype, which every variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`.

        Raises:
            ValueError: This raster carries no variables, they carry different
                non-spatial dimensions, or they differ in dtype while `dtype`
                is None.

        Examples:
            >>> ds[["B04", "B08"]].gs.to_numpy().shape
            (2, 4, 256, 256)
        """
        return self._stacked(dtype).values

    def _stacked(self, dtype: DTypeLike | None) -> xr.DataArray:
        """Stack every variable onto the band axis, in the order held.

        Args:
            dtype: Cast applied to the stack. None keeps the source dtype,
                which every variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`, carrying no attrs.

        Raises:
            ValueError: This raster carries no variables or already spans the
                band axis, a variable does not span the grid, the variables
                carry different axes, or they differ in dtype while `dtype` is
                None.
        """
        var_names = self.variables
        if not var_names:
            raise ValueError(
                "this raster carries no variables to stack; it holds only coordinates"
            )

        if BAND_DIMENSION in self._data.dims:
            raise ValueError(
                f"this raster already carries a {BAND_DIMENSION!r} dimension, "
                f"which the stacked variables would name; rename it first"
            )

        grid_dims = self.grid_dims
        ungridded = sorted(
            name
            for name in var_names
            if not set(grid_dims) <= set(self._data.variables[name].dims)
        )
        if ungridded:
            raise ValueError(
                f"{ungridded} do not span the grid {list(grid_dims)}, so stacking "
                f"them would repeat one value across every pixel; drop them or "
                f"place them on the grid first"
            )

        # Past the grid, the variables must span the same axes to stack.
        axes: dict[str, tuple[str, ...]] = {}
        for name in var_names:
            dims = self._data.variables[name].dims
            axes[name] = tuple(str(dim) for dim in dims if dim not in grid_dims)

        drifted = sorted(
            name for name in var_names[1:] if axes[name] != axes[var_names[0]]
        )
        if drifted:
            raise ValueError(
                f"{drifted} carry different axes from {var_names[0]!r}'s "
                f"{axes[var_names[0]]}; stack variables sharing one shape"
            )

        # Stacking promotes silently, so disagreeing dtypes are refused ahead of it.
        dtypes = {str(self._data.variables[name].dtype) for name in var_names}
        if dtype is None and len(dtypes) > 1:
            raise ValueError(
                f"{list(var_names)} carry different dtypes ({sorted(dtypes)}); stacking "
                f"them would promote them, so pass dtype= to choose one"
            )

        # A model reads the grid last, bands right before it: [T, C, H, W].
        stacked = (
            self._data[list(var_names)]
            .to_dataarray(dim=BAND_DIMENSION)
            .transpose(*axes[var_names[0]], BAND_DIMENSION, *grid_dims)
        )
        return stacked if dtype is None else stacked.astype(dtype)

    def to_tensor(self, *, dtype: str | torch.dtype | None = None) -> torch.Tensor:
        """Stack every variable this raster carries into one model-input tensor.

        Select and order the variables with xarray before stacking them.

        Args:
            dtype: Torch dtype or its YAML-friendly name. None preserves the
                prepared raster dtype. A dtype numpy cannot hold, such as
                `torch.bfloat16`, stacks as float32 and narrows on conversion.

        Returns:
            Tensor shaped `(*axes, band, y, x)`.

        Raises:
            ValueError: This raster carries no variables, or they carry
                different non-spatial dimensions.

        Examples:
            >>> ds[["B04", "B08"]].gs.to_tensor().dtype
            torch.uint16
        """
        return tensor(lambda stacking: self.to_numpy(dtype=stacking), dtype)

    def to_cog(
        self,
        destination: str | PathLike[str],
        *,
        layout: str | LeafPath = "nested",
        split_bands: bool = False,
        map_scale: float | None = None,
        overwrite: bool = False,
        catalog: str | PathLike[str] | None = None,
        id: str | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> Path:
        """Write this raster as a tree of Cloud Optimized GeoTIFFs.

        A GeoTIFF holds one instant of one grid, so a cube spreads across
        files. `layout` decides how: see `io.layout`.

        Args:
            destination: Directory the tree is written into, or the file path
                when the raster carries no time axis and `split_bands` is off,
                which writes it as one file.
            layout: `"nested"`, `"flat"`, or a callable placing one leaf from
                its instant and variable. Ignored unless `split_bands` is set.
            split_bands: Give each variable its own single-band file, rather
                than keeping them as bands of one file per instant.
            map_scale: Map denominator used to write pixels per centimetre in
                every leaf.
            catalog: Optional GeoParquet recording the saved assets.
            id: Record identity; None uses the saved raster's anchor.
            overwrite: Replace leaves that already exist.
            **options: COG creation options passed to every leaf.

        Returns:
            Saved directory or file path.

        Raises:
            FileExistsError: A leaf exists and `overwrite` is false.
            KeyError: `layout` names no known arrangement.
            ValueError: This raster carries no locatable grid, or spans a
                non-spatial axis other than time.

        Examples:
            >>> ds.gs.to_cog("scene")  # scene/20250601T103031.tif, ...
            >>> ds.gs.to_cog("scene", layout="flat", split_bands=True)
        """
        from geosave_engine.geodata.io.layout import write_tree

        from geosave_engine.geodata.io.assets import write_catalog

        saved = write_tree(
            self._data,
            destination,
            layout=layout,
            split_bands=split_bands,
            map_scale=map_scale,
            overwrite=overwrite,
            **options,
        )

        return cast("Path", write_catalog(saved, catalog, id=id, overwrite=overwrite))

    def to_zarr(
        self,
        destination: str | PathLike[str],
        *,
        compute: bool = True,
        overwrite: bool = False,
        catalog: str | PathLike[str] | None = None,
        id: str | None = None,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> Path | Delayed:
        """Write this raster to Zarr.

        Args:
            destination: Output path ending in `.zarr`.
            compute: False defers writing pixels and publishing the catalog.
            catalog: Optional GeoParquet recording the saved assets.
            id: Record identity; None uses the saved raster's anchor.
            overwrite: Replace an existing destination when true.
            **write_options: Supported xarray Zarr write options.

        Returns:
            Destination path, or a delayed task returning it when `compute=False`.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.
        """
        from geosave_engine.geodata.io import zarr

        from geosave_engine.geodata.io.assets import write_catalog

        saved = zarr.write(
            self._data,
            destination,
            compute=compute,
            overwrite=overwrite,
            **write_options,
        )

        return write_catalog(saved, catalog, id=id, overwrite=overwrite)

    def to_netcdf(
        self,
        destination: str | PathLike[str],
        *,
        compute: bool = True,
        engine: NetCDFEngine = "netcdf4",
        overwrite: bool = False,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> Path | Delayed:
        """Write this raster to netCDF.

        Args:
            destination: Output path ending in `.nc`, `.nc4`, or `.cdf`.
            compute: False returns a delayed task that writes and returns its path.
            engine: Concrete xarray netCDF writing engine.
            overwrite: Replace an existing destination when true.
            **write_options: Supported xarray netCDF write options.

        Returns:
            Destination path, or a delayed task returning it when `compute=False`.

        Raises:
            FileExistsError: The destination exists and overwrite is false.
            TypeError: An option is unsupported.
            ValueError: The destination is invalid.
        """
        from geosave_engine.geodata.io import netcdf

        return netcdf.write(
            self._data,
            destination,
            compute=compute,
            engine=engine,
            overwrite=overwrite,
            **write_options,
        )

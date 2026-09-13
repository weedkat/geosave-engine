"""Build raster Datasets and read them through the `gs` Dataset accessor.

Georeferencing is odc-geo's. GeoSave adds only the CF axis semantics odc
omits, which `write_crs` writes.

Examples:
    A raster carrying no CRS holds pixels without claiming ground position, so
    `GeoRaster.geobox` refuses, and every member reading the grid with it.
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
from typing import TYPE_CHECKING, Any, Literal, Unpack, cast, overload

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.transform import nodata, packing, warp

from .anchor import GeoAnchor
from .base import GeoAccessor, tensor
from .convention import (
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
    from odc.geo import SomeCRS, SomeResolution

    from geosave_engine.geodata.transform.warp import Resampling

    import torch

    from geosave_engine.geodata.attrs import AttrsModel

    from geosave_engine.geodata.utils.io.netcdf import (
        NetCDFEngine,
        NetCDFWriteOptions,
    )
    from geosave_engine.geodata.utils.io.geotiff import COGWriteOptions
    from geosave_engine.geodata.utils.io.layout import Layout
    from geosave_engine.geodata.utils.io.zarr import ZarrWriteOptions
    from geosave_engine.geodata.utils.datetime import DateRange

    import holoviews as hv

    from geosave_engine.geodata import DataArray, Dataset
    from .vector import GeoVector

# One variable's pixels, alone or paired with the axes it carries.
type RasterVariable = np.ndarray | tuple[np.ndarray, Sequence[str]]


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

    shadowed = sorted(set(raster_axes) & set(grid_coords))
    if shadowed:
        raise ValueError(
            f"{shadowed} name coordinates this grid already supplies "
            f"{list(grid_coords)}; name the leading axes something else"
        )

    # An axis is as long as the variables along it say, and they must agree.
    axis_length: dict[str, tuple[str, int]] = {}
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
            first, length = axis_length.setdefault(axis, (name, pixels.shape[position]))
            if length != pixels.shape[position]:
                raise ValueError(
                    f"{name!r} makes axis {axis!r} {pixels.shape[position]} long but "
                    f"{first!r} makes it {length}; one axis is one length"
                )

        if geobox is None:
            data_vars[name] = (variable_dims, pixels)
        else:
            data_vars[name] = wrap_xr(
                pixels, geobox, dims=variable_dims, axis=len(variable_axes)
            )

    empty_axes = [axis for axis in raster_axes if axis not in axis_length]
    if empty_axes:
        raise ValueError(
            f"{empty_axes} are named but no variable lies along them; drop them "
            f"or name them on a variable"
        )

    miscounted = [
        f"{axis!r} got {len(labels)} labels for an axis of {axis_length[axis][1]}"
        for axis, labels in coords.items()
        if labels is not None and len(labels) != axis_length[axis][1]
    ]
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

    if geobox is None:
        return cast("Dataset", built)
    return built.gs.write_crs()


@xr.register_dataset_accessor("gs")
class GeoRaster(GeoAccessor["Dataset"]):
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

    @property
    def times(self) -> np.ndarray | None:
        """Read the time coordinate labels.

        Returns:
            Labels in axis order, or None for timeless data.
        """
        if TIME_COORDINATE not in self._data.coords:
            return None
        return self._data.variables[TIME_COORDINATE].values

    @property
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage.

        A resampled axis carries a `TimeSpec`, so a label standing for a month
        covers that month. An axis carrying none names instants.

        Returns:
            First and last covered instant, or None for timeless data.
        """
        labels = self.times
        if labels is None:
            return None
        spec = attrs.AttrsNamespace.from_attrs(
            self._data.variables[TIME_COORDINATE].attrs
        ).get(attrs.TimeSpec)
        return (spec or attrs.TimeSpec.instants()).timespan(labels)

    @property
    def anchor(self) -> GeoAnchor:
        """Read exact spatial and temporal coverage.

        Returns:
            Anchor over this raster's grid and time span.

        Raises:
            ValueError: The Dataset carries no locatable grid.
        """
        return GeoAnchor(self.geobox, timespan=self.timespan)

    @overload
    def rebase(
        self,
        *models: AttrsModel,
        target: str | Sequence[str] | None = None,
        inplace: Literal[False] = False,
        **model_kwargs: Mapping[str, Any] | None,
    ) -> Dataset: ...

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
    ) -> Dataset | None:
        """Return a copy of this raster carrying the supplied attrs.

        Args:
            *models: Model instances to apply to `target`.
            target: Variable or coordinate name the models describe, or
                several of them. None writes to the Dataset's own attrs.
            inplace: Write into this raster rather than returning a new one.
            **model_kwargs: Model name mapped to its field values, or to None
                to drop that model.

        Returns:
            New Dataset carrying the attrs without copying pixel data, or None
            when `inplace` is set.

        Raises:
            KeyError: A keyword names no registered model.
            ValueError: `target` names neither a variable nor a coordinate.
            ValidationError: A supplied value does not satisfy its field.

        Examples:
            >>> ds.gs.rebase(ACDD(title="Sentinel-2 Level-2A"))
            >>> ds.gs.rebase(cf={"units": "1"}, target="B04")
        """
        if inplace:
            attrs.rebase(
                self._data, *models, target=target, inplace=True, **model_kwargs
            )
            return None
        return attrs.rebase(
            self._data, *models, target=target, inplace=False, **model_kwargs
        )

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
            New Dataset whose spatial coordinates carry the standard name,
            units, and axis their grid gives them.

        Raises:
            ValueError: The raster carries no locatable grid and `crs` names
                none, or `crs` names no known reference system.

        Examples:
            >>> ds.gs.write_crs("EPSG:32633").y.attrs["standard_name"]
            'projection_y_coordinate'
        """
        result = self._data if crs is None else self._data.odc.assign_crs(crs)
        for name, semantics in attrs.CFCoordinate.from_geobox(result.gs.geobox).items():
            result = result.gs.rebase(semantics, target=name)
        return cast("Dataset", result)

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

        unknown = sorted(set(var_names) - set(self.variables))
        if unknown:
            raise ValueError(
                f"{unknown} are not data variables of this raster; it carries "
                f"{list(self.variables)}"
            )

        # astype wraps, so an unheld -1 would name the pixel 65535 on uint16.
        for name in var_names if value is not None else ():
            dtype = self._data[name].dtype
            with np.errstate(invalid="ignore"):
                stored = np.asarray(value).astype(dtype)
            if not np.array_equal(stored, np.asarray(value), equal_nan=True):
                raise ValueError(
                    f"{value!r} marks no pixel of {name!r}, which is {dtype} and "
                    f"stores it as {stored.item()!r}; CF asks a fill value to have "
                    f"the variable's own type"
                )

        # Nodata owns both spellings of a fill value, and mirrors them.
        return attrs.rebase(
            self._data, attrs.Nodata(_FillValue=value), target=var_names
        )

    def unpack(self) -> Dataset:
        """Read physical values out of every variable's stored digital numbers.

        Returns:
            New Dataset of physical values, a variable unchanged where it
            carries neither `scale_factor` nor `add_offset`.

        Examples:
            >>> ds.gs.unpack().B04.max().item()
            0.09
        """
        return packing.decode(self._data)

    def mask(
        self, valid: xr.DataArray | np.ndarray, *, fill: float | int | None = None
    ) -> Dataset:
        """Make nodata every pixel `valid` does not keep.

        Args:
            valid: Boolean array, True where a pixel is real data. A
                DataArray names its own axes and may span fewer than this
                raster, broadcasting over the rest; a bare numpy array names
                none, so it is read as the grid alone and must match its
                shape.
            fill: Value the blanked pixels take, also written onto the result
                for variables carrying no fill value yet. None reads what
                each variable already carries.

        Returns:
            New Dataset, its unkept pixels holding the fill value their
            variable carries.

        Raises:
            ValueError: `valid` is not boolean, does not span the grid, spans
                an axis this raster does not, `fill` does not fit a
                variable's dtype, or a variable carries no fill value and
                none is given.

        Examples:
            >>> clear = ds.gs.mask(ds.scl.isin([4, 5, 6, 7]))
        """
        return nodata.mask(self._data, valid, fill=fill)

    def decode(self) -> Dataset:
        """Replace each variable's fill value with NaN.

        Returns:
            New Dataset holding NaN where the pixels were nodata, a variable
            unchanged where it carries no fill value.

        Examples:
            >>> ds.gs.decode().red.dtype
            dtype('float64')
        """
        return nodata.decode(self._data)

    def reproject(
        self,
        crs: SomeCRS,
        *,
        resampling: Resampling = "nearest",
        resolution: SomeResolution | None = None,
    ) -> Dataset:
        """Warp pixels into another CRS, deriving the grid to land on.

        Reach for `reproject_match` where the target grid already exists;
        this is for when only the CRS is decided and the grid is sized from
        the source.

        Args:
            crs: Coordinate reference system to land in.
            resampling: GDAL resampling kernel.
            resolution: Output pixel size. None keeps the source's ground
                sampling as closely as the new CRS allows.

        Returns:
            New Dataset in `crs`, on a grid covering the source.

        Raises:
            ValueError: The raster carries no locatable grid, a variable does
                not span it, or `resampling` would blend a variable whose
                values are class codes.

        Examples:
            >>> ds.gs.reproject("EPSG:3857").gs.crs.epsg
            3857
        """
        return warp.reproject(
            self._data, crs, resampling=resampling, resolution=resolution
        )

    def reproject_match(
        self,
        match: GeoBox | xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        resampling: Resampling = "nearest",
    ) -> Dataset:
        """Warp pixels onto a grid that already exists.

        Args:
            match: Grid to land on, or any xarray object carrying one. Its
                CRS, resolution, and extent are all adopted, so no resolution
                is taken.
            resampling: GDAL resampling kernel.

        Returns:
            New Dataset on the matched grid, carrying its own `spatial_ref`
            and the CF semantics its axes earn.

        Raises:
            ValueError: The raster carries no locatable grid, a variable does
                not span it, or `resampling` would blend a variable whose
                values are class codes.

        Examples:
            >>> dem = srtm.gs.reproject_match(scene, resampling="bilinear")
            >>> dem.gs.geobox == scene.gs.geobox
            True
        """
        return warp.reproject_match(self._data, match, resampling=resampling)

    def crop(self, vector: GeoVector, *, mask: bool = True) -> Dataset:
        """Cut the raster down to a vector's extent.

        Args:
            vector: Geometries to cut against, in this raster's CRS.
            mask: Also make nodata the pixels outside the geometries, which
                then take their own variable's fill value.

        Returns:
            New Dataset covering the vector's extent.

        Raises:
            ValueError: `vector` is in a different CRS, does not overlap the
                raster, or `mask` is set while a variable carries no fill
                value to mark nodata with.
        """
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize
        from shapely import union_all

        if vector.crs != self.crs:
            raise ValueError(
                f"vector is in {vector.crs} but the raster is in {self.crs}; "
                f"reproject the vector before cropping"
            )
        geometry = Geometry(union_all(vector.gdf.geometry.values), crs=self.crs)

        # odc's own apply_mask writes NaN, which promotes every integer variable.
        cut = cast("Dataset", self._data.odc.crop(geometry, apply_mask=False))
        if not mask:
            return cut
        return nodata.mask(cut, rasterize(geometry, cut.gs.geobox))

    def to_array(self, *, dtype: DTypeLike | None = None) -> DataArray:
        """Stack every variable this raster carries into one array.

        Applies no packing and no display scaling — stretching a composite is
        `viz.plot.rgb`'s job, at draw time. Select and order the variables with
        xarray before stacking them.

        Args:
            dtype: Cast applied to the stacked array. None keeps the source
                dtype, which every stacked variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`, carrying this raster's
            own attrs, its coordinates' attrs, and the models every stacked
            variable carries alike. A variable's attrs win a key collision.

        Raises:
            ValueError: This raster carries no variables or already spans the
                stacked axis, the variables carry different non-spatial
                dimensions, they differ in dtype while `dtype` is None, or they
                carry a model field one array cannot hold two of, such as two
                scale factors.

        Warns:
            DroppedAttrsWarning: The variables carry a field differently and
                its model drops rather than refuses the disagreement.

        Examples:
            >>> ds[["ndvi"]].gs.to_array()
            >>> ds[["B04", "B03", "B02"]].gs.to_array()
        """
        stacked = self._stacked(dtype)

        # Stacking drops the variables' own attrs; the rest of the raster's survive it.
        merged = attrs.merge(
            [self._data.variables[name] for name in self.variables],
            action="stacking",
        )
        attrs.rebase(stacked, merged, target=None, inplace=True)
        return cast("DataArray", stacked)

    def plot(
        self,
        variable: str | tuple[str, str, str] | None = None,
        *,
        cmap: str | Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        cols: int = 4,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> hv.Element | hv.Layout:
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
            title: Panel title, above the axes. None leaves hvplot's own.
            xlabel: Caption below the axes, e.g. a place name. None leaves
                hvplot's own.

        Returns:
            Image or RGB element over this raster's grid, or a Layout of one
            panel per `time` value, `cols` wide.

        Raises:
            KeyError: A named variable is absent.
            ValueError: `variable` is None while this raster carries more than
                one variable and measures no red, green, and blue among them,
                two variables measure one colour, or the drawing refuses the
                raster.

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

        absent = [name for name in selected if name not in self._data.data_vars]
        if absent:
            raise KeyError(
                f"{absent} are not data variables of this raster; it carries "
                f"{self.variables}"
            )
        array = self._data[list(selected)].gs.to_array()

        # A composite carries colour rather than class codes, so it draws no legend.
        legend = None
        if len(selected) == 1:
            legend = attrs.AttrsNamespace.from_attrs(
                self._data.variables[selected[0]].attrs
            ).get(attrs.Legend)

        return draw(
            array,
            cmap=cmap,
            clim=clim,
            class_map=legend.class_map if legend else None,
            color_map=legend.color_map if legend else None,
            cols=cols,
            title=title,
            xlabel=xlabel,
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

    def to_tensor(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Stack every variable this raster carries into one model-input tensor.

        Select and order the variables with xarray before stacking them.

        Args:
            dtype: Tensor dtype, which the variables are also stacked in. None
                casts to `torch.float32`, which keeps unsigned imagery off
                `torch.uint16` — a dtype torch accepts and carries no
                arithmetic kernels for. A dtype numpy cannot hold, such as
                `torch.bfloat16`, stacks as float32 and narrows on the way out.

        Returns:
            Tensor shaped `(*axes, band, y, x)`.

        Raises:
            ValueError: This raster carries no variables, or they carry
                different non-spatial dimensions.

        Examples:
            >>> ds[["B04", "B08"]].gs.to_tensor().dtype
            torch.float32
        """
        return tensor(lambda stacking: self.to_numpy(dtype=stacking), dtype)

    def to_cog(
        self,
        destination: str | PathLike[str],
        *,
        layout: Layout | Literal["nested", "flat"] = "nested",
        overwrite: bool = False,
        **options: Unpack[COGWriteOptions],
    ) -> None:
        """Write this raster as a tree of Cloud Optimized GeoTIFFs.

        A GeoTIFF holds one instant of one grid, so a cube spreads across
        files. `layout` decides how: see `utils.io.layout`.

        Args:
            destination: Directory the tree is written into, or the file path
                when the raster is one instant written as one file.
            layout: `"nested"` or `"flat"` on their defaults, or a configured
                `NestedLayout` or `FlatLayout`.
            overwrite: Replace leaves that already exist.
            **options: COG creation options passed to every leaf.

        Raises:
            FileExistsError: A leaf exists and `overwrite` is false.
            KeyError: `layout` names neither `"nested"` nor `"flat"`.
            ValueError: This raster carries no locatable grid, or spans a
                non-spatial axis other than time.

        Examples:
            >>> ds.gs.to_cog("scene")  # scene/20250601T103031/B04.tif, ...
            >>> ds.gs.to_cog("scene", layout=FlatLayout(split_bands=False))
        """
        from geosave_engine.geodata.utils.io.layout import FlatLayout, NestedLayout

        layouts: dict[str, type[Layout]] = {
            "nested": NestedLayout,
            "flat": FlatLayout,
        }
        chosen = layouts[layout]() if isinstance(layout, str) else layout
        chosen.write(self._data, destination, overwrite=overwrite, **options)

    def to_zarr(
        self,
        destination: str | PathLike[str],
        *,
        overwrite: bool = False,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> Path | Delayed:
        """Write this raster to Zarr.

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
        """Write this raster to netCDF.

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

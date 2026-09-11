"""The `gs` xarray Dataset accessor, and the two states a raster is in.

Placement is odc-geo's. GeoSave adds only the CF axis semantics odc omits,
stated by `write_crs`.

Examples:
    Unplaced. No CRS, so `GeoRaster.geobox` refuses and every ground-referenced
    member with it. Tiling and variable access still work:

    >>> png
    <xarray.Dataset> Size: 262kB
    Dimensions:  (y: 512, x: 512)
    Dimensions without coordinates: y, x
    Data variables:
        band     (y, x) uint8 ...

    Placed. odc resolves a GeoBox and names the grid mapping in encoding:

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
    >>> opened.red.encoding["grid_mapping"]
    'spatial_ref'

    `write_crs` states what those axes measure, which odc does not:

    >>> raster.y.attrs
    {'units': 'metre', 'resolution': -10.0, 'crs': 'EPSG:32633',
     'standard_name': 'projection_y_coordinate', 'axis': 'Y'}

    A geographic CRS names the axes `latitude` and `longitude`, following
    odc-geo:

    >>> raster.gs.grid_dims
    ('latitude', 'longitude')
    >>> raster.latitude.attrs["standard_name"], raster.latitude.attrs["units"]
    ('latitude', 'degrees_north')

Outside this: variable naming, dtype, packing, chunking, axis direction, and
time ordering. Variable order carries no meaning either, because Zarr returns
variables and groups alphabetically and netCDF in insertion order.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Unpack, cast, overload

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.errors import DroppedAttrsWarning

from .anchor import GeoAnchor
from .array import (
    BAND_DIMENSION,
    UNPLACED_DIMENSIONS,
    _CRS_COORDINATE,
    _TIME_COORDINATE,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from dask.delayed import Delayed
    from datetime import datetime as dt, timedelta
    from os import PathLike
    from numpy.typing import DTypeLike
    from odc.geo import CRS, BoundingBox, Resolution, SomeCRS, SomeResolution

    import torch

    from geosave_engine.geodata.attrs import AttrsHeader, AttrsModel
    from geosave_engine.geodata.transform.time import EmptyBuckets, Reducer
    from geosave_engine.geodata.utils.datetime import Freq

    from geosave_engine.geodata.utils.io.netcdf import (
        NetCDFEngine,
        NetCDFWriteOptions,
    )
    from geosave_engine.geodata.utils.io.geotiff import COGWriteOptions
    from geosave_engine.geodata.utils.io.layout import Layout
    from geosave_engine.geodata.utils.io.zarr import ZarrWriteOptions
    from geosave_engine.geodata.utils.datetime import DateRange

    from geosave_engine.geodata.attrs import TilingMode

    import holoviews as hv

    from geosave_engine.geodata import DataArray, Dataset
    from .vector import GeoVector

# One variable's pixels, alone or paired with the declared axes it carries.
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
    axes `coords` declares ahead of them. A bare array carries every declared
    axis; pair it with its own names to carry only some.

    Args:
        variables: Data variable name mapped to its pixels, or to a
            `(pixels, axes)` pair naming the declared axes it carries.
        geobox: Grid placing the trailing two axes. None leaves the raster
            unplaced, so it holds pixels without claiming ground position.
        nodata: Value standing for absent pixels, declared on every variable
            as `Packing.fill_value`. None declares none.
        **coords: Leading axis name mapped to its labels, in array order. None
            labels an axis carrying none. Pass a name Python reserves as
            `**{"class": labels}`.

    Returns:
        Placed Dataset when `geobox` is given, its spatial coordinates naming
        what they measure, otherwise an unplaced Dataset whose spatial dims
        are named `y` and `x`.

    Raises:
        ValueError: `variables` is empty, a variable names an undeclared axis,
            its rank does not match the axes it carries, its trailing axes do
            not match `geobox`, two variables size one axis differently, an
            axis is labelled with the wrong number of values, an axis no
            variable carries is declared, or an axis name collides with a
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

    spatial_dims = UNPLACED_DIMENSIONS if geobox is None else geobox.dimensions
    grid_coords = (*spatial_dims, _CRS_COORDINATE)
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
                f"{name!r} lies along {unknown}, which no keyword declares; this "
                f"raster declares {list(raster_axes)}"
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
            f"{empty_axes} are declared but no variable lies along them; drop them "
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
class GeoRaster:
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
        self._data: Dataset = cast("Dataset", data)

    @property
    def geobox(self) -> GeoBox:
        """Read the pixel grid, which places this raster on the ground.

        Every ground-referenced member reads the grid through here, so a
        raster carrying no CRS refuses them all rather than answering in
        pixel coordinates that mean nothing.

        Returns:
            Grid including CRS, transform, bounds, and shape.

        Raises:
            ValueError: The Dataset declares no CRS, carries no spatial dims,
                or describes its grid by ground control points rather than a
                transform.
        """
        grid = self._data.odc.geobox
        if grid is None:
            raise ValueError(
                "Dataset carries no locatable grid; it declares no CRS or spatial dims"
            )
        if not isinstance(grid, GeoBox):
            raise ValueError(
                f"Dataset grid is a {type(grid).__name__}; expected a regular GeoBox"
            )
        if grid.crs is None:
            raise ValueError(
                "Dataset grid declares no CRS, so it is indexed in pixels rather than placed on "
                "the ground; assign one with odc.geo.xr.assign_crs before asking where it is"
            )
        return grid

    @property
    def crs(self) -> CRS:
        """Read the coordinate reference system.

        Returns:
            CRS declared by the grid mapping coordinate.

        Raises:
            ValueError: The Dataset declares no CRS.
        """
        return cast("CRS", self.geobox.crs)  # geobox already refused a CRS-less grid

    @property
    def bounds(self) -> BoundingBox:
        """Read the grid's extent in its own CRS.

        Returns:
            Bounding box covering every pixel.

        Raises:
            ValueError: The Dataset carries no locatable grid.
        """
        return self.geobox.boundingbox

    @property
    def resolution(self) -> Resolution:
        """Read the pixel size in CRS units.

        Returns:
            Signed resolution along x and y.

        Raises:
            ValueError: The Dataset carries no locatable grid.
        """
        return self.geobox.resolution

    @property
    def grid_dims(self) -> tuple[str, str]:
        """Name the two dimensions the grid spans.

        Returns:
            `("y", "x")` for a projected CRS, `("latitude", "longitude")` for a
            geographic one, and `("y", "x")` for an unplaced raster, whose
            pixels span those axes without claiming ground position.
        """
        grid = self._data.odc.geobox
        return UNPLACED_DIMENSIONS if grid is None else grid.dimensions

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
        if _TIME_COORDINATE not in self._data.coords:
            return None
        return self._data.variables[_TIME_COORDINATE].values

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
            self._data.variables[_TIME_COORDINATE].attrs
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

    @property
    def attrs(self) -> AttrsHeader:
        """Read the typed attrs this Dataset carries.

        Returns:
            Detached header, reread on every access because xarray attrs are
            mutable in place.
        """
        return attrs.read(self._data)

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
        """State what this raster's spatial coordinates measure.

        The grid decides the answer: a projected raster states
        `projection_y_coordinate` and `projection_x_coordinate`, a geographic
        one `latitude` and `longitude`. No pixel moves; `reproject` does that.

        Args:
            crs: CRS the existing coordinates are in, for a raster that
                declares none or declares the wrong one. None reads the CRS
                the raster already carries.

        Returns:
            New Dataset whose spatial coordinates state the standard name,
            units, and axis their grid gives them.

        Raises:
            ValueError: The raster carries no locatable grid and `crs` names
                none, or `crs` names no known reference system.

        Examples:
            >>> stated = ds.gs.write_crs("EPSG:32633")
            >>> stated.y.attrs["standard_name"], stated.y.attrs["axis"]
            ('projection_y_coordinate', 'Y')
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
        """Declare the stored value standing for absent pixels.

        `Packing` mirrors the value across both `_FillValue` and odc's own
        `nodata`, so no reader can find the two naming different pixels.

        Args:
            value: Stored value marking absence. None clears the declaration.
            target: Variable names to write, defaulting to every data variable.

        Returns:
            New Dataset whose named variables declare `value` as their fill.

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

        # Packing owns both spellings of a declared fill, and mirrors them.
        return attrs.rebase(
            self._data, attrs.Packing(_FillValue=value), target=var_names
        )

    def crop(self, vector: GeoVector, *, mask: bool = True) -> Dataset:
        """Cut the raster down to a vector's extent.

        Args:
            vector: Geometries to cut against, in this raster's CRS.
            mask: Also set pixels outside the geometries to nodata.

        Returns:
            New Dataset covering the vector's extent.

        Raises:
            ValueError: `vector` is in a different CRS, or does not overlap
                the raster.
        """
        from odc.geo.geom import Geometry
        from shapely import union_all

        if vector.crs != self.crs:
            raise ValueError(
                f"vector is in {vector.crs} but the raster is in {self.crs}; "
                f"reproject the vector before cropping"
            )
        geometry = Geometry(union_all(vector.gdf.geometry.values), crs=self.crs)
        return cast("Dataset", self._data.odc.crop(geometry, apply_mask=mask))

    def reproject(
        self,
        how: SomeCRS | GeoBox,
        *,
        resampling: str = "nearest",
        resolution: SomeResolution | None = None,
    ) -> Dataset:
        """Warp this raster onto another CRS or an exact target grid.

        Args:
            how: Target CRS, or the exact GeoBox to land on.
            resampling: GDAL resampling kernel, e.g. `"nearest"`, `"bilinear"`.
            resolution: Output pixel size when `how` is a CRS. None keeps the
                source resolution.

        Returns:
            New Dataset on the target grid, carrying its own `spatial_ref`.

        Raises:
            ValueError: This raster carries no locatable grid, `resolution` is
                given alongside a GeoBox, or `resampling` would blend a
                categorical variable.

        Examples:
            >>> ds.gs.reproject("EPSG:4326").gs.crs
            CRS('EPSG:4326')
        """
        from geosave_engine.geodata.transform.grid import reproject

        return cast(
            "Dataset",
            reproject(self._data, how, resampling=resampling, resolution=resolution),
        )

    def resample(
        self,
        freq: Freq,
        method: Reducer = "median",
        *,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | dt = "start_day",
        offset: str | timedelta | None = None,
    ) -> Dataset:
        """Collapse this raster's time axis onto coarser buckets.

        A bucket covering no observation is dropped off the axis, with a
        `DroppedBucketsWarning`; call `.gs.interpolate` afterward to fill it
        back in.

        Args:
            freq: Target cadence — any pandas offset alias. The alias fixes
                where the bucket edges fall: `"MS"` on month starts, `"ME"` on
                month ends, `"5D"` every five days from `origin`.
            method: Named reducer each bucket collapses with.
            closed: Which of a bucket's two edges is inclusive. None takes the
                alias default.
            label: Which edge names the bucket in the `time` coordinate. None
                takes the alias default. It never moves the edges.
            origin: Timestamp the edge grid is phased from, or a strategy name
                such as `"start_day"`.
            offset: Shift added on top of `origin`.

        Returns:
            New Dataset on the bucketed time axis, carrying a `time_bnds`
            coordinate and a `TimeSpec` recording the resample.

        Raises:
            ValueError: This raster carries no `time` coordinate, already
                records a resample, pandas does not know `freq`, `freq` buckets
                more finely than the axis observes, or `method` would blend a
                categorical variable.

        Examples:
            >>> daily.gs.resample("MS").gs.timespan
            (datetime(2024, 1, 1, 0, 0), datetime(2024, 3, 31, 23, 59, 59, 999999))
        """
        from geosave_engine.geodata.transform.time import resample

        return cast(
            "Dataset",
            resample(
                self._data,
                freq,
                method,
                closed=closed,
                label=label,
                origin=origin,
                offset=offset,
            ),
        )

    def interpolate(self, empty: EmptyBuckets) -> Dataset:
        """Fill this bucketed raster's missing calendar positions.

        Args:
            empty: What a missing bucket becomes. `"keep"` emits it as
                nodata, `"nearest"` repeats the nearest observed bucket's
                values, and `"linear"` interpolates between neighbours.

        Returns:
            New Dataset on the full bucket grid this raster's own cadence
            covers, carrying a recomputed `time_bnds` coordinate.

        Raises:
            ValueError: This raster carries no `TimeSpec`, or `empty='linear'`
                would interpolate a categorical variable.

        Examples:
            >>> monthly.gs.interpolate("nearest").gs.timespan == monthly.gs.timespan
            True
        """
        from geosave_engine.geodata.transform.time import interpolate

        return cast("Dataset", interpolate(self._data, empty))

    def rename_vars(self, mapping: Mapping[str, str]) -> Dataset:
        """Rename data variables without changing their order or pixels.

        Args:
            mapping: Existing variable names mapped to replacement names.

        Returns:
            New Dataset with renamed data variables, each keeping its own
            attrs.

        Raises:
            KeyError: A source variable is absent.
            ValueError: A replacement is empty, names a coordinate, or creates
                a duplicate.

        Examples:
            >>> ds.gs.rename_vars({"B04": "red"}).gs.variables
            ('red', 'B08')
        """
        from geosave_engine.geodata.transform.variables import rename

        return cast("Dataset", rename(self._data, mapping))

    def tile(
        self,
        shape: tuple[int, int],
        *,
        group_id: str,
        overlap: int | float | tuple[int, int] = 0,
        mode: TilingMode = "reflect",
    ) -> list[Dataset]:
        """Cut this raster into equally shaped tiles.

        The trailing edges are padded rather than truncated, so every tile
        matches `shape` and a model reading them sees one input spec. Tiles
        read as ordinary rasters and stay lazy until their pixels are needed.

        Args:
            shape: Tile height and width in pixels.
            group_id: Identifier every tile of this operation shares, used to
                route tiles back to this raster when stitching.
            overlap: Shared pixels as a count, a fraction in `[0, 1)`, or
                row-column counts.
            mode: How the trailing edges are padded. `"constant"` extends every
                variable with its own declared fill value.

        Returns:
            Tiles in row-major order, each stamped with its own `Tiling`
            and placed when this raster is placed.

        Raises:
            ValueError: This raster names no spatial dimensions, `shape`
                exceeds its own shape in either axis, or `mode` is
                `"constant"` while a variable declares no fill value.

        Examples:
            >>> tiles = ds.gs.tile((256, 256), group_id="scene-001", overlap=32)
            >>> len(tiles), tiles[3].gs.attrs.root.get("tiling").tile_index
            (16, 3)
        """
        from geosave_engine.geodata.transform.tiles import tile

        return cast(
            "list[Dataset]",
            tile(self._data, shape, group_id=group_id, overlap=overlap, mode=mode),
        )

    def time_window(
        self,
        slot: int,
        *,
        stride: int | None = None,
    ) -> list[Dataset]:
        """Cut this raster into fixed-length windows along time.

        A window names no reference bucket — read one off its own `time`
        values afterward (`[0]`/`[-1]`/middle for a left/right/middle
        convention, matching `TimeSpec.time_label`'s own vocabulary).

        Args:
            slot: Window length, in buckets.
            stride: Buckets between consecutive window starts. None defaults
                to `slot`, giving non-overlapping, back-to-back windows.

        Returns:
            Windows in walk order, each on `slot` buckets of this raster's
            own time axis, unchanged otherwise.

        Raises:
            ValueError: This raster carries no `TimeSpec`, its time axis is
                not calendar-contiguous at its own recorded cadence, or
                `slot` exceeds the axis length.

        Examples:
            >>> windows = monthly.gs.time_window(3, stride=1)
            >>> len(windows), windows[0].sizes["time"]
            (10, 3)
        """
        from geosave_engine.geodata.transform.time import time_window

        return cast("list[Dataset]", time_window(self._data, slot, stride=stride))

    def to_array(
        self,
        variables: Sequence[str] | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> DataArray:
        """Stack this raster's variables into one array.

        Applies no packing and no display scaling — stretching a composite is
        `viz.plot.rgb`'s job, at draw time.

        Args:
            variables: Variables to stack, in order. None reads every one.
            dtype: Cast applied to the stacked array. None keeps the source
                dtype, which every stacked variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`, carrying this raster's
            own attrs, its coordinates' attrs, and the models every stacked
            variable states alike. A variable's attrs win a key collision.

        Raises:
            KeyError: A named variable is absent.
            ValueError: `variables` is empty, this raster already spans the
                stacked axis, the variables carry different non-spatial
                dimensions, they differ in dtype while `dtype` is None, or they
                state a model field one array cannot hold two of, such as two
                scale factors.

        Warns:
            DroppedAttrsWarning: The variables state a field differently and
                its model drops rather than refuses the disagreement.

        Examples:
            >>> ds.gs.to_array(["ndvi"])
            >>> ds.gs.to_array(("B04", "B03", "B02"))
        """
        var_names = self.variables if variables is None else tuple(variables)
        stacked = self._stacked(var_names, dtype)

        # Stacking drops the variables' own attrs; the rest of the raster's survive it.
        merged, dropped = attrs.AttrsNamespace.combine(
            [
                attrs.AttrsNamespace.from_attrs(self._data.variables[name].attrs)
                for name in var_names
            ]
        )
        if dropped:
            warnings.warn(
                f"stacking {list(var_names)} drops attrs they did not state alike: "
                f"{sorted(dropped)}",
                DroppedAttrsWarning,
                stacklevel=2,
            )
        stacked.attrs = {**stacked.attrs, **merged.to_attrs()}
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

        Three names give a composite, one gives an Image, and None reads
        `RenderHints` before falling back to a sole variable. A `Legend` the
        variable declares supplies the class and colour maps.

        Args:
            variable: One data variable, three ordered red, green, blue, or
                None to read what this raster declares.
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
            ValueError: `variable` is None while this raster declares no
                `RenderHints.rgb_variables` and carries more than one variable,
                or the drawing itself refuses the raster.

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
            hint = self.attrs.root.get(attrs.RenderHints)
            if hint is None or hint.rgb_variables is None:
                raise ValueError(
                    f"this raster carries {self.variables} and declares no "
                    f"RenderHints.rgb_variables, so it names nothing to draw; "
                    f"pass one variable, three for a composite, or declare "
                    f"them with ds.gs.rebase(RenderHints(rgb_variables=(...)))"
                )
            selected = tuple(hint.rgb_variables)

        array = self.to_array(selected)

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

    def to_numpy(
        self,
        variables: Sequence[str] | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> np.ndarray:
        """Stack this raster's variables into one model-input array.

        Every axis besides the grid leads, in this raster's own order, then the
        variables, then the grid pair — torchgeo's `[T, C, H, W]`. No batch axis
        is added, so one raster is one sample.

        Args:
            variables: Variables to read, in the order the model expects.
                None reads every variable in Dataset order.
            dtype: Cast applied to the stacked array. None keeps the source
                dtype, which every selected variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`.

        Raises:
            KeyError: A named variable is absent.
            ValueError: `variables` is empty, the selected variables carry
                different non-spatial dimensions, or they differ in dtype while
                `dtype` is None.

        Examples:
            >>> ds.gs.to_numpy(("B04", "B08")).shape
            (2, 4, 256, 256)
        """
        var_names = self.variables if variables is None else tuple(variables)
        return self._stacked(var_names, dtype).values

    def _stacked(
        self, var_names: tuple[str, ...], dtype: DTypeLike | None
    ) -> xr.DataArray:
        """Stack the named variables onto the band axis.

        Args:
            var_names: Data variable names, in stacking order.
            dtype: Cast applied to the stack. None keeps the source dtype,
                which every named variable must then share.

        Returns:
            Array shaped `(*axes, band, y, x)`, carrying no attrs.

        Raises:
            KeyError: A named variable is absent.
            ValueError: `var_names` is empty, this raster already spans the band
                axis, a variable does not span the grid, the variables carry
                different axes, or they differ in dtype while `dtype` is None.
        """
        if not var_names:
            raise ValueError(
                f"no variables to stack; name at least one of "
                f"{list(self.variables)}, or pass None for every one"
            )

        absent = [name for name in var_names if name not in self._data.data_vars]
        if absent:
            raise KeyError(
                f"{absent} are not data variables of this raster; it carries "
                f"{self.variables}"
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

        # torchgeo returns [T, C, H, W], so the axes lead and the grid trails.
        stacked = (
            self._data[list(var_names)]
            .to_array(dim=BAND_DIMENSION)
            .transpose(*axes[var_names[0]], BAND_DIMENSION, *grid_dims)
        )
        return stacked if dtype is None else stacked.astype(dtype)

    def to_tensor(
        self,
        variables: Sequence[str] | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Stack this raster's variables into one model-input tensor.

        Args:
            variables: Variables to read, in the order the model expects.
                None reads every variable in Dataset order.
            dtype: Tensor dtype, which the variables are also stacked in. None
                casts to `torch.float32`, which keeps unsigned imagery off
                `torch.uint16` — a dtype torch accepts and carries no
                arithmetic kernels for. A dtype numpy cannot hold, such as
                `torch.bfloat16`, stacks as float32 and narrows on the way out.

        Returns:
            Tensor shaped `(*axes, band, y, x)`.

        Raises:
            KeyError: A named variable is absent.
            ValueError: `variables` is empty, or the selected variables carry
                different non-spatial dimensions.

        Examples:
            >>> ds.gs.to_tensor(("B04", "B08")).dtype
            torch.float32
        """
        import torch

        target = dtype or torch.float32
        try:
            stacking_dtype = torch.empty(0, dtype=target).numpy().dtype
        except TypeError:
            stacking_dtype = np.dtype(np.float32)

        stacked = np.ascontiguousarray(self.to_numpy(variables, dtype=stacking_dtype))
        return torch.as_tensor(stacked, dtype=target)

    def to_cog(
        self,
        target: str | PathLike[str],
        *,
        layout: Layout | Literal["nested", "flat"] = "nested",
        overwrite: bool = False,
        **options: Unpack[COGWriteOptions],
    ) -> None:
        """Write this raster as a tree of Cloud Optimized GeoTIFFs.

        A GeoTIFF holds one instant of one grid, so a cube spreads across
        files. `layout` decides how: see `utils.io.layout`.

        Args:
            target: Directory the tree is written into, or the file path when
                the raster is one instant written as one file.
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
            >>> ds.gs.to_cog("scene")
            >>> ds.gs.to_cog("scene", layout=FlatLayout(split_bands=False))
        """
        from geosave_engine.geodata.utils.io.layout import FlatLayout, NestedLayout

        layouts: dict[str, type[Layout]] = {
            "nested": NestedLayout,
            "flat": FlatLayout,
        }
        chosen = layouts[layout]() if isinstance(layout, str) else layout
        chosen.write(self._data, target, overwrite=overwrite, **options)

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

"""Transformations of the pixel grid."""

from __future__ import annotations

import operator
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.attrs import combine, stamp
from geosave_engine.geodata.core.raster import GeoRaster

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from odc.geo import SomeCRS, SomeResolution

# Kernels that carry an observed value across rather than blending several.
_CATEGORICAL_RESAMPLING = frozenset({"nearest", "mode"})

_FILL_VALUE_ATTRIBUTE = "_FillValue"
_ODC_NODATA_ATTRIBUTE = "nodata"


def _declared_fill(array: xr.DataArray) -> float | int | None:
    """Read the fill value a variable declares, keeping its stored dtype.

    `odc.nodata` coerces the value to float, which promotes an integer raster
    wherever the result is filled, so the declaration is read as stored.

    Args:
        array: Data variable that may declare absence.

    Returns:
        The declared value, or None when the variable declares none.
    """
    declared = array.attrs.get(_FILL_VALUE_ATTRIBUTE)
    return declared if declared is not None else array.attrs.get(_ODC_NODATA_ATTRIBUTE)


def _shared_fill(arrays: Sequence[xr.DataArray], name: str) -> float | int:
    """Return the one fill value every raster declares for a variable.

    Args:
        arrays: One raster's copy of the variable, per raster being mosaicked.
        name: Variable name, used only to name it in the error.

    Returns:
        Shared fill value.

    Raises:
        ValueError: The rasters declare more than one fill value, or none, so
            a pixel no raster covers has nothing to hold.
    """
    declared = [_declared_fill(array) for array in arrays]
    first, *rest = declared
    # Two declared NaNs are the same fill, and `nan != nan` would read them apart.
    if any(
        not (value == first or (value != value and first != first)) for value in rest
    ):
        raise ValueError(
            f"rasters declare different fill values for {name!r} "
            f"({sorted(set(declared), key=str)}); a mosaic resolves absence to one value"
        )
    if first is None:
        raise ValueError(
            f"{name!r} declares no fill value, so a pixel no raster covers has "
            f"nothing to hold; declare one with Packing(fill_value=...)"
        )
    return first


def _require_one_dtype(arrays: Sequence[xr.DataArray], name: str) -> None:
    """Refuse a variable the rasters store in more than one dtype.

    Args:
        arrays: One raster's copy of the variable, per raster being mosaicked.
        name: Variable name, used only to name it in the error.

    Raises:
        ValueError: The rasters store the variable in different dtypes, which
            laying them onto one grid would promote.
    """
    dtypes = {array.dtype for array in arrays}
    if len(dtypes) > 1:
        raise ValueError(
            f"rasters carry different dtypes for {name!r} "
            f"({sorted(str(dtype) for dtype in dtypes)}); laying them onto one grid "
            f"would promote them"
        )


def _require_one_coord(arrays: Sequence[xr.DataArray], dim: str) -> None:
    """Refuse rasters that label a non-spatial axis differently.

    Args:
        arrays: One raster's coordinate along `dim`, per raster.
        dim: Coordinate name, used only to name it in the error.

    Raises:
        ValueError: A raster labels `dim` differently from the first, so the
            mosaicked result would have no single axis.
    """
    first, *rest = arrays
    drifted = [
        index for index, array in enumerate(rest, start=1) if not array.equals(first)
    ]
    if drifted:
        raise ValueError(
            f"rasters {drifted} label {dim!r} differently from raster 0; align "
            f"them onto one axis before mosaicking"
        )


def _require_one_crs(grids: Sequence[GeoBox]) -> None:
    """Refuse grids that do not share one CRS.

    odc.geo's own intersection/union raise on a CRS mismatch too, but wraps
    it in a message that reads as an empty-input error; naming the CRSs here
    keeps the actual cause from getting lost in that wrapping.

    Args:
        grids: Geoboxes to check.

    Raises:
        ValueError: `grids` carries more than one CRS.
    """
    named: set[str] = set()
    for grid in grids:
        if grid.crs is None:
            named.add("no CRS")
        elif grid.crs.epsg:
            named.add(f"EPSG:{grid.crs.epsg}")
        else:
            named.add(str(grid.crs))

    crs = sorted(named)
    if len(crs) > 1:
        raise ValueError(f"the rasters carry {crs}; reproject them onto one CRS first")


def align(rasters: Sequence[xr.Dataset]) -> list[xr.Dataset]:
    """Cut rasters down to the grid cells they all share.

    Pixels are sliced, never interpolated, so grids that are not already on
    one pixel phase refuse instead of being snapped onto one.

    Args:
        rasters: Datasets on one CRS, resolution, and pixel phase, at least
            one.

    Returns:
        The rasters in call order, each cut to their shared extent.

    Raises:
        ValueError: `rasters` is empty, a raster carries no locatable grid,
            the grids differ in CRS, resolution, or pixel phase, or they share
            no cell.

    Examples:
        >>> north, south = transform.grid.align([north, south])
        >>> north.gs.geobox == south.gs.geobox
        True
    """
    if not rasters:
        raise ValueError("aligning needs at least one raster")

    grids = [GeoRaster(raster).geobox for raster in rasters]
    _require_one_crs(grids)
    try:
        shared = reduce(operator.and_, grids)
    except ValueError as error:
        raise ValueError(
            f"the rasters are not on one pixel grid, so they cannot be cut to a "
            f"shared extent; reproject them onto one grid first ({error})"
        ) from error
    if shared.is_empty():
        raise ValueError(
            f"the rasters share no grid cell; their extents are "
            f"{[grid.boundingbox for grid in grids]}"
        )
    return [
        raster.isel(dict(zip(grid.dimensions, grid.overlap_roi(shared), strict=True)))
        for raster, grid in zip(rasters, grids, strict=True)
    ]


def reproject(
    raster: xr.Dataset,
    how: SomeCRS | GeoBox,
    *,
    resampling: str = "nearest",
    resolution: SomeResolution | None = None,
) -> xr.Dataset:
    """Warp the raster onto another CRS or an exact target grid.

    Args:
        raster: Dataset carrying a locatable grid.
        how: Target CRS, or the exact GeoBox to land on.
        resampling: GDAL resampling kernel, e.g. `"nearest"`, `"bilinear"`.
        resolution: Output pixel size when `how` is a CRS. None keeps the
            source resolution.

    Returns:
        New Dataset on the target grid, carrying its own `spatial_ref`.

    Raises:
        ValueError: `raster` carries no locatable grid, `resolution` is given
            alongside a GeoBox, or `resampling` would blend a categorical
            variable.

    Examples:
        >>> transform.grid.reproject(utm, "EPSG:4326").gs.crs
        CRS('EPSG:4326')
    """
    source = GeoRaster(raster)
    source.geobox  # noqa: B018 — refuses a raster with nothing to reproject
    if resolution is not None and isinstance(how, GeoBox):
        raise ValueError(
            "resolution applies only when reprojecting onto a CRS; the supplied "
            "GeoBox already fixes the output pixel size"
        )

    categorical = sorted(source.categorical)
    if categorical and resampling not in _CATEGORICAL_RESAMPLING:
        raise ValueError(
            f"{categorical} carry a class map, so {resampling!r} would blend "
            f"class values into ones that mean nothing; pick one of "
            f"{sorted(_CATEGORICAL_RESAMPLING)} or drop them before warping"
        )

    options = {} if resolution is None else {"resolution": resolution}
    warped = raster.odc.reproject(how, resampling=resampling, **options)

    # odc's Dataset accessor leaves the source WKT on spatial_ref, so the CRS is restated.
    return warped.gs.write_crs(how.crs if isinstance(how, GeoBox) else how)


def _is_fill(data: xr.DataArray, fill: float | int) -> xr.DataArray:
    """Elementwise mask of pixels holding a variable's declared fill.

    `nan == nan` is never true, so a NaN fill (the only value unequal to
    itself) is matched through that instead, alongside plain equality.

    Args:
        data: Array to test.
        fill: Declared fill value a pixel may hold.

    Returns:
        Boolean array, True where a pixel holds `fill`.
    """
    # A NaN fill is the only value unequal to itself, so it matches through that.
    return data != data if fill != fill else data == fill


class _MosaicPlan(NamedTuple):
    """What every raster of one mosaic must agree on.

    Args:
        union: Grid covering every raster's extent.
        names: Data variable names every raster carries, in order.
        fills: Value marking absence, per variable.
        spatial: Names of the two spatial dims the rasters carry.
    """

    union: GeoBox
    names: tuple[str, ...]
    fills: dict[str, float | int]
    spatial: tuple[str, str]


def _plan(rasters: Sequence[xr.Dataset]) -> _MosaicPlan:
    """Read the one grid, variable set, and fill value a mosaic needs.

    Args:
        rasters: Datasets to lay together.

    Returns:
        What the rasters agree on, for either mosaic to place them by.

    Raises:
        ValueError: `rasters` is empty, a raster carries no locatable grid, the
            grids differ in CRS, resolution, or pixel phase, the rasters carry
            different variables, dtypes, or non-spatial coordinates, or a
            variable declares no fill value.
    """
    if not rasters:
        raise ValueError("mosaicking needs at least one raster")

    grids = [GeoRaster(raster).geobox for raster in rasters]
    _require_one_crs(grids)
    try:
        union = reduce(operator.or_, grids)
    except ValueError as error:
        raise ValueError(
            f"the rasters are not on one pixel grid, so they cannot be laid onto "
            f"a shared one; reproject them onto one grid first ({error})"
        ) from error

    names = GeoRaster(rasters[0]).variables
    mismatched = sorted(
        str(index)
        for index, raster in enumerate(rasters)
        if GeoRaster(raster).variables != names
    )
    if mismatched:
        raise ValueError(
            f"rasters {mismatched} carry different variables from {names}; a "
            f"mosaic lays one set of variables onto a wider grid"
        )

    fills: dict[str, float | int] = {}
    for name in names:
        arrays = [raster[name] for raster in rasters]
        fills[name] = _shared_fill(arrays, name)
        _require_one_dtype(arrays, name)

    spatial = union.dimensions
    for dim in (str(name) for name in rasters[0][names[0]].dims if name not in spatial):
        if dim in rasters[0].coords:
            _require_one_coord([raster.coords[dim] for raster in rasters], dim)

    return _MosaicPlan(union, names, fills, spatial)


def mosaic(rasters: Sequence[xr.Dataset]) -> xr.Dataset:
    """Lay rasters side by side onto the grid they jointly cover.

    Pixels are placed, never interpolated, and each raster fills only the
    pixels no earlier one supplied, so an earlier raster wins an overlap. Every
    raster is held at the union's size, so `mosaic_to_zarr` suits wider extents.

    Args:
        rasters: Datasets on one CRS, resolution, and pixel phase, carrying the
            same variables and dtypes, and each declaring a fill value.

    Returns:
        New Dataset covering every raster's extent, holding the fill value
        wherever no raster supplied a pixel.

    Raises:
        ValueError: `rasters` is empty, a raster carries no locatable grid, the
            grids differ in CRS, resolution, or pixel phase, the rasters carry
            different variables, dtypes, or non-spatial coordinates, or a
            variable declares no fill value.

    Examples:
        >>> covered = transform.grid.mosaic([north, south])
        >>> covered.gs.geobox.shape
        Shape2d(x=512, y=1024)
    """

    plan = _plan(rasters)
    rows, columns = plan.spatial
    placed = dict(xr_coords(plan.union))

    grown = (
        raster.reindex(
            {rows: placed[rows], columns: placed[columns]}, fill_value=plan.fills
        )
        for raster in rasters
    )
    # combine_first keys on NaN, not a declared fill, so absence is read through _is_fill instead.
    covered = reduce(
        lambda first, second: first.where(
            xr.Dataset(
                {
                    name: ~_is_fill(first[name], fill)
                    for name, fill in plan.fills.items()
                }
            ),
            second,
        ),
        grown,
    )
    return stamp(covered, combine(rasters)).gs.write_crs()


def mosaic_to_zarr(
    rasters: Sequence[xr.Dataset],
    destination: str | PathLike[str],
    *,
    overwrite: bool = False,
    chunks: tuple[int, int] | None = None,
) -> Path:
    """Lay rasters onto one Zarr store covering the grid they jointly cover.

    Each raster is written into its own region of the store, so a mosaic costs
    one write per raster rather than one graph over the whole extent. Pixels no
    raster covers are never written and read back as the fill value.

    Args:
        rasters: Datasets on one CRS, resolution, and pixel phase, carrying the
            same variables and dtypes, and each declaring a fill value.
        destination: Output path ending in `.zarr`.
        overwrite: Replace an existing destination when true.
        chunks: Store chunking over the spatial axes. None chunks by the first
            raster's own shape, which lands each raster on chunk boundaries.

    Returns:
        Written destination path.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        ValueError: `rasters` is empty, a raster carries no locatable grid, the
            grids differ in CRS, resolution, or pixel phase, the rasters carry
            different variables, dtypes, or non-spatial coordinates, or a
            variable declares no fill value.

    Examples:
        >>> transform.grid.mosaic_to_zarr(scenes, "region.zarr")
        PosixPath('region.zarr')
    """
    import dask.array as da

    from geosave_engine.geodata.utils.io import zarr

    plan = _plan(rasters)
    rows, columns = plan.spatial
    reference = rasters[0]
    leading = [
        str(dim) for dim in reference[plan.names[0]].dims if dim not in plan.spatial
    ]
    block = chunks or (reference.sizes[rows], reference.sizes[columns])
    leading_shape = tuple(reference.sizes[dim] for dim in leading)

    # The store is created empty, so a pixel no raster covers is never written.
    shape = tuple(int(size) for size in plan.union.shape)
    skeleton = xr.Dataset(
        {
            name: (
                (*leading, rows, columns),
                da.full(
                    (*leading_shape, *shape),
                    plan.fills[name],
                    dtype=reference[name].dtype,
                    chunks=(*leading_shape, *block),
                ),
            )
            for name in plan.names
        },
        coords={
            **{
                dim: reference.coords[dim] for dim in leading if dim in reference.coords
            },
            **dict(xr_coords(plan.union)),
        },
    )
    zarr.write(
        stamp(skeleton, combine(rasters)).gs.write_crs(),
        destination,
        overwrite=overwrite,
        compute=False,
    )
    path = Path(destination)

    for raster in rasters:
        window = {rows: raster.coords[rows], columns: raster.coords[columns]}
        # A region write replaces it, so what already landed is read back first.
        standing = zarr.read(path).sel(window)
        absent = xr.Dataset(
            {name: _is_fill(standing[name], plan.fills[name]) for name in plan.names},
            coords=standing.coords,
        )
        merged = xr.where(absent, raster, standing, keep_attrs=False)
        merged = merged.drop_vars(
            [name for name in merged.coords if name not in (*leading, rows, columns)]
        )
        # One block per region, so no two of them ever share a store chunk.
        merged.chunk({rows: -1, columns: -1}).to_zarr(
            path, region="auto", consolidated=False
        )

    return path

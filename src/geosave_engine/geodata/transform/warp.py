"""Moving a raster's pixels onto another grid.

`odc.stac.load` warps at load time, given a geobox. This is the same
operation for a raster already in memory, which has no load to hook into.

Examples:
    A delivered DEM has to reach the imagery's grid before a model reads both,
    and a quicklook wants a CRS rather than an exact grid::

        dem = reproject(srtm, scene, resampling="bilinear")
        webmap = reproject(scene, "EPSG:3857")

    Several rasters reach one grid together, which is the only way to say what
    ground they jointly cover::

        optical, radar, dem = align([optical, radar, srtm], extent="intersection")
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, lcm
from typing import TYPE_CHECKING, Literal, cast, get_args

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.crs import norm_crs
from odc.geo.geobox import (
    GeoBox,
    geobox_intersection_conservative,
    geobox_union_conservative,
)

import geosave_engine.geodata.attrs as attrs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo import SomeCRS, SomeResolution

# Kernel names odc forwards to rasterio.warp.Resampling.
type Resampling = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "gauss",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
    "rms",
]

# What names the grid to warp onto, directly or by naming only its CRS.
type Target = GeoBox | xr.DataArray | xr.Dataset | xr.DataTree | SomeCRS

# Ground a set of aligned rasters covers.
type Extent = Literal["union", "intersection", "match"]

# Asking for a raster's own pixel size rather than the one the target names.
type Native = Literal["native"]
NATIVE: Native = "native"

# Kernels that carry a stored value across rather than blending several.
_VALUE_PRESERVING = frozenset({"nearest", "mode"})

# Reprojection arithmetic drifts a pixel size by a hair, which is not a new one.
_RATIO_TOLERANCE = 1e-6


def _zoom_ratio(source: GeoBox, target: GeoBox) -> int:
    """Count how many of a target's pixels span one of the source's.

    Args:
        source: Grid whose own pixel size is kept.
        target: Lattice it nests onto, which must be at least as fine.

    Returns:
        Whole number of target pixels across one source pixel, 1 where the two
        already agree.

    Raises:
        ValueError: `target` is the coarser of the two, or the sizes stand in
            no whole-numbered ratio.

    Examples:
        >>> _zoom_ratio(sixty_metre, ten_metre)
        6
    """
    along_x = abs(source.resolution.x) / abs(target.resolution.x)
    along_y = abs(source.resolution.y) / abs(target.resolution.y)
    ratio = round(along_x)
    if ratio < 1:
        raise ValueError(
            f"a pixel of {source.resolution} spans {along_x:.4g} of one of "
            f"{target.resolution}, so keeping it would refine that lattice; nest "
            f"onto a grid at least as fine, or take one resolution for all"
        )
    if not isclose(along_x, ratio, rel_tol=_RATIO_TOLERANCE) or not isclose(
        along_y, ratio, rel_tol=_RATIO_TOLERANCE
    ):
        raise ValueError(
            f"a pixel of {source.resolution} spans {along_x:.4g} by {along_y:.4g} "
            f"of {target.resolution}, which no whole number of pixels nests on; "
            f"take one resolution for all instead"
        )
    return ratio


def _variable_resampling(
    resampling: Resampling | Mapping[str, Resampling], variable: str
) -> Resampling:
    """Read the kernel one variable warps with.

    Args:
        resampling: One kernel for every variable, or a mapping naming each
            variable's own, which `"*"` answers the rest of.
        variable: Variable to read the kernel for, group-qualified as
            `"<group>/<variable>"` inside a stack.

    Returns:
        The kernel that variable warps with.

    Raises:
        KeyError: A mapping names neither `variable` nor `"*"`.
    """
    if not isinstance(resampling, Mapping):
        return resampling
    if variable in resampling:
        return resampling[variable]
    if "*" in resampling:
        return resampling["*"]
    raise KeyError(
        f"{variable!r} names no resampling kernel and the mapping carries no '*' "
        f"to answer it; name it, or give '*' the kernel the rest warp with"
    )


def _group_resampling(
    resampling: Resampling | Mapping[str, Resampling], group: str
) -> Resampling | Mapping[str, Resampling]:
    """Re-scope a per-variable mapping onto the group whose variables read it.

    Args:
        resampling: One kernel for every variable, or a mapping keyed by
            group-qualified variable names.
        group: Group whose own keys are kept, unqualified.

    Returns:
        The kernel unchanged, or a mapping keyed by that group's plain variable
        names, carrying `"*"` where the caller gave one.
    """
    if not isinstance(resampling, Mapping):
        return resampling
    qualifier = f"{group}/"
    return {
        key.removeprefix(qualifier): kernel
        for key, kernel in resampling.items()
        if key.startswith(qualifier) or key == "*"
    }


def _finest(grids: Sequence[GeoBox]) -> GeoBox:
    """Pick the grid whose pixels cover least ground, so none is coarsened.

    Pixel sizes rank against each other only within one CRS, degrees and metres
    naming no common scale, so grids spread across several leave the first of
    them to stand.

    Args:
        grids: Grids to rank, at least one.

    Returns:
        The finest, or the first where they carry more than one CRS.
    """
    if len({grid.crs for grid in grids}) > 1:
        return grids[0]
    return min(grids, key=lambda grid: abs(grid.resolution.x))


def _source_geobox(data: xr.DataArray | xr.Dataset | xr.DataTree) -> GeoBox | None:
    """Read the grid a raster's pixels currently sit on.

    A stack need not publish a grid at its root, in which case the finest of
    its groups' grids is the one it warps from, so no group is coarsened for
    sitting later in the stack than another.

    Args:
        data: Raster of any kind.

    Returns:
        The grid it sits on, or None where nothing says where its pixels land.
    """
    geobox = data.gs.geobox
    if isinstance(geobox, GeoBox):
        return geobox

    if isinstance(data, xr.DataTree):
        grids = [
            grid
            for grid in (raster.gs.geobox for raster in data.gs.rasters.values())
            if isinstance(grid, GeoBox)
        ]
        if grids:
            return _finest(grids)
    return None


def _target_geobox(
    source: GeoBox, target: Target, resolution: SomeResolution | Native | None
) -> GeoBox:
    """Read the grid a target names, carrying `source` into a bare CRS.

    Args:
        source: Grid the pixels sit on now. A CRS target is carried into that
            CRS from here; a grid target replaces it outright.
        target: Grid to land on, a raster already on one, or a CRS.
        resolution: Output pixel size. None takes the target's own, or
            `source`'s ground sampling where the target names only a CRS.
            `"native"` keeps `source`'s pixel size either way, nesting it onto
            a target grid's lattice.

    Returns:
        The exact grid to warp onto.

    Raises:
        CRSError: `target` names no CRS pyproj recognises.
        ValueError: `target` is a raster on no locatable grid or names no CRS,
            `resolution` is a size the target already fixes, or `"native"`
            stands in no whole-numbered ratio to a target grid.
    """
    if isinstance(target, xr.DataArray | xr.Dataset | xr.DataTree):
        geobox = _source_geobox(target)
        if geobox is None:
            raise ValueError(
                "the target raster sits on no locatable grid to land on; pass a "
                "GeoBox, a CRS, or a raster placed by a regular grid"
            )
    elif isinstance(target, GeoBox):
        geobox = target
    else:
        crs = norm_crs(target)
        if crs is None:
            raise ValueError(
                f"{target!r} names no grid and no CRS to land in; pass a GeoBox, "
                f"a raster carrying one, or a CRS such as 'EPSG:3857'"
            )
        # A CRS fixes no pixel size, so keeping the source's is odc's "auto".
        scale = "auto" if resolution in (None, NATIVE) else resolution
        return source.to_crs(
            crs, resolution=cast("Literal['auto', 'fit', 'same']", scale)
        )

    if resolution == NATIVE:
        return geobox.zoom_out(_zoom_ratio(source, geobox))
    if resolution is not None:
        raise ValueError(
            f"the target grid already fixes its pixel size at {geobox.resolution}, "
            f"so resolution {resolution!r} contradicts it; drop resolution, ask for "
            f"{NATIVE!r} to keep the source's, or name a CRS instead of a grid"
        )
    return geobox


def reproject[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    target: Target,
    *,
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
    resolution: SomeResolution | Native | None = None,
) -> T:
    """Warp pixels onto the grid a target names.

    A target grid is adopted whole — CRS, resolution, and extent — while a bare
    CRS only decides the projection, sizing the grid from the source. Every
    group of a stack lands on it, and data already there is returned untouched.

    Args:
        data: DataArray, Dataset, or DataTree whose pixels sit on a locatable
            grid.
        target: Grid to land on, a raster already on one, or a CRS.
        resampling: One GDAL kernel for every variable, or a mapping naming
            each variable's own, which `"*"` answers the rest of. Inside a
            stack a key names its group, as `"<group>/<variable>"`.
        resolution: Output pixel size. None takes the target's own, or the
            source's ground sampling where the target names only a CRS. A grid
            derived from a CRS snaps to the pixel lattice and covers the source
            whole, so it reaches past the source's own bounds. `"native"` keeps
            each variable's own pixel size instead, nesting it onto the target's
            lattice, so a stack's groups come out at their own resolutions.

    Returns:
        New object of the same kind on the target grid, carrying its own
        `spatial_ref` and the CF semantics its axes earn.

    Raises:
        CRSError: `target` names no CRS pyproj recognises.
        KeyError: A `resampling` mapping names neither a variable nor `"*"`.
        TypeError: `data` is not an xarray object this warps.
        ValueError: `data` or `target` sits on no locatable grid, `data` holds
            no data variable, `resolution` contradicts a target grid or stands
            in no whole-numbered ratio to it, a variable of `data` does not
            span its own grid, or `resampling` would blend a variable whose
            values are class codes.

    Examples:
        The target grid comes from whatever already defines it — another
        raster, or the anchor the workspace loaded against:

        >>> dem = reproject(srtm, scene, resampling="bilinear")
        >>> dem.gs.geobox == scene.gs.geobox
        True

        A CRS decides only the projection, so the grid is sized from the
        source unless a resolution says otherwise:

        >>> reproject(scene, "EPSG:3857").gs.crs.epsg
        3857
        >>> reproject(scene, "EPSG:4326", resolution=0.001).gs.resolution.x
        0.001

        Resampling in place names a grid rather than a CRS, since a grid
        derived from a CRS is padded out to the lattice:

        >>> reproject(scene, scene.gs.geobox.zoom_out(2)).gs.geobox.shape
        Shape2d(x=4, y=4)

        Class codes must not be averaged, so a blending kernel refuses. A
        stack holding both imagery and labels names a kernel per variable:

        >>> reproject(landcover, scene, resampling="bilinear")
        Traceback (most recent call last):
        ValueError: ['landcover'] carry a class map, so 'bilinear' would blend ...
        >>> reproject(scene, target, resampling={"*": "bilinear", "labels/mask": "mode"})
    """
    if not isinstance(data, xr.DataArray | xr.Dataset | xr.DataTree):
        raise TypeError(
            f"warping takes a DataArray, Dataset, or DataTree, got "
            f"{type(data).__name__}"
        )

    blended: dict[str, Resampling] = {}
    for name in attrs.flag_variables(data):
        kernel = _variable_resampling(resampling, name)
        if kernel not in _VALUE_PRESERVING:
            blended[name] = kernel
    if blended:
        named = ", ".join(f"{name!r}: 'mode'" for name in sorted(blended))
        raise ValueError(
            f"{sorted(blended)} carry a class map, so "
            f"{sorted(set(blended.values()))} would blend their codes into values "
            f"naming no class; warp them with a value-preserving kernel, as "
            f"resampling={{'*': ..., {named}}}, or drop the "
            f"Legend first"
        )

    source = _source_geobox(data)
    if source is None:
        raise ValueError(
            f"{type(data).__name__} sits on no locatable grid, so nothing says "
            f"where its pixels start from; open it georeferenced first"
        )
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import stack

        rasters = data.gs.rasters
        lattice = _target_geobox(
            source, target, None if resolution == NATIVE else resolution
        )
        return cast(
            "T",
            stack(
                {
                    name: reproject(
                        raster,
                        lattice,
                        resampling=_group_resampling(resampling, name),
                        resolution=resolution,
                    )
                    for name, raster in rasters.items()
                }
            ),
        )

    geobox = _target_geobox(source, target, resolution)
    if geobox == source:
        return cast("T", data)

    if isinstance(data, xr.DataArray):
        band = cast("xr.DataArray", data)
        kernel = _variable_resampling(resampling, str(band.name))
        return cast("T", band.odc.reproject(geobox, resampling=kernel))

    # odc's Dataset path keeps the old spatial_ref attrs, so it reports the source.
    raster = cast("xr.Dataset", data)
    if not raster.data_vars:
        raise ValueError(
            "the Dataset holds no data variable, so it names no pixels to warp; "
            "select the variables you mean before warping"
        )
    grid_dims = set(raster.gs.grid_dims)
    data_vars: dict[str, xr.DataArray] = {}
    for variable, values in raster.data_vars.items():
        if not grid_dims <= set(values.dims):
            raise ValueError(
                f"{str(variable)!r} spans {list(values.dims)}, not the grid "
                f"{sorted(grid_dims)}, so it names no pixels to warp; drop it "
                f"or place it on the grid first"
            )
        data_vars[str(variable)] = values.odc.reproject(
            geobox, resampling=_variable_resampling(resampling, str(variable))
        )
    return cast("T", xr.Dataset(data_vars, attrs=dict(raster.attrs)).gs.write_crs())


def common_grid(
    rasters: Sequence[xr.DataArray | xr.Dataset | xr.DataTree],
    *,
    target: Target | None = None,
    extent: Extent = "union",
    resolution: SomeResolution | Native | None = None,
) -> GeoBox:
    """Read the one grid several rasters would align onto.

    The CRS and pixel edges are the reference's own, which is the finest of
    the rasters unless `target` names another. Only the ground it covers is
    `extent`'s to decide.

    Args:
        rasters: Rasters to measure, at least one, each on a locatable grid.
        target: Grid to align onto, a raster already on one, or a CRS the
            finest raster's grid is carried into. None takes the finest
            raster's grid as it stands, so none is coarsened for its place in
            the order.
        extent: Ground the grid covers. `"union"` reaches every raster;
            `"intersection"` keeps only the ground they all cover; `"match"`
            adopts the reference's own.
        resolution: Pixel size of the grid. None takes the reference's own.
            `"native"` returns the lattice the rasters nest on instead, at the
            finest of their pixel sizes and shaped so every one of them divides
            it — grown for `"union"` and `"match"`, trimmed for
            `"intersection"`, by under one coarse pixel either way.

    Returns:
        The grid they all warp onto, or the lattice they nest on.

    Raises:
        ValueError: `rasters` is empty, one sits on no locatable grid, `target`
            names no grid or CRS, the grid to align onto carries no CRS,
            `extent` names no ground, `"intersection"` leaves none in common, or
            `"native"` finds pixel sizes in no whole-numbered ratio.

    Examples:
        >>> common_grid([optical, radar, srtm], target=optical).shape
        Shape2d(x=512, y=512)
    """
    if not rasters:
        raise ValueError("no rasters to align; pass at least one")
    if extent not in get_args(Extent.__value__):
        raise ValueError(
            f"{extent!r} names no ground to cover; reach every raster with "
            f"'union', keep what they all cover with 'intersection', or adopt "
            f"the reference's own with 'match'"
        )

    grids: list[GeoBox] = []
    for ordinal, raster in enumerate(rasters):
        geobox = _source_geobox(raster)
        if geobox is None:
            raise ValueError(
                f"raster {ordinal} sits on no locatable grid, so nothing says "
                f"where its pixels land against the rest; open it georeferenced "
                f"first"
            )
        grids.append(geobox)

    # The finest raster is the reference, so none is coarsened for its place.
    basis = _finest(grids)
    reference = basis
    if target is not None:
        reference = _target_geobox(
            basis, target, None if resolution == NATIVE else resolution
        )
    elif resolution not in (None, NATIVE) and basis.crs is not None:
        reference = _target_geobox(basis, basis.crs, resolution)

    crs = reference.crs
    if crs is None:
        raise ValueError(
            "the grid to align onto carries no CRS, so nothing says where the "
            "other rasters land against it; write one with gs.write_crs first"
        )

    if extent == "match":
        common = reference
    elif extent == "union":
        # enclosing re-lays a footprint on the reference's own pixel grid.
        common = geobox_union_conservative(
            [reference.enclosing(grid.extent.to_crs(crs)) for grid in grids]
        )
    else:
        common = geobox_intersection_conservative(
            [reference.enclosing(grid.extent.to_crs(crs)) for grid in grids]
        )

    if resolution == NATIVE:
        # An undivided shape makes zoom_out round up, off the shared ground.
        step = lcm(*(_zoom_ratio(grid, common) for grid in grids))
        if extent == "intersection":
            kept = common.shape.y - common.shape.y % step
            across = common.shape.x - common.shape.x % step
            common = common[:kept, :across]
        else:
            common = common.pad_wh(step)

    if extent == "intersection" and not (common.shape.x and common.shape.y):
        raise ValueError(
            "the rasters cover no ground in common, so an intersection holds no "
            "pixel; align them on 'union' or load them over one footprint"
        )
    return common


def align[T: xr.DataArray | xr.Dataset | xr.DataTree](
    rasters: Sequence[T],
    *,
    target: Target | None = None,
    extent: Extent = "union",
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
    resolution: SomeResolution | Native | None = None,
) -> tuple[T, ...]:
    """Warp several rasters onto one grid, so one window reads them all.

    Args:
        rasters: Rasters to align, at least one, each on a locatable grid.
        target: Grid to align onto, a raster already on one, or a CRS the
            finest raster's grid is carried into. None takes the finest
            raster's grid as it stands, so none is coarsened for its place in
            the order.
        extent: Ground the aligned rasters cover. `"union"` reaches every
            raster, holding fill where one does not; `"intersection"` drops
            every edge outside the ground they all cover; `"match"` adopts the
            reference's own.
        resampling: One GDAL kernel for every variable, or a mapping naming
            each variable's own, which `"*"` answers the rest of. Pixels move
            only where a raster does not already sit on the grid they align
            onto.
        resolution: Pixel size every raster lands at. None takes the
            reference's own. `"native"` keeps each raster's instead, landing
            them on nesting grids rather than one.

    Returns:
        One raster per input, in the order given, every one on the same exact
        GeoBox — or, under `"native"`, on grids covering one extent at their
        own pixel sizes.

    Raises:
        KeyError: A `resampling` mapping names neither a variable nor `"*"`.
        ValueError: `rasters` is empty, one sits on no locatable grid, `target`
            names no grid or CRS, the grid to align onto carries no CRS,
            `extent` names no ground, `"intersection"` leaves none in common, or
            `"native"` finds pixel sizes in no whole-numbered ratio.

    Examples:
        A delivered DEM and a radar scene onto the optical grid, which one
        tiler then cuts together:

        >>> optical, radar, dem = align([optical, radar, srtm], target=optical)
        >>> {r.gs.geobox for r in (optical, radar, dem)} == {optical.gs.geobox}
        True

        Keeping only the ground every raster reaches leaves no fill behind:

        >>> [r.gs.geobox.shape for r in align(scenes, extent="intersection")]
        [Shape2d(x=180, y=180), Shape2d(x=180, y=180)]

        Keeping every raster's own pixel size lands them on nesting grids,
        which one ground extent still cuts together:

        >>> [r.gs.geobox.shape.x for r in align([b10, b20, b60], resolution="native")]
        [384, 192, 64]
    """
    geobox = common_grid(rasters, target=target, extent=extent, resolution=resolution)
    # The grid already carries any size asked for; only "native" travels on.
    return tuple(
        reproject(
            raster,
            geobox,
            resampling=resampling,
            resolution=NATIVE if resolution == NATIVE else None,
        )
        for raster in rasters
    )

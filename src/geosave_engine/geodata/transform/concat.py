"""Extending the time axis by laying rasters end to end."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.convention import TIME_COORDINATE
from geosave_engine.geodata.transform import nodata

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# How non-time coordinates align, forwarded to `xr.concat`'s own `join`.
type Join = Literal["outer", "inner", "left", "right", "exact", "override"]

# How non-time-varying data variables are checked, forwarded to `xr.concat`'s own `compat`.
type Compat = Literal[
    "identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"
]


def concat_time[T: xr.DataArray | xr.Dataset | xr.DataTree](
    rasters: Sequence[T],
    *,
    join: Join = "exact",
    compat: Compat = "equals",
    data_vars: str | Iterable[str] = "all",
    coords: str | Iterable[str] = "different",
    fill_value: float | int | Mapping[str, float | int] | None = None,
) -> T:
    """Extend the time axis by laying rasters end to end.

    Only `time` is extended. `join="exact"` requires every other coordinate
    to already agree, refusing otherwise; a looser `join` instead aligns them,
    padding what one raster lacks with `fill_value`.

    Args:
        rasters: Datasets, DataArrays, or DataTrees sharing a grid and dtype,
            each carrying a `time` coordinate, at least one. A DataTree
            recurses group by group.
        join: How non-time coordinates are aligned, forwarded to `xr.concat`:
            `"exact"`, `"outer"`, `"inner"`, `"left"`, or `"right"`.
        compat: How non-time-varying data variables are checked for
            agreement, forwarded to `xr.concat`.
        data_vars: Which data variables concatenate along `time` rather than
            being taken from the first raster, forwarded to `xr.concat`.
        coords: Which coordinates concatenate along `time` rather than being
            taken from the first raster, forwarded to `xr.concat`.
        fill_value: Value written where `join` pads a raster that does not
            reach a position, scalar or per-variable. None reads what each
            variable already carries; only consulted when `join` is not
            `"exact"`.

    Returns:
        New object of the kind given, spanning every raster's time labels in
        the order the rasters were given. Nothing is reordered, so sort the
        result with `.sortby("time")` where a chronological axis is wanted —
        resampling, interpolation and label-based selection each require one.

    Raises:
        ValueError: `rasters` is empty, the stacks hold different groups, they
            carry different CRSs, `join` pads a variable that carries no fill
            value and none is given, or the rasters differ in a way
            `join`/`compat` refuses.

    Examples:
        >>> series = concat_time([january, february, march])

        Rasters that arrived out of order lay down in that order all the same.
        Sort the result where the axis has to be chronological; a DataTree
        carries no `sortby`, so a stack sorts group by group::

            >>> concat_time([march, january]).sortby("time")
            >>> stack({n: g.sortby("time") for n, g in tree.gs.rasters.items()})
    """
    if not rasters:
        raise ValueError("no rasters to concatenate; pass at least one")
    if isinstance(rasters[0], xr.DataTree):
        from geosave_engine.geodata.core.stack import stack

        trees = cast("Sequence[xr.DataTree]", rasters)
        groups = trees[0].gs.rasters.keys()
        for ordinal, tree in enumerate(trees[1:], start=1):
            if set(tree.gs.groups) != set(groups):
                raise ValueError(
                    f"stack {ordinal} holds {sorted(tree.gs.groups)} and stack 0 "
                    f"holds {sorted(groups)}, so they name different rasters to lay "
                    f"end to end; concatenate stacks carrying the same groups"
                )
        return cast(
            "T",
            stack(
                {
                    name: concat_time(
                        [tree.gs.rasters[name] for tree in trees],
                        join=join,
                        compat=compat,
                        data_vars=data_vars,
                        coords=coords,
                        fill_value=fill_value,
                    )
                    for name in groups
                }
            ),
        )

    bands = cast("Sequence[xr.DataArray | xr.Dataset]", rasters)
    # Grid axes are `join`'s to settle; a CRS it never reads, so check it here.
    crs_names = {band.gs.crs_name for band in bands}
    if len(crs_names) > 1:
        raise ValueError(
            f"the rasters carry {sorted(str(name) for name in crs_names)}, so "
            f"laying them end to end would stack pixels naming different ground; "
            f"reproject them onto one CRS first"
        )

    reference = bands[0]
    resolved = fill_value
    if resolved is None and join != "exact":
        resolved = (
            nodata.required_fill_value(reference)
            if isinstance(reference, xr.DataArray)
            else nodata.required_fill_values(reference)
        )

    header = attrs.merge(bands)
    result: xr.DataArray | xr.Dataset
    if isinstance(bands[0], xr.DataArray):
        result = xr.concat(
            cast("Sequence[xr.DataArray]", bands),
            dim=TIME_COORDINATE,
            join=join,
            compat=compat,
            data_vars=data_vars,
            coords=coords,
            fill_value=resolved,
        )
    else:
        result = xr.concat(
            cast("Sequence[xr.Dataset]", bands),
            dim=TIME_COORDINATE,
            join=join,
            compat=compat,
            data_vars=data_vars,
            coords=coords,
            fill_value=resolved,
        )
    return cast("T", attrs.rebase(result, header))

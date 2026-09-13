"""Extending the time axis by laying rasters end to end."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import pandas as pd
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


def _require_chronological(rasters: Sequence[xr.DataArray | xr.Dataset]) -> None:
    """Refuse rasters whose time labels are not strictly chronological.

    Args:
        rasters: Datasets or DataArrays, each carrying a `time` coordinate,
            in the order they are meant to concatenate.

    Raises:
        ValueError: A raster's own time labels are not chronological, or one
            raster ends at or after the next one starts.
    """
    labels = [raster.coords[TIME_COORDINATE].values for raster in rasters]
    for index, own in enumerate(labels):
        if not pd.Index(own).is_monotonic_increasing:
            raise ValueError(
                f"raster {index}'s own time labels are not chronological; "
                f"sort it before concatenating"
            )
    for index in range(len(labels) - 1):
        if labels[index][-1] >= labels[index + 1][0]:
            raise ValueError(
                f"raster {index} ends at {labels[index][-1]}, at or after "
                f"raster {index + 1} starts at {labels[index + 1][0]}; concat "
                f"needs chronological, non-overlapping rasters"
            )


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
            each carrying a `time` coordinate, in chronological order with no
            overlap. A DataTree recurses group by group.
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
        New object of the kind given, spanning every raster's time labels, in
        the order given.

    Raises:
        ValueError: `rasters` is empty, a raster's own time labels are not
            chronological, two rasters overlap or are out of order, `join`
            pads a variable that carries no fill value and none is given, or
            they differ in a way `join`/`compat` refuses.

    Examples:
        >>> series = concat_time([january, february, march])
    """
    if isinstance(rasters[0], xr.DataTree):
        from geosave_engine.geodata.core.stack import stack

        trees = cast("Sequence[xr.DataTree]", rasters)
        groups = trees[0].gs.rasters.keys()
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
    _require_chronological(bands)

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

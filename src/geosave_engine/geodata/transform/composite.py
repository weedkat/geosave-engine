"""Collapsing several observations of the same ground into one, per pixel.

`mosaic` reduces an explicit list of rasters; `resample` cuts one raster's time
axis into calendar buckets and `reduce` collapses them.

Examples:
    An AOI spanning two granules, cloud covered by the week before, and a
    series too noisy to train on::

        scene = mosaic([granule_a, granule_b])
        monthly = reduce(resample(scene, "MS"), "median")
        gapless = interpolate(monthly, "nearest")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox, geobox_union_conservative
from xarray.core.resample import DataArrayResample, DatasetResample

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.convention import TIME_COORDINATE
from geosave_engine.geodata.transform import nodata, warp
from geosave_engine.geodata.utils.datetime import freq_offset

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime as dt, timedelta

    from geosave_engine.geodata import DataArray, Dataset
    from geosave_engine.geodata.utils.datetime import Freq

# How an absent cell reads its value off the observed buckets around it.
type Interpolation = Literal["nearest", "linear"]

# How long a run of absent cells may be, as pandas reads a duration.
type Gap = str | pd.Timedelta | timedelta | np.timedelta64

# Named reducers a bucket may collapse with.
type Reducer = Literal[
    "mean", "median", "sum", "min", "max", "std", "var", "first", "last"
]

# Which raster an overlapping pixel is taken from — a whole pixel, never a blend.
type MosaicMethod = Literal["first", "last"]


def mosaic(rasters: Sequence[xr.Dataset], *, method: MosaicMethod = "first") -> Dataset:
    """Lay several rasters onto the grid they jointly cover.

    Only `y`/`x` are laid out; every other coordinate must already agree
    across `rasters`. A pixel is always taken whole from one raster, never
    blended across two.

    Args:
        rasters: Datasets in one CRS and resolution, at least one, in
            preference order.
        method: Which raster an overlapping pixel is taken from. `"first"`
            takes the earliest raster carrying a value there, leaving
            absence only where none does; `"last"` takes the latest.

    Returns:
        New Dataset covering every raster's ground, absent only where no
        raster carried a value.

    Raises:
        ValueError: `rasters` is empty, one sits on no locatable grid, or no
            one grid holds them all because they disagree on CRS, resolution
            or pixel phase.

    Examples:
        >>> scene = mosaic([granule_32TNS, granule_32TPS])
    """
    if not rasters:
        raise ValueError("no rasters to mosaic; pass at least one")

    header = attrs.merge(rasters)
    fills = nodata.required_fill_values(rasters[0], header.data_vars)

    grids: list[GeoBox] = []
    for ordinal, raster in enumerate(rasters):
        geobox = raster.gs.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(
                f"raster {ordinal} sits on no locatable grid, so no ground says "
                f"where it overlaps the rest; open it georeferenced first"
            )
        grids.append(geobox)

    reference = grids[0]
    for ordinal, grid in enumerate(grids[1:], start=1):
        if grid.crs != reference.crs:
            mismatch = (
                f"CRS {rasters[ordinal].gs.crs_name} against {rasters[0].gs.crs_name}"
            )
        elif grid.resolution != reference.resolution:
            mismatch = f"resolution {grid.resolution} against {reference.resolution}"
        else:
            continue
        raise ValueError(
            f"raster {ordinal} carries {mismatch} of raster 0, so no one grid holds "
            f"both; align them onto one grid first"
        )

    # Sharing a CRS and resolution leaves only phase to refuse a union over.
    try:
        covered = geobox_union_conservative(grids)
    except ValueError as offset:
        raise ValueError(
            "the rasters lay their pixel edges on different lines, so no one grid "
            "holds them at that phase; align them onto one grid first"
        ) from offset

    covering = tuple(warp.reproject(raster, covered) for raster in rasters)
    sources = covering if method == "first" else covering[::-1]
    composite = sources[0]
    for source in sources[1:]:
        gaps = xr.Dataset({name: nodata.is_fill(composite[name]) for name in fills})
        composite = xr.where(gaps, source, composite)

    # Coordinate attrs are each raster's own tile-specific grid; the union's own are already right.
    composite_header = attrs.AttrsHeader(
        root=header.root, variables=header.data_vars, var_names=header.var_names
    )
    return cast("Dataset", attrs.rebase(composite, composite_header).gs.write_crs())


@overload
def reduce(resampler: DataArrayResample, method: Reducer) -> DataArray: ...


@overload
def reduce(resampler: DatasetResample, method: Reducer) -> Dataset: ...


def reduce(
    resampler: DataArrayResample | DatasetResample, method: Reducer
) -> DataArray | Dataset:
    """Collapse each bucket into one value per cell.

    The resampler carries its own grouping, so nothing else says how to group.

    Args:
        resampler: Observations cut into buckets, as `resample` cuts them.
        method: Named reducer each bucket collapses with.

    Returns:
        New object of the kind the resampler holds, one position per bucket,
        carrying the CF cell method it was collapsed with.

    Raises:
        ValueError: A collapsed variable holds class codes and `method` is not
            a point sample of them.

    Examples:
        >>> monthly = reduce(resample(daily, "MS"), "median")
        >>> monthly.sizes["time"], monthly.red.attrs["cell_methods"]
        (12, 'time: median')
    """
    cell_method = attrs.CELL_METHODS[method]
    values = getattr(resampler, method)()

    # A class code neither averages nor ranks, so only a point sample keeps it.
    flagged = attrs.flag_variables(values)
    if flagged and cell_method != "point":
        raise ValueError(
            f"{list(flagged)} hold class codes, which {method!r} either blends "
            f"into codes naming no class or orders as if they ranked; reduce "
            f"them with 'first' or 'last', select the variables you mean, or "
            f"drop the Legend first"
        )

    collapsed = attrs.CFVariable(cell_methods=f"{TIME_COORDINATE}: {cell_method}")
    if isinstance(values, xr.DataArray):
        return cast("DataArray", attrs.rebase(values, collapsed))
    return cast("Dataset", attrs.rebase(values, collapsed, target=values.gs.variables))


@overload
def resample(
    data: xr.DataArray,
    freq: Freq,
    *,
    closed: Literal["left", "right"] | None = ...,
    label: Literal["left", "right"] | None = ...,
    origin: str | dt = ...,
    offset: str | timedelta | None = ...,
    skip_nodata: bool = ...,
) -> DataArrayResample: ...


@overload
def resample(
    data: xr.Dataset,
    freq: Freq,
    *,
    closed: Literal["left", "right"] | None = ...,
    label: Literal["left", "right"] | None = ...,
    origin: str | dt = ...,
    offset: str | timedelta | None = ...,
    skip_nodata: bool = ...,
) -> DatasetResample: ...


def resample(
    data: xr.DataArray | xr.Dataset,
    freq: Freq,
    *,
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    origin: str | dt = "start_day",
    offset: str | timedelta | None = None,
    skip_nodata: bool = True,
) -> DataArrayResample | DatasetResample:
    """Cut a time axis into calendar buckets, recording how they were cut.

    The `time` coordinate carries a `TimeSpec` from here on, saying what its
    labels stand for. No pixel moves, and no value changes but the nodata
    ones `skip_nodata` leaves NaN.

    Args:
        data: DataArray or Dataset carrying a `time` coordinate.
        freq: Target cadence — any pandas offset alias. The alias fixes where
            bucket edges fall: `"MS"` on month starts, `"5D"` every five days.
        closed: Which bucket edge is inclusive, `"left"` or `"right"`. None
            resolves as pandas does, by alias: a period-end one such as `"ME"`
            or `"W"` closes on the right, every other alias on the left.
        label: Which edge names the bucket, resolved the same way. The
            recorded `TimeSpec` carries whichever edge both resolved to.
        origin: Timestamp the edge grid is phased from, or a strategy name.
        offset: Shift added on top of `origin`.
        skip_nodata: Leave NaN where a variable carries its nodata value, so
            a reducer skips those pixels. False keeps the stored value, which
            every reducer then takes for a reading and blends into the result.

    Returns:
        xarray's resampler over the buckets, holding data whose `time`
        coordinate records the bucketing. Reduce it with any reducer xarray
        offers; a bucket covering no observation survives as absent.

    Raises:
        ValueError: `data` carries no `time` coordinate, or pandas does not
            know `freq`.

    Examples:
        >>> monthly = resample(daily, "MS").median()
        >>> monthly.time.values
        array(['2024-01-01', '2024-02-01', '2024-03-01'], dtype='datetime64[ns]')
        >>> monthly.gs.attrs.coords["time"].get(TimeSpec).time_freq
        'MS'
    """
    if TIME_COORDINATE not in data.coords:
        raise ValueError(
            f"{type(data).__name__} carries no {TIME_COORDINATE!r} coordinate, "
            f"so it holds no observations to bucket; label its time axis first"
        )

    alias = freq_offset(freq)
    bucketing = attrs.TimeSpec.from_resample(
        alias, closed=closed, label=label, origin=origin, offset=offset
    )
    values = nodata.to_nan(data) if skip_nodata else data
    return attrs.rebase(values, bucketing, target=TIME_COORDINATE).resample(
        {TIME_COORDINATE: alias},
        closed=closed,
        label=label,
        origin=origin,
        offset=offset,
    )


@overload
def interpolate(
    data: xr.DataArray, method: Interpolation, *, max_gap: Gap | None = ...
) -> DataArray: ...


@overload
def interpolate(
    data: xr.Dataset, method: Interpolation, *, max_gap: Gap | None = ...
) -> Dataset: ...


def interpolate(
    data: xr.DataArray | xr.Dataset,
    method: Interpolation,
    *,
    max_gap: Gap | None = None,
) -> DataArray | Dataset:
    """Fill the cells a bucketed axis left absent, along time.

    `resample` emits every bucket its cadence covers, so one observing nothing
    is already on the axis, holding absence. What that becomes is settled cell
    by cell, so a partly clouded bucket fills only the cells it missed.

    Args:
        data: DataArray or Dataset whose `time` coordinate records the
            bucketing it was cut with, as `reduce` leaves it.
        method: How an absent cell reads its value. `"nearest"` repeats the
            nearest observed bucket, reaching a bucket with none on one side
            of it; `"linear"` interpolates between the observed buckets on
            either side, leaving a cell with only one side absent.
        max_gap: Longest run of absent cells to fill, measured between the
            observations either side of it, e.g. `"45D"`. None fills a run of
            any length. Bounding one needs `bottleneck` installed, which
            xarray reads run lengths through.

    Returns:
        New object of the kind given, on the same axis, its absent cells filled
        as `method` says. A cell observed once stays absent, having nothing to
        read between.

    Raises:
        ValueError: The `time` axis is absent or records no bucketing, or
            `method="linear"` would interpolate class codes.

    Examples:
        A cloudy January observed nothing, so its bucket is absent throughout.
        A sequence model wants a reading there:

        >>> gapless = interpolate(monthly, "nearest")
        >>> gapless.red.isnull().any().item()
        False
    """
    if TIME_COORDINATE not in data.coords:
        raise ValueError(
            f"{type(data).__name__} carries no {TIME_COORDINATE!r} coordinate, so "
            f"it holds no buckets to fill; resample its time axis first"
        )

    header = attrs.read(data)
    timespec = header.coords[TIME_COORDINATE].get(attrs.TimeSpec)
    if timespec is None or timespec.time_freq is None:
        raise ValueError(
            f"the {TIME_COORDINATE!r} axis names instants rather than buckets, so "
            f"no bucket observed nothing; resample it to a cadence first"
        )

    flagged = attrs.flag_variables(data)
    if method == "linear" and flagged:
        raise ValueError(
            f"{list(flagged)} hold class codes, which 'linear' blends into codes "
            f"naming no class; fill them with 'nearest', select the variables "
            f"you mean, or drop the Legend first"
        )

    # Only the nearest reading reaches a bucket with none on one side of it.
    fill_value = "extrapolate" if method == "nearest" else np.nan
    filled = data.interpolate_na(
        dim=TIME_COORDINATE,
        method=method,
        max_gap=max_gap,
        fill_value=fill_value,
    )
    return cast("DataArray | Dataset", attrs.rebase(filled, header))

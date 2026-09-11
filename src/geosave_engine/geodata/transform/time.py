"""Transformations along the time axis."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

from geosave_engine.geodata.attrs import (
    Packing,
    TimeSpec,
    combine,
    rebase,
    stamp,
)
from geosave_engine.geodata.core.raster import GeoRaster
from geosave_engine.geodata.errors import DroppedBucketsWarning, UnmatchedBucketsWarning
from geosave_engine.geodata.utils.datetime import (
    bucket_labels,
    edge_rules,
    freq_offset,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime as dt, timedelta

    from geosave_engine.geodata.utils.datetime import DateRange, Freq

type Reducer = Literal[
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "first",
    "last",
    "std",
    "var",
    "count",
]

type EmptyBuckets = Literal["keep", "nearest", "linear"]

# Reducers that pick an observed value rather than blending several.
_VALUE_PRESERVING: frozenset[str] = frozenset({"first", "last", "max", "min"})

_TIME_BOUNDS_COORDINATE = "time_bnds"
_TIME_BOUNDS_DIMENSION = "bnds"


def _bucket_grid(labels: np.ndarray, spec: TimeSpec) -> pd.DatetimeIndex:
    """Rebuild the full grid a spec's cadence covers over some labels.

    Args:
        labels: Observed bucket labels, ascending.
        spec: Cadence, edges, and origin the labels were bucketed with.

    Returns:
        Every bucket label the cadence puts between the first and the last.

    Raises:
        ValueError: pandas does not know the spec's own `freq`.
    """
    covered = cast(
        "DateRange",
        (
            pd.Timestamp(labels[0]).to_pydatetime(),
            pd.Timestamp(labels[-1]).to_pydatetime(),
        ),
    )
    return bucket_labels(
        covered,
        spec.freq,
        closed=spec.time_closed,
        label=spec.time_label,
        origin=spec.time_origin,
        offset=spec.time_offset,
    )


def _require_ordered_labels(labels: pd.DatetimeIndex) -> None:
    """Refuse target labels that do not name one bucket each, in order.

    Args:
        labels: Bucket labels to stretch onto.

    Raises:
        ValueError: `labels` descends or repeats.
    """
    if not labels.is_monotonic_increasing:
        raise ValueError("labels must be ascending")
    if labels.has_duplicates:
        raise ValueError("labels must not repeat")


def _with_bounds(raster: xr.Dataset, spec: TimeSpec) -> xr.Dataset:
    """Attach the cell edges each of a spec's buckets spans.

    Args:
        raster: Dataset on a bucketed time axis.
        spec: Bucketing the labels stand for.

    Returns:
        New Dataset carrying a `time_bnds` coordinate over the time axis.
    """
    return raster.assign_coords(
        {
            _TIME_BOUNDS_COORDINATE: (
                ("time", _TIME_BOUNDS_DIMENSION),
                spec.bounds(raster["time"].values),
            )
        }
    )


def resample(
    raster: xr.Dataset,
    freq: Freq,
    method: Reducer = "median",
    *,
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
    origin: str | dt = "start_day",
    offset: str | timedelta | None = None,
) -> xr.Dataset:
    """Collapse the time axis onto coarser buckets.

    An empty bucket is dropped, with a `DroppedBucketsWarning` — call
    `interpolate` after to fill it back in. Adds a `time_bnds` coordinate and
    rebases a `TimeSpec`.

    Args:
        raster: Dataset carrying a `time` coordinate.
        freq: Target cadence — any pandas offset alias. The alias fixes where
            the bucket edges fall: `"MS"` on month starts, `"ME"` on month
            ends, `"5D"` every five days from `origin`.
        method: Named reducer each bucket collapses with.
        closed: Which of a bucket's two edges is inclusive, passed to pandas.
            None takes the alias default — right for `"ME"`, `"W"`; left else.
        label: Which edge names the bucket in the `time` coordinate, passed to
            pandas. None takes the alias default. It never moves the edges;
            `time_bnds` always spans the whole bucket.
        origin: Timestamp the edge grid is phased from, or a strategy name
            such as `"start_day"`.
        offset: Shift added on top of `origin`.

    Returns:
        New Dataset on the bucketed time axis, carrying a `time_bnds`
        coordinate and a `TimeSpec` recording the resample. A blending reducer
        drops the scaling it invalidates and declares NaN as the fill value it
        leaves, or no fill value where it stays integral.

    Raises:
        ValueError: `raster` carries no `time` coordinate, `raster` already
            records a resample, pandas does not know `freq`, `freq` buckets
            more finely than the axis already observes, or `method` would
            blend a categorical variable.

    Examples:
        >>> monthly = transform.time.resample(daily, "MS")
        >>> monthly.gs.timespan
        (datetime(2024, 1, 1, 0, 0), datetime(2024, 3, 31, 23, 59, 59, 999999))
    """
    if "time" not in raster.coords:
        raise ValueError(
            "raster carries no 'time' coordinate, so there is no axis to bucket"
        )
    if raster.gs.attrs.root.get(TimeSpec) is not None:
        raise ValueError(
            "raster already records a resample; bucketing it again composes the "
            "reducers unpredictably, so resample the raw axis to the target cadence"
        )

    alias = freq_offset(freq)
    closed, label = edge_rules(alias, closed, label)
    labels = raster.time.values

    buckets = (
        pd.Series(0, index=pd.DatetimeIndex(labels))
        .resample(alias, closed=closed, label=label, origin=origin, offset=offset)
        .count()
    )
    if buckets.size > labels.size:
        raise ValueError(
            f"bucketing at {alias!r} makes {buckets.size} buckets for "
            f"{labels.size} observations, so it upsamples; pick a cadence at "
            f"least as coarse as the axis already observes"
        )

    categorical = sorted(GeoRaster(raster).categorical)
    if categorical and method not in _VALUE_PRESERVING:
        raise ValueError(
            f"{categorical} carry a class map, so {method!r} would average "
            f"class values; pick one of {sorted(_VALUE_PRESERVING)} or drop "
            f"them before bucketing"
        )

    grouped = raster.resample(
        time=alias, closed=closed, label=label, origin=origin, offset=offset
    )
    reducer = getattr(grouped, method, None)
    if not callable(reducer):
        raise ValueError(
            f"{method!r} is not a resample reducer; pass a name xarray defines, "
            f"such as 'median', 'mean', or 'first'"
        )
    computed = cast("xr.Dataset", reducer())

    # An unobserved bucket counts as NaN here, not 0 as pandas' own resample gives.
    occupancy = raster.time.resample(
        time=alias, closed=closed, label=label, origin=origin, offset=offset
    ).count()
    occupied = occupancy > 0
    if not bool(occupied.all()):
        warnings.warn(
            f"{int((~occupied).sum())} of {occupancy.size} buckets at {alias!r} "
            f"covered no observation and were dropped; call interpolate() to "
            f"fill them back in",
            DroppedBucketsWarning,
            stacklevel=2,
        )
    # Every observation lands in some bucket, so at least one is occupied.
    computed = computed.sel(time=occupancy.time[occupied])

    # An empty bucket forces a NaN slice, which upcasts a reducer that only picks.
    if method in _VALUE_PRESERVING:
        computed = computed.assign(
            {
                name: computed[name].astype(array.dtype)
                for name, array in raster.data_vars.items()
                if np.issubdtype(array.dtype, np.integer)
            }
        )

    # A blending reducer leaves physical values, which the stored packing no longer describes.
    if method not in _VALUE_PRESERVING:
        # Blending to a float marks absence with NaN; an integral sum keeps none.
        blended = tuple(
            str(name)
            for name, array in computed.data_vars.items()
            if np.issubdtype(array.dtype, np.floating)
        )
        integral = tuple(
            str(name) for name in computed.data_vars if str(name) not in blended
        )
        if blended:
            computed = computed.gs.rebase(
                Packing(scale_factor=None, add_offset=None, _FillValue=float("nan")),
                target=blended,
            )
        if integral:
            computed = computed.gs.rebase(
                Packing(scale_factor=None, add_offset=None, _FillValue=None),
                target=integral,
            )

    spec = TimeSpec.from_resample(
        alias,
        method=method,
        closed=closed,
        label=label,
        origin=origin,
        offset=offset,
    )
    return _with_bounds(computed, spec).gs.rebase(spec, target=(None, "time"))


def interpolate(raster: xr.Dataset, empty: EmptyBuckets) -> xr.Dataset:
    """Fill a bucketed axis's missing calendar positions.

    Rebuilds the full bucket grid `raster`'s own cadence covers and settles
    the positions `resample` dropped, leaving an observed bucket's own nodata
    pixels alone. Calendar-idempotent — a gapless raster comes back unchanged.

    Args:
        raster: Dataset carrying a `TimeSpec` (a `resample` result), whose
            axis may skip calendar positions.
        empty: What a missing bucket becomes. `"keep"` emits it as nodata,
            `"nearest"` repeats the nearest observed bucket's values, and
            `"linear"` interpolates between neighbours.

    Returns:
        New Dataset on the full bucket grid `raster`'s own cadence covers,
        carrying a recomputed `time_bnds` coordinate.

    Raises:
        ValueError: `raster` carries no `TimeSpec`, or `empty='linear'` would
            interpolate a categorical variable.

    Examples:
        >>> gapless = transform.time.interpolate(monthly, "nearest")
        >>> gapless.gs.timespan == monthly.gs.timespan
        True
    """
    spec = _require_timespec(raster)
    categorical = sorted(GeoRaster(raster).categorical)
    if categorical and empty == "linear":
        raise ValueError(
            f"{categorical} carry a class map, so empty='linear' would "
            f"interpolate class values into ones that mean nothing; pass "
            f"empty='nearest' or 'keep'"
        )

    labels = raster.time.values
    full = _bucket_grid(labels, spec)

    filled = raster.drop_vars(_TIME_BOUNDS_COORDINATE, errors="ignore")
    if empty == "keep":
        filled = filled.reindex(time=full)
    elif empty == "nearest":
        filled = filled.reindex(time=full, method="nearest")
    else:
        filled = filled.interp(time=full, method="linear")

    return _with_bounds(filled, spec).gs.rebase(spec, target=(None, "time"))


def _require_timespec(raster: xr.Dataset) -> TimeSpec:
    """Read the TimeSpec a bucketed raster must carry.

    Args:
        raster: Dataset expected to carry a `TimeSpec`.

    Returns:
        The raster's own `TimeSpec`.

    Raises:
        ValueError: `raster` carries no `TimeSpec`.
    """
    spec = GeoRaster(raster).attrs.root.get(TimeSpec)
    if not isinstance(spec, TimeSpec) or spec.freq is None:
        raise ValueError(
            "raster carries no TimeSpec cadence, so its time axis names instants "
            "rather than a bucketed grid; resample it onto a cadence first"
        )
    return spec


def _require_contiguous(labels: np.ndarray, spec: TimeSpec, subject: str) -> None:
    """Refuse a bucket sequence that skips a calendar position.

    Rebuilds the grid `spec` would produce over `labels`' own first-to-last
    span (not a time_bnds edge — that sits exactly on the seam between two
    buckets, folding a right-closed cadence into a phantom neighbour).

    Args:
        labels: Observed bucket labels to check, ascending.
        spec: Cadence, edges, and origin the labels were bucketed with.
        subject: What `labels` belongs to, named in the error.

    Raises:
        ValueError: `labels` does not match the rebuilt grid exactly.
    """
    expected = _bucket_grid(labels, spec)
    if not np.array_equal(labels, expected.values):
        missing = expected[~expected.isin(labels)]
        raise ValueError(
            f"{subject} is missing bucket(s) at {list(missing.values)} for its "
            f"own recorded cadence ({spec.freq!r}); isel() around the gap, or "
            f"call interpolate(..., 'nearest') or 'linear' before windowing"
        )


def broadcast(raster: xr.Dataset, labels: pd.DatetimeIndex) -> xr.Dataset:
    """Stretch a coarser or timeless raster's own buckets onto a finer axis.

    Not `xarray`'s shape-mechanical broadcasting — this is calendar-aware and
    invents nothing: a label finds the one bucket whose `time_bnds` covers it.
    An unneeded source bucket is left out, with `UnmatchedBucketsWarning`.

    Args:
        raster: Dataset to stretch — either timeless, or carrying a `time`
            coordinate and a `time_bnds` coordinate giving each label's
            bucket, as `resample` writes. A recorded `TimeSpec` refines which
            edge each bucket owns; a right label matches on `(start, end]`, a
            left label (the default without a spec) on `[start, end)`.
        labels: Target bucket labels to stretch onto, ascending and unique.

    Returns:
        New Dataset on `labels`, each label carrying its covering bucket's
        values, stamped, with neither `time_bnds` nor a `TimeSpec`.

    Raises:
        ValueError: `labels` is not ascending or repeats; `raster` carries a
            `time` coordinate but no `time_bnds`; a label in `labels` falls in
            no bucket of `raster`'s own axis (a dropped bucket, an edge the
            bucket does not own, or outside its covered span); or a bucket of
            `raster` sits between two matched buckets, so `raster` buckets
            more finely than `labels` and broadcasting would silently drop
            most of it.

    Examples:
        >>> labels = pd.DatetimeIndex(monthly.time.values)
        >>> transform.time.broadcast(yearly_dem, labels).sizes["time"]
        12
    """
    _require_ordered_labels(labels)

    if "time" not in raster.coords:
        return raster.expand_dims(time=labels)
    if _TIME_BOUNDS_COORDINATE not in raster.coords:
        raise ValueError(
            "raster carries a 'time' coordinate but no 'time_bnds', so its "
            "buckets have no span to stretch; resample it onto a cadence first"
        )

    bounds = raster[_TIME_BOUNDS_COORDINATE].values.astype("datetime64[us]")
    starts, ends = bounds[:, 0], bounds[:, 1]
    target = labels.values.astype("datetime64[us]")

    spec = GeoRaster(raster).attrs.root.get(TimeSpec)
    if isinstance(spec, TimeSpec) and spec.time_label == "right":
        # A right label owns its bucket's closing edge, so match on (start, end].
        index = np.searchsorted(ends, target, side="left")
        covered = (index < starts.size) & (
            target > starts[np.clip(index, 0, starts.size - 1)]
        )
    else:
        # A left label owns its bucket's opening edge, so match on [start, end).
        index = np.searchsorted(starts, target, side="right") - 1
        covered = (index >= 0) & (target < ends[np.clip(index, 0, ends.size - 1)])
    if not bool(covered.all()):
        raise ValueError(
            f"labels {list(labels[~covered])} fall in no bucket of the raster's "
            f"own time axis (a bucket resample dropped, or outside its covered "
            f"span); interpolate(raster, 'nearest') or trim them first"
        )

    unmatched = sorted(set(range(starts.size)) - set(index.tolist()))
    if unmatched:
        # A bucket caught between two matched ones means the source is finer than labels.
        lo, hi = int(index.min()), int(index.max())
        interior = [i for i in unmatched if lo < i < hi]
        if interior:
            raise ValueError(
                f"buckets at {list(raster.time.values[interior])} sit between "
                f"matched buckets but match no label; raster buckets more "
                f"finely than labels, so broadcasting would silently drop "
                f"most of it — resample it to a coarser cadence first"
            )
        warnings.warn(
            f"{len(unmatched)} of {starts.size} buckets, at "
            f"{list(raster.time.values[unmatched])}, sit outside labels' own "
            f"span and were left out of the result",
            UnmatchedBucketsWarning,
            stacklevel=2,
        )

    stretched = raster.isel(time=index).drop_vars(
        _TIME_BOUNDS_COORDINATE, errors="ignore"
    )
    stretched = stretched.assign_coords(time=labels)
    # Broadcasting repeats each bucket's value, so the recorded bucketing no longer holds.
    return rebase(stretched, timespec=None)


def broadcast_stack(
    rasters: Mapping[str, xr.Dataset], labels: pd.DatetimeIndex
) -> xr.DataTree:
    """Broadcast each named raster onto one shared axis, then stack them.

    The pre-`stack` step for rasters on mismatched cadences: a timeless raster
    and one already on `labels` pass through, a coarser one is `broadcast`,
    and a finer one refuses. The assembled stack shares one exact `time` axis.

    Args:
        rasters: Group name mapped to a raster Dataset, as `stack` takes.
        labels: Shared target axis, ascending and unique — typically the
            finest raster's own `time` values.

    Returns:
        Flat DataTree whose groups all sit on `labels`.

    Raises:
        ValueError: `rasters` is empty; `labels` is not ascending or repeats;
            a raster carries `time` but no `time_bnds`; a label falls in no
            bucket of some raster's axis; or a raster buckets more finely
            than `labels`.

    Examples:
        >>> shared = pd.DatetimeIndex(optical.time.values)
        >>> broadcast_stack({"optical": optical, "dem": dem}, shared).gs.groups
        ('optical', 'dem')
    """
    from geosave_engine.geodata.core.stack import stack

    _require_ordered_labels(labels)
    if not rasters:
        raise ValueError("a raster stack needs at least one named raster")

    aligned: dict[str, xr.Dataset] = {}
    for name, raster in rasters.items():
        if "time" not in raster.coords or raster.get_index("time").equals(labels):
            aligned[name] = raster
            continue
        try:
            aligned[name] = broadcast(raster, labels)
        except ValueError as error:
            raise ValueError(f"raster {name!r}: {error}") from error

    return stack(aligned)


def time_window(
    raster: xr.Dataset,
    slot: int,
    *,
    stride: int | None = None,
) -> list[xr.Dataset]:
    """Cut a bucketed raster into fixed-length windows along time.

    A window names no reference bucket — read one off its own `time` values
    afterward (`[0]`/`[-1]`/middle, matching `TimeSpec.time_label`'s vocabulary).

    Args:
        raster: Dataset on a bucketed, calendar-contiguous time axis
            (carrying a `TimeSpec`; a gappy `resample` result needs
            `interpolate(raster, "nearest")` or `"linear"` first — a gap
            would break the bucket-position-to-calendar-time correspondence
            this relies on).
        slot: Window length, in buckets.
        stride: Buckets between consecutive window starts. None defaults to
            `slot`, giving non-overlapping, back-to-back windows.

    Returns:
        Windows in walk order, each a Dataset on `slot` buckets of
        `raster`'s own time axis, unchanged otherwise.

    Raises:
        ValueError: `raster` carries no `TimeSpec`, its time axis is not
            calendar-contiguous at its own recorded cadence, or `slot`
            exceeds the axis length.

    Examples:
        >>> windows = transform.time.time_window(monthly, 3, stride=1)
        >>> len(windows), windows[0].sizes["time"]
        (10, 3)
    """
    spec = _require_timespec(raster)
    if slot <= 0:
        raise ValueError(f"slot must be positive, got {slot}")
    step = slot if stride is None else stride
    if step <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    _require_contiguous(raster.time.values, spec, "raster's time axis")

    length = raster.sizes["time"]
    if slot > length:
        raise ValueError(
            f"slot {slot} is larger than the raster's own {length} buckets; "
            f"a window is cut from the raster's own axis, so it cannot exceed it"
        )

    # length - slot + 1 is the last start that still fits a full slot inside the axis.
    return [
        raster.isel(time=slice(start, start + slot))
        for start in range(0, length - slot + 1, step)
    ]


def time_window_stack(
    tree: xr.DataTree,
    slot: Mapping[str, int],
    *,
    stride: int,
) -> list[xr.DataTree]:
    """Cut every group of a raster stack into fixed-length windows along time.

    Every group already shares one exact time axis (`stack` refuses one that
    does not), so one shared `stride` walk gives each group its own `slot`
    of context around the same calendar position.

    Args:
        tree: Raster stack whose groups share one bucketed time axis.
        slot: Window length in buckets, keyed by group name.
        stride: Buckets between consecutive window starts, shared across
            every group so a window index means one calendar position
            regardless of group.

    Returns:
        Windowed stacks in walk order, each holding every group cut to its
        own `slot` length around the same calendar position.

    Raises:
        ValueError: `tree` holds no group, a group's name is missing from
            `slot`, a group's time axis is not calendar-contiguous, or a
            `slot` value exceeds the shared axis length.

    Examples:
        >>> windows = transform.time.time_window_stack(
        ...     scene, {"sentinel-2-l2a": 3, "dem": 1}, stride=1
        ... )
        >>> windows[0].gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    from geosave_engine.geodata.core.stack import GeoStack

    groups = GeoStack(tree).groups
    if not groups:
        raise ValueError("a raster stack needs at least one group to window")
    missing = sorted(set(groups) - set(slot))
    if missing:
        raise ValueError(f"{missing} are groups of this stack but name no slot")
    invalid = sorted(name for name in groups if slot[name] <= 0)
    if invalid:
        raise ValueError(f"slot for {invalid} must be positive")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    root = tree.dataset
    if "time" not in root.coords:
        raise ValueError("this stack carries no time axis to window")

    # Some group must be the resample() call that established the shared axis.
    specs = (
        GeoRaster(tree[name].to_dataset()).attrs.root.get(TimeSpec) for name in groups
    )
    spec = next((found for found in specs if isinstance(found, TimeSpec)), None)
    if spec is None:
        raise ValueError(
            "no group carries a TimeSpec, so the shared axis cannot be checked "
            "for gaps; at least one group must come from resample()"
        )

    _require_contiguous(root.time.values, spec, "the stack's shared time axis")

    axis_length = root.sizes["time"]
    widest = max(slot[name] for name in groups)
    if widest > axis_length:
        raise ValueError(
            f"slot {widest} is larger than the shared axis's own {axis_length} "
            f"buckets; a window is cut from the stack's own axis, so it cannot "
            f"exceed it"
        )

    # Groups no longer share one time length, so the root keeps only the grid.
    spatial_root = root.drop_vars(["time", "time_bnds"], errors="ignore")
    grouped = {name: tree[name].to_dataset() for name in groups}
    # The widest slot decides how far in the first anchor can sit.
    return [
        xr.DataTree.from_dict(
            {
                "/": spatial_root,
                **{
                    f"/{name}": raster.isel(
                        time=slice(anchor - slot[name] + 1, anchor + 1)
                    )
                    for name, raster in grouped.items()
                },
            }
        )
        for anchor in range(widest - 1, axis_length, stride)
    ]


def concat(rasters: Sequence[xr.Dataset]) -> xr.Dataset:
    """Join rasters end to end along time.

    Operands are joined in the order given. Labels out of order refuse rather
    than sort, and a differing dtype refuses rather than promote.

    Args:
        rasters: Datasets on one exact GeoBox, carrying the same data
            variables at the same dtypes, and ascending non-overlapping time
            labels, at least one.

    Returns:
        New Dataset spanning every operand's time labels, ascending.

    Raises:
        ValueError: `rasters` is empty, an operand carries no `time`
            coordinate, only some operands record a resample, resampled
            operands record different cadences, the operands sit on different
            grids, name different data variables, store one at different
            dtypes, carry duplicate or descending time labels, or disagree on
            an attr.

    Examples:
        >>> transform.time.concat([january, february]).sizes["time"]
        59
    """
    if not rasters:
        raise ValueError("concatenating along time needs at least one raster")

    reference, *others = rasters
    for position, raster in enumerate(rasters):
        if "time" not in raster.coords:
            raise ValueError(
                f"raster {position} carries no 'time' coordinate, so there is "
                f"no axis to join it along"
            )

    carried = [raster.gs.attrs.root.get(TimeSpec) for raster in rasters]
    resampled = [spec is not None for spec in carried]
    if any(resampled) and not all(resampled):
        raise ValueError(
            "some operands record a resample and some do not; bucket them all to "
            "one cadence or none before joining, so the join carries one time_bnds"
        )
    if all(resampled):
        specs = [cast("TimeSpec", spec) for spec in carried]
        mismatched = [
            position for position, spec in enumerate(specs) if spec != specs[0]
        ]
        if mismatched:
            position = mismatched[0]
            raise ValueError(
                f"raster {position} records {specs[position]!r}, but raster 0 "
                f"records {specs[0]!r}; bucket every operand at the same cadence "
                f"before joining, or their time_bnds mean different things"
            )

    grid = GeoRaster(reference).geobox
    dtypes = {str(name): array.dtype for name, array in reference.data_vars.items()}
    for position, raster in enumerate(others, start=1):
        other_grid = GeoRaster(raster).geobox
        if other_grid != grid:
            raise ValueError(
                f"raster {position} is on {other_grid} but raster 0 is on "
                f"{grid}; reproject or resample it onto one grid before joining"
            )
        other_dtypes = {
            str(name): array.dtype for name, array in raster.data_vars.items()
        }
        if other_dtypes.keys() != dtypes.keys():
            raise ValueError(
                f"raster {position} carries {sorted(other_dtypes)} but raster 0 "
                f"carries {sorted(dtypes)}; joining along time needs one "
                f"variable set, so select or merge them first"
            )
        promoted = sorted(
            name for name, dtype in other_dtypes.items() if dtype != dtypes[name]
        )
        if promoted:
            raise ValueError(
                f"raster {position} stores {promoted} at a different dtype from "
                f"raster 0; cast one side first, because joining them would "
                f"promote the pixels silently"
            )

    labels = np.concatenate([raster.time.values for raster in rasters])
    if np.unique(labels).size != labels.size:
        raise ValueError(
            "the rasters carry overlapping time labels; drop the repeats before "
            "joining, because a repeated label makes selection ambiguous"
        )
    if not bool((np.diff(labels) > np.timedelta64(0)).all()):
        raise ValueError(
            "the rasters' time labels descend across the operands; pass them in "
            "ascending order, because joining does not sort them"
        )

    joined = xr.concat(rasters, dim="time", join="exact")
    return stamp(joined, combine(rasters))

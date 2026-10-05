"""Cutting a time axis into fixed-length sequences a model reads one at a time.

A frame holds `length` consecutive instants and nothing else says which one it
is — its own time labels do. `frames` cuts one raster; `stack_frames` cuts a
stack, laying groups that observe on their own cadences onto one axis first.

Examples:
    A year of monthly composites, read as half-years overlapping by a
    quarter, then cut into tiles::

        halves = frames(monthly, 6, stride=3)
        tiles = Tiles(halves, (256, 256), overlap=32)

    Radar and optical observed on different days, read four instants at a
    time, each group keeping the dates it actually observed on::

        cut = stack_frames(scene, 4, tolerance="10D")
"""

from __future__ import annotations

import warnings
from functools import reduce
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.profile import TIME_COORDINATE
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.errors import (
    DroppedInstantsWarning,
    DroppedFramesWarning,
    UncoveredInstantsWarning,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from geosave_engine.geodata import DataTree

# How far an instant may reach for a scene, as pandas reads a duration.
type Tolerance = str | pd.Timedelta | timedelta | np.timedelta64

# What a frame does when a group observed nothing near one of its instants.
type FrameMode = Literal["strict", "drop"]


def _spans(labels: np.ndarray, length: int, stride: int | None) -> list[slice]:
    """Place frames along a time axis, warning about instants none reaches.

    Frames start at the first instant and stride forward, so whatever does not
    fill a frame is left at the end.

    Args:
        labels: The axis's time labels, in axis order.
        length: Instants each frame holds, at most the axis length.
        stride: Instants between the starts of consecutive frames. None
            strides by `length`, so frames abut and cover each instant once.
            Below `length` they overlap; above it they skip instants between.

    Returns:
        One slice per frame, in axis order.

    Raises:
        ValueError: `length` or `stride` is below 1, or `length` exceeds the axis.

    Warns:
        DroppedInstantsWarning: Instants at the end fill no frame.

    Examples:
        Eight instants in frames of three, striding by two. Instant 7 is
        reached by no frame, so it is warned about rather than dropped
        quietly::

            instant  0  1  2  3  4  5  6  7
                     [-----]
                           [-----]
                                 [-----]
                                          ^  fills no frame

        >>> [(span.start, span.stop) for span in _spans(labels, 3, stride=2)]
        [(0, 3), (2, 5), (4, 7)]
    """
    if length < 1:
        raise ValueError(
            f"a frame of {length} instants holds no observations; length counts the "
            f"instants one frame spans, so pass at least 1"
        )
    step = length if stride is None else stride
    if step < 1:
        raise ValueError(
            f"a stride of {step} never advances the frame; pass at least 1, or "
            f"leave it None to stride by length"
        )
    if length > len(labels):
        raise ValueError(
            f"a frame of {length} instants does not fit the {len(labels)} this "
            f"time axis spans; cut frames of at most {len(labels)}, or load a longer "
            f"series"
        )

    end = length + (len(labels) - length) // step * step
    if end < len(labels):
        warnings.warn(
            f"{[str(label) for label in labels[end:]]} at the end of the time "
            f"axis do not fill a frame of {length}, so they were left out; cut "
            f"at a length and stride the axis divides by",
            DroppedInstantsWarning,
            stacklevel=3,
        )
    return [slice(start, start + length) for start in range(0, end - length + 1, step)]


def frames[T: (xr.DataArray, xr.Dataset)](
    data: T, length: int, *, stride: int | None = None
) -> tuple[T, ...]:
    """Cut a time axis into fixed-length frames of consecutive instants.

    Frames are taken by position, so nothing is read off the time labels and
    no pixel moves. This is `stack_frames` for a lone raster, whose instants
    are already the sequence.

    Args:
        data: DataArray or Dataset spanning a labelled `time` dimension.
        length: Instants each frame holds, at most the axis length.
        stride: Instants between the starts of consecutive frames. None
            strides by `length`, so frames abut and cover each instant once.
            Below `length` they overlap; above it they skip instants between.

    Returns:
        One frame per stride position, in axis order, each of the kind given
        and spanning `length` instants. Variables that do not span `time` pass
        into every frame whole.

    Raises:
        ValueError: `data` carries no `time` dimension coordinate, `length` or
            `stride` is below 1, or `length` exceeds the axis length.

    Warns:
        DroppedInstantsWarning: Instants at the end did not fill a frame.

    Examples:
        Twelve monthly composites as four abutting quarters::

            time     Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
                     [---------]
                                 [---------]
                                             [---------]
                                                         [---------]

        >>> quarters = frames(monthly, 3)
        >>> len(quarters), str(quarters[1].time.values[0])[:10]
        (4, '2024-04-01')

        Half-years overlapping by a quarter reach every month of the same
        year, because three strides of 3 plus a frame of 6 is exactly 12::

            time     Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
                     [---------------------]
                                 [---------------------]
                                             [---------------------]

        >>> [str(half.time.values[0])[:7] for half in frames(monthly, 6, stride=3)]
        ['2024-01', '2024-04', '2024-07']

        Seven monthly instants in frames of three leave July over, which
        warns rather than passing unnoticed::

            time     Jan Feb Mar Apr May Jun Jul
                     [---------]
                                 [---------]
                                             ^  DroppedInstantsWarning

        >>> len(frames(seven_months, 3))
        2
    """
    if TIME_COORDINATE not in data.dims or TIME_COORDINATE not in data.coords:
        raise ValueError(
            f"{type(data).__name__} carries no {TIME_COORDINATE!r} dimension "
            f"coordinate, so it holds no sequence of instants to cut into frames; load it "
            f"over time and label its time axis first"
        )
    labels = data.coords[TIME_COORDINATE].values
    return tuple(
        data.isel({TIME_COORDINATE: span}) for span in _spans(labels, length, stride)
    )


def _joint_axis(
    rasters: Mapping[str, xr.Dataset], tolerance: pd.Timedelta
) -> pd.DataFrame:
    """Lay every group's instants onto one axis and match each group back to it.

    The axis spans only the interval every group covers, but matching searches
    each group's whole axis, so a scene just outside that interval still
    answers an instant on its edge. Instants naming no new scene are dropped.

    Args:
        rasters: Group name mapped to a raster spanning a labelled `time`
            dimension.
        tolerance: How far an instant may reach for a scene.

    Returns:
        One row per joint instant, indexed by it, and one column per group,
        holding that group's row into its own time axis or `-1` where it
        observed nothing within `tolerance`.

    Raises:
        ValueError: A group spans no instants, carries an unlabelled one, or
            has labels that are not chronological or repeat; the groups share
            no interval; or a group matched no instant at all.

    Warns:
        UncoveredInstantsWarning: A group observes outside the shared interval.

    Examples:
        Optical and radar observing on their own dates, paired within ten
        days. Only the interval both cover names instants, though a scene
        outside it still answers one on the edge::

            s2    0=Jan10   1=Mar08   2=Mar28   3=Apr12   4=Jul03
            s1              0=Mar05             1=Apr04   2=Jul01
                            |<----- joint coverage ----->|

        >>> _joint_axis({"s2": s2, "s1": s1}, pd.Timedelta("10D"))
                    s2  s1
        2024-03-05   1   0
        2024-03-28   2   1
        2024-04-12   3   1
        2024-07-01   4   2

        Jan10 falls outside the coverage and names no instant; Jul03 falls
        outside it too but still answers the last one. Mar08 named the same
        pair as Mar05 and was dropped, and s1's row 1 answers two instants.
    """
    labels: dict[str, pd.DatetimeIndex] = {}
    for name, raster in rasters.items():
        index = pd.DatetimeIndex(raster.coords[TIME_COORDINATE].values)
        if index.empty:
            raise ValueError(
                f"{name!r} spans no instants at all, so it answers nothing and "
                f"no interval reaches it; drop the group or load it over a "
                f"period it observed"
            )
        if index.hasnans:
            raise ValueError(
                f"{name!r} carries an unlabelled instant (NaT), which is nearest "
                f"to nothing; drop those scenes or label when they were observed"
            )
        if not index.is_monotonic_increasing:
            raise ValueError(
                f"{name!r}'s time labels are not chronological, so no scene of "
                f"it is nearest to an instant; sort its time axis first"
            )
        if index.has_duplicates:
            raise ValueError(
                f"{name!r} observes the same instant more than once, so an "
                f"instant matches no single scene of it; composite it to one "
                f"scene per instant first"
            )
        labels[name] = index

    start = max(index[0] for index in labels.values())
    stop = min(index[-1] for index in labels.values())
    if start > stop:
        periods = {name: f"{index[0]}..{index[-1]}" for name, index in labels.items()}
        raise ValueError(
            f"the groups observe over no shared interval ({periods}), so no "
            f"instant reaches every one of them; load them over one period"
        )

    inside = {
        name: index[(index >= start) & (index <= stop)]
        for name, index in labels.items()
    }
    outside = {
        name: len(labels[name]) - len(index)
        for name, index in inside.items()
        if len(index) < len(labels[name])
    }
    if outside:
        warnings.warn(
            f"{outside} instants lie outside {start}..{stop}, the interval every "
            f"group covers, so they name no instant of the joint axis; the groups "
            f"observe over different periods",
            UncoveredInstantsWarning,
            stacklevel=3,
        )

    instants = reduce(pd.Index.union, inside.values())
    table = pd.DataFrame(
        {
            name: index.get_indexer(instants, method="nearest", tolerance=tolerance)
            for name, index in labels.items()
        },
        index=instants,
    )

    unjoined = sorted(table.columns[table.eq(-1).all()])
    if unjoined:
        raise ValueError(
            f"{unjoined} observed nothing within {tolerance} of any of the "
            f"{len(table)} joint instants, so they join none of them; widen the "
            f"tolerance or check that they cover the same period"
        )

    # An instant naming the same scene in every group as the one before says nothing new.
    changed = table.ne(table.shift()).any(axis=1)
    return table.loc[changed]


def stack_frames(
    tree: xr.DataTree,
    length: int,
    *,
    tolerance: Tolerance,
    stride: int | None = None,
    mode: FrameMode = "strict",
) -> tuple[DataTree, ...]:
    """Cut a stack into frames, joining groups that observe on their own dates.

    Groups need not observe together: their instants are laid onto one joint
    axis, each instant of which takes the scene every group observed nearest to
    it. Each group keeps its own dates, and one spanning no `time` joins whole.

    Args:
        tree: Stack whose time-spanning groups each carry a chronological
            `time` dimension coordinate.
        length: Joint instants each frame holds, at most the axis length.
        tolerance: How far an instant may reach for a scene, e.g. `"10D"`. A
            gap exactly this long still reaches; an instant equidistant from
            two scenes takes the later. A group observing nothing that near an
            instant leaves it incomplete.
        stride: Joint instants between the starts of consecutive frames. None
            strides by `length`.
        mode: What an incomplete instant does. `"strict"` refuses to emit any
            frame covering one; `"drop"` emits the rest.

    Returns:
        One frame per stride position, each a stack of the same groups, its
        time-spanning ones spanning `length` instants.

    Raises:
        ValueError: `mode` names no mode; `tolerance` is not a positive
            duration; no group spans `time`; `length` or `stride` is below 1;
            `length` exceeds the joint axis; a group spans no instants, carries
            an unlabelled one, or has labels that are unchronological or
            repeat; the groups share no interval; a group matches no instant;
            or `mode` is `"strict"` and a frame covers an incomplete instant.

    Warns:
        UncoveredInstantsWarning: A group observes outside the shared interval.
        DroppedInstantsWarning: Instants at the end did not fill a frame.
        DroppedFramesWarning: `mode="drop"` skipped a frame.

    Examples:
        Optical, radar and a timeless DEM, read two instants at a time. The
        joint axis is built first, then cut; each group answers with the
        dates it actually observed on::

            s2     Jan10   Mar08   Mar28   Apr12   Jul03
            s1             Mar05           Apr04   Jul01
            dem    (no time axis — joins every frame whole)

            joint  Mar05   Mar28   Apr12   Jul01
                   [--------------]                 frame 0
                                   [-------------]  frame 1

        >>> cut = stack_frames(scene, 2, tolerance="10D")
        >>> len(cut)
        2
        >>> [str(d)[:10] for d in cut[0]["s2"].time.values]
        ['2024-03-08', '2024-03-28']
        >>> [str(d)[:10] for d in cut[0]["s1"].time.values]
        ['2024-03-05', '2024-04-04']
        >>> tuple(cut[0]["dem"].sizes)
        ('y', 'x')

        Tighten the tolerance and instants no group reaches appear, so no
        frame covering one can be filled. Here `"drop"` leaves nothing at
        all, which is the cost of never inventing a scene::

            joint  Mar05   Mar28   Apr04   Apr12   Jul01
            s2       1       2      none     3       4
            s1       0      none     1      none     2

        >>> stack_frames(scene, 2, tolerance="5D")
        Traceback (most recent call last):
        ValueError: ['s1'] observed nothing within 5 days of 2024-03-28 ...
        >>> len(stack_frames(scene, 2, tolerance="5D", mode="drop"))
        0
    """
    if mode not in ("strict", "drop"):
        raise ValueError(
            f"{mode!r} is not a frame mode; refuse an incomplete instant with "
            f"'strict' or skip over it with 'drop'"
        )
    reach = pd.Timedelta(tolerance)
    if not isinstance(reach, pd.Timedelta) or reach <= pd.Timedelta(0):
        raise ValueError(
            f"a tolerance of {tolerance!r} reaches no scene; pass a positive "
            f"duration such as '10D'"
        )

    rasters = tree.gs.rasters
    series = {
        name: raster
        for name, raster in rasters.items()
        if TIME_COORDINATE in raster.dims and TIME_COORDINATE in raster.coords
    }
    if not series:
        raise ValueError(
            f"no group of this stack spans a {TIME_COORDINATE!r} dimension "
            f"coordinate, so it holds no sequence to cut into frames; load it over time "
            f"and label its time axis first"
        )

    table = _joint_axis(series, reach)
    complete = (table.to_numpy() != -1).all(axis=1)

    cut: list[DataTree] = []
    skips = 0
    for span in _spans(table.index.to_numpy(), length, stride):
        rows = table.iloc[span]
        if not complete[span].all():
            if mode == "strict":
                incomplete = span.start + int(complete[span].argmin())
                absent = sorted(table.columns[table.iloc[incomplete].to_numpy() == -1])
                raise ValueError(
                    f"{absent} observed nothing within {reach} of "
                    f"{table.index[incomplete]}, so the frame starting at "
                    f"{rows.index[0]} is incomplete; widen the tolerance, "
                    f"interpolate the gap, or pass mode='drop'"
                )
            skips += 1
            continue
        selected = stack(
            {
                name: raster.isel({TIME_COORDINATE: rows[name].to_numpy()})
                if name in series
                else raster
                for name, raster in rasters.items()
            }
        )
        cut.append(attrs.rebase(selected, tree.gs.attrs.root))
    if skips:
        warnings.warn(
            f"{skips} of {skips + len(cut)} frames cover an instant some "
            f"group observed nothing within {reach} of, so they were not emitted; "
            f"widen the tolerance or interpolate the gaps to keep them",
            DroppedFramesWarning,
            stacklevel=2,
        )
    return tuple(cut)

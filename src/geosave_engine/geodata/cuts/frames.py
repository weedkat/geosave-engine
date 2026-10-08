"""Cut windows along time into frames of consecutive instants."""

from __future__ import annotations

import warnings
from functools import reduce
from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from geosave_engine import __path__ as _package_paths
from geosave_engine.geodata.warnings import (
    DroppedFramesWarning,
    DroppedInstantsWarning,
    UncoveredInstantsWarning,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from geosave_engine.geodata import GeoDataFrame

# How far an instant may reach for a scene, as pandas reads a duration.
type Tolerance = str | pd.Timedelta | timedelta | np.timedelta64

# What a frame does when a group observed nothing near one of its instants.
type FrameMode = Literal["strict", "drop"]


def frames(
    windows: gpd.GeoDataFrame,
    length: int,
    *,
    tolerance: Tolerance,
    stride: int | None = None,
    mode: FrameMode = "strict",
) -> GeoDataFrame:
    """Cut each window into frames, joining groups that observe on their own dates.

    Groups need not observe together: their instants are laid onto one joint
    axis, each instant of which takes the scene every group observed nearest
    to it. Each group keeps its own dates, and a timeless one joins whole.

    Args:
        windows: Windows as `stacks` or `chips` lists them, each stating the
            time labels of its groups.
        length: Joint instants each frame holds, at most the axis length.
        tolerance: How far an instant may reach for a scene, e.g. `"10D"`. A
            gap exactly this long still reaches; an instant equidistant from
            two scenes takes the later.
        stride: Joint instants between the starts of consecutive frames. None
            strides by `length`.
        mode: What an incomplete instant does. `"strict"` refuses to emit any
            frame covering one; `"drop"` emits the rest.

    Returns:
        One row per frame, windows in the order given and frames in time
        order. `id` gains `/frame-<n>`, `parent` names the window cut, and
        `times`, `start_datetime` and `end_datetime` state the frame's own
        instants.

    Raises:
        ValueError: `mode` names no mode; `tolerance` is not a positive
            duration; a window has no dated group; `length` or `stride` is
            below 1; `length` exceeds the joint axis; a group's labels are
            unchronological or repeat; the groups share no interval; a group
            matches no instant; or `mode` is `"strict"` and a frame covers an
            incomplete instant.

    Warns:
        UncoveredInstantsWarning: A group observes outside the shared interval.
        DroppedInstantsWarning: Instants at the end did not fill a frame.
        DroppedFramesWarning: `mode="drop"` skipped a frame.

    Examples:
        Optical, radar and a timeless DEM, read two instants at a time::

            s2     Jan10   Mar08   Mar28   Apr12   Jul03
            s1             Mar05           Apr04   Jul01
            dem    (no time labels; joins every frame whole)

            joint  Mar05   Mar28   Apr12   Jul01
                   [--------------]                 frame 0
                                   [-------------]  frame 1

        >>> cut = frames(stacks(items), 2, tolerance="10D")
        >>> cut["id"].tolist()
        ['s0/frame-0', 's0/frame-1']
        >>> cut.iloc[0]["times"]["s2"]
        ['2024-03-08T00:00:00', '2024-03-28T00:00:00']
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

    cut = []
    skips = 0
    for _, window in windows.iterrows():
        series = {
            name: pd.DatetimeIndex(labels)
            for name, labels in window["times"].items()
            if labels is not None
        }
        if not series:
            raise ValueError(
                f"no group of {window['id']!r} states time labels, so it holds "
                f"no sequence to cut into frames; save it over time first"
            )
        table = _joint_axis(series, reach)
        complete = (table.to_numpy() != -1).all(axis=1)

        emitted = 0
        for span in _spans(table.index.to_numpy(), length, stride):
            rows = table.iloc[span]
            if not complete[span].all():
                if mode == "strict":
                    incomplete = span.start + int(complete[span].argmin())
                    absent = sorted(
                        table.columns[table.iloc[incomplete].to_numpy() == -1]
                    )
                    raise ValueError(
                        f"{absent} observed nothing within {reach} of "
                        f"{table.index[incomplete]}, so the frame starting at "
                        f"{rows.index[0]} is incomplete; widen the tolerance, "
                        f"interpolate the gap, or pass mode='drop'"
                    )
                skips += 1
                continue
            taken = {
                name: series[name][rows[name].to_numpy()] if name in series else None
                for name in window["times"]
            }
            dated = [
                stamp
                for stamps in taken.values()
                if stamps is not None
                for stamp in stamps
            ]
            cut.append(
                {
                    **window.to_dict(),
                    "id": f"{window['id']}/frame-{emitted}",
                    "parent": window["id"],
                    "times": {
                        name: None
                        if stamps is None
                        else [s.isoformat() for s in stamps]
                        for name, stamps in taken.items()
                    },
                    "start_datetime": min(dated),
                    "end_datetime": max(dated),
                }
            )
            emitted += 1
    if skips:
        warnings.warn(
            f"{skips} of {skips + len(cut)} frames cover an instant some "
            f"group observed nothing within {reach} of, so they were not emitted; "
            f"widen the tolerance or interpolate the gaps to keep them",
            DroppedFramesWarning,
            skip_file_prefixes=tuple(_package_paths),
        )
    # No window cut leaves an empty table, which still names its columns.
    columns = list(dict.fromkeys([*windows.columns, *[]]))
    return cast(
        "GeoDataFrame",
        gpd.GeoDataFrame(cut, columns=columns, geometry="geometry", crs=windows.crs),
    )


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
            skip_file_prefixes=tuple(_package_paths),
        )
    return [slice(start, start + length) for start in range(0, end - length + 1, step)]


def _joint_axis(
    series: Mapping[str, pd.DatetimeIndex], tolerance: pd.Timedelta
) -> pd.DataFrame:
    """Lay every group's instants onto one axis and match each group back to it.

    The axis spans only the interval every group covers, but matching searches
    each group's whole axis, so a scene just outside that interval still
    answers an instant on its edge. Instants naming no new scene are dropped.

    Args:
        series: Group name mapped to its time labels, in axis order.
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
    for name, index in series.items():
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
            skip_file_prefixes=tuple(_package_paths),
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

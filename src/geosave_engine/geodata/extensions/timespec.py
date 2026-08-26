"""TimeSpec: the time-axis bucketing a raster was actually read through. See TimeSpec for details."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as dt, timedelta
from typing import ClassVar, Literal, Self, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from geosave_engine.geodata.extensions.base import GeoExtension
from geosave_engine.geodata.utils.datetime import DateRange, Freq, edge_rules, freq_offset


def _as_datetimes(index: pd.DatetimeIndex) -> np.ndarray:
    """Timestamps as plain datetimes, microsecond precision.

    Args:
        index: Timestamps to convert.

    Returns:
        Object array of `datetime.datetime`, same length and order.
    """
    return index.to_numpy(dtype="datetime64[us]").astype(object)


class TimeSpec(GeoExtension):
    """How a raster's time axis was bucketed — one resample's own config.

    Evidence a bucketing actually happened, not a caller's claim — set
    only by `resample_time` or carried forward by `concat`, never through
    `rebase()`. The `time` coord holds one label per bucket; `bounds`
    turns those back into the span each one stands for.

    Args:
        freq: Cadence label — any pandas offset alias, e.g. `"5D"`,
            `"ME"`. None reads as `"D"`.
        method: Named reducer each bucket collapsed with, e.g. `"median"`.
            None for a bucket collapsed by a callable, which has no name.
        closed: Which bucket edge was inclusive. None reads as `"left"`.
        label: Which bucket edge the `time` coord holds. None infers it
            from the labels themselves.
        origin: First edge of the bucket grid, resolved — a real
            timestamp, not a strategy name like `"start_day"`. None means
            unrecorded, and `bounds` re-bins the labels it's given.
        offset: Shift applied on top of the origin strategy. Already
            baked into `origin`.
    """

    NAMESPACE: ClassVar[str] = "timespec"
    SETTABLE: ClassVar[bool] = False

    freq: Freq | None = None
    method: str | None = None
    closed: Literal["left", "right"] | None = None
    label: Literal["left", "right"] | None = None
    origin: dt | None = None
    offset: str | timedelta | None = None

    @classmethod
    def from_resample(
        cls,
        times: np.ndarray,
        freq: Freq,
        *,
        method: str | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | dt = "start_day",
        offset: str | timedelta | None = None,
    ) -> TimeSpec:
        """Record one resample, every edge rule resolved to what pandas will use.

        Args:
            times: The time values about to be bucketed — the raw ones,
                before resampling.
            freq: Any pandas offset alias, e.g. `"5D"`, `"ME"`.
            method: Named reducer the buckets collapse with. None for a callable.
            closed: Which bucket edge is inclusive. None takes pandas' default for `freq`.
            label: Which bucket edge labels the result. None takes pandas' default for `freq`.
            origin: Origin strategy or timestamp bucket edges snap to.
            offset: Shift applied on top of `origin`.

        Returns:
            TimeSpec whose `origin` is the grid's resolved first edge, so
            `bounds` can place any label without the raw times.

        Raises:
            ValueError: pandas doesn't know `freq`.
        """
        alias = freq_offset(freq)
        closed, label = edge_rules(alias, closed, label)

        binner = pd.Series(0, index=pd.DatetimeIndex(times)).resample(
            alias, closed=closed, label=label, origin=origin, offset=offset
        ).binner
        return cls(
            freq=alias,
            method=method,
            closed=closed,
            label=label,
            origin=_as_datetimes(cast(pd.DatetimeIndex, binner.take([0])))[0],
            offset=offset,
        )

    def bounds(self, times: np.ndarray) -> list[DateRange]:
        """Span each time label stands for.

        Args:
            times: A `time` coord's own values, ascending. A value inside
                a bucket rather than on its edge gets that bucket's span —
                what a raster that was never resampled carries.

        Returns:
            One inclusive `(start, end)` per entry in `times`, same order.

        Raises:
            ValueError: pandas doesn't know this spec's own `freq`.

        Examples:
            >>> TimeSpec(freq="5D", label="left", closed="left").bounds(times)
            [(datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 5, 23, 59, 59, 999999)), ...]
        """
        labels = pd.DatetimeIndex(times)
        grid = self._grid(labels, freq_offset(self.freq or "D"))

        # a label sits on one edge of its own bucket, so look each one up from the side its own edge isn't on
        right_labelled = self.label == "right" if self.label is not None else labels[0] != grid[0]
        index = grid.searchsorted(labels, side="left" if right_labelled else "right")
        starts, ends = grid[index - 1], grid[index]

        # the open edge belongs to the next bucket along, so shave it to keep the range inclusive
        _microsecond = pd.Timedelta(microseconds=1)
        if self.closed == "right":
            starts = starts + _microsecond
        else:
            ends = ends - _microsecond
        return list(zip(_as_datetimes(starts), _as_datetimes(ends)))

    def _grid(self, labels: pd.DatetimeIndex, alias: str) -> pd.DatetimeIndex:
        """Bucket edges covering every label, running one step past the last.

        Args:
            labels: The time labels to cover.
            alias: Canonical pandas offset alias for this spec's own freq.

        Returns:
            Edge timestamps, ascending — bucket `i` spans `[grid[i], grid[i + 1])`.
        """
        if self.origin is not None:
            offset = to_offset(alias)
            start = self.origin
            # a label older than the recorded origin still sits in a bucket, so step the grid back to reach it
            steps = 0
            while start - steps * offset > labels[0]:
                steps += 1
            return pd.date_range(start=start - steps * offset, end=labels[-1] + offset, freq=alias)
        # nothing recorded, so bin the labels the way a fresh resample would
        binner = pd.Series(0, index=labels).resample(
            alias, closed=self.closed, label=self.label, origin="start_day", offset=self.offset
        ).binner
        return cast(pd.DatetimeIndex, binner)

    @classmethod
    def combine(cls, values: Sequence[GeoExtension | None]) -> Self | None:
        """Keep this bucketing only if every input shares it exactly.

        Args:
            values: This namespace's value from each array being composed.

        Returns:
            The shared TimeSpec, or None when inputs disagree — composing
            differently-resampled rasters is valid, just no longer evidence
            of one bucketing.
        """
        first = values[0]
        return first if isinstance(first, cls) and all(value == first for value in values[1:]) else None


def span_from_times(times: np.ndarray, spec: TimeSpec | None = None) -> DateRange:
    """Span the buckets behind a set of time labels cover.

    Args:
        times: A `time` coord's own values, any order.
        spec: How those labels were bucketed. None reads each as a whole day.

    Returns:
        Inclusive `(start, end)` — earliest label's bucket start to
        latest label's bucket end.
    """
    # only the outermost buckets set the span, so don't expand every step to get it
    edges = (spec or TimeSpec()).bounds(np.array([times.min(), times.max()]))
    return edges[0][0], edges[-1][1]

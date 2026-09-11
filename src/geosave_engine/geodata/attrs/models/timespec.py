"""The time-axis bucketing a raster was read through."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as dt, timedelta
from typing import ClassVar, Final, Literal, Self

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import model_validator

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.geodata.utils.datetime import (
    DateRange,
    Freq,
    edge_rules,
    freq_offset,
)

_CF_CELL_METHOD: Final[dict[str, str]] = {
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "min": "minimum",
    "max": "maximum",
    "std": "standard_deviation",
    "var": "variance",
}


class TimeSpec(AttrsModel):
    """What one label on a raster's time axis stands for.

    A resampled axis names buckets, and this records the resample that cut
    them. An axis stating no cadence names instants, so a raster carrying no
    `TimeSpec` reads as `TimeSpec.instants()`.

    Args:
        freq: Cadence — any pandas offset alias, e.g. `"5D"`, `"MS"`, `"ME"`.
            None for an axis of instants, which states no other field.
        time_method: Named reducer each bucket collapsed with, e.g.
            `"median"`. None for a bucket collapsed by a callable.
        time_closed: Which bucket edge pandas treated as inclusive.
        time_label: Which bucket edge names the bucket in the `time` coord.
        time_origin: Timestamp the edge grid was phased from, or a strategy
            name like `"start_day"`.
        time_offset: Shift added on top of `time_origin`.

    Raises:
        ValueError: A bucket field is stated without a `freq` to cut buckets.

    Examples:
        >>> TimeSpec.instants().bounds(labels)[0]
        array(['2024-01-15T10:30:00.000000', '2024-01-15T10:30:00.000001'], ...)
    """

    NAME: ClassVar[str] = "timespec"

    freq: Freq | None = None
    time_method: str | None = None
    time_closed: Literal["left", "right"] | None = None
    time_label: Literal["left", "right"] | None = None
    time_origin: str | dt | None = None
    time_offset: str | timedelta | None = None

    @model_validator(mode="after")
    def _refuse_buckets_without_a_cadence(self) -> Self:
        """Refuse bucket fields on an axis that cuts no buckets.

        Returns:
            This spec, unchanged.

        Raises:
            ValueError: `freq` is None while another field describes buckets.
        """
        if self.freq is not None:
            return self
        stated = sorted(
            name
            for name in type(self).model_fields
            if name != "freq" and getattr(self, name) is not None
        )
        if stated:
            raise ValueError(
                f"{stated} describe buckets but freq is None, so the axis names "
                f"instants; state a freq or drop them"
            )
        return self

    @classmethod
    def instants(cls) -> Self:
        """Build the spec of an axis whose labels are instants, not buckets.

        Returns:
            Spec stating no cadence, which every raster without its own reads as.
        """
        return cls()

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine a time specification, refusing disagreements.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side did not state it.

        Returns:
            Combined model, and the attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: A time field disagrees or `models` is empty.
        """
        return cls._combine_fields(sides, must_agree=cls.model_fields)

    @property
    def cell_methods(self) -> str | None:
        """CF `cell_methods` phrase describing this bucketing.

        Returns:
            `"time: <name>"` for a reducer with a CF cell-method name, or None
            when the reducer keeps an observed value (`first`, `last`) or has
            no CF equivalent.
        """
        name = _CF_CELL_METHOD.get(self.time_method or "")
        return f"time: {name}" if name is not None else None

    def bounds(self, labels: np.ndarray) -> np.ndarray:
        """Half-open `[start, end)` edges of the bucket each label names.

        Each label sits on one edge of its bucket and the other is one `freq`
        step away, `time_label` saying which. A label on an axis of instants
        spans the one microsecond it names.

        Args:
            labels: The `time` coordinate.

        Returns:
            `(len(labels), 2)` `datetime64[us]` array, row `i` holding label
            `i`'s bucket start and next edge.

        Raises:
            ValueError: pandas doesn't know this spec's own `freq`.
        """
        index = pd.DatetimeIndex(labels)
        if self.freq is None:
            edges = index.to_numpy("datetime64[us]")
            return np.stack([edges, edges + np.timedelta64(1, "us")], axis=1)

        # time_offset already sits in `labels`; consecutive edges stay one step apart.
        step = to_offset(self.freq)
        low, high = (
            (index - step, index)
            if self.time_label == "right"
            else (index, index + step)
        )
        return np.stack(
            [low.to_numpy("datetime64[us]"), high.to_numpy("datetime64[us]")], axis=1
        )

    def timespan(self, labels: np.ndarray) -> DateRange:
        """Read the inclusive period `labels` cover under this spec.

        Args:
            labels: The `time` coordinate.

        Returns:
            First and last covered instant.

        Raises:
            ValueError: pandas doesn't know this spec's own `freq`.

        Examples:
            >>> monthly.timespan(labels)
            (datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 3, 31, 23, 59, 59, 999999))
        """
        edges = self.bounds(labels)
        # The closing edge opens the next bucket, so back off one instant.
        last = edges[:, 1].max() - np.timedelta64(1, "us")
        return edges[:, 0].min().astype(object), last.astype(object)

    @classmethod
    def from_resample(
        cls,
        freq: Freq,
        *,
        method: str | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | dt = "start_day",
        offset: str | timedelta | None = None,
    ) -> TimeSpec:
        """Record a resample call, its edge rules resolved to pandas' defaults.

        Args:
            freq: Any pandas offset alias, e.g. `"5D"`, `"MS"`.
            method: Named reducer the buckets collapse with. None for a callable.
            closed: Which bucket edge is inclusive. None resolves to the alias
                default.
            label: Which bucket edge names the bucket. None resolves to the
                alias default.
            origin: Timestamp the edge grid is phased from, or a strategy name.
            offset: Shift added on top of `origin`.

        Returns:
            TimeSpec holding the resolved call.

        Raises:
            ValueError: pandas doesn't know `freq`.
        """
        alias = freq_offset(freq)
        closed, label = edge_rules(alias, closed, label)
        return cls(
            freq=alias,
            time_method=method,
            time_closed=closed,
            time_label=label,
            time_origin=origin,
            time_offset=offset,
        )

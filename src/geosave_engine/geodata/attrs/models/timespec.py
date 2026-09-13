"""The time-axis bucketing a raster was read through."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as dt, timedelta
from typing import ClassVar, Literal, Self

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


class TimeSpec(AttrsModel):
    """Where the edges fall of the buckets a raster's time labels name.

    A resampled axis names buckets; this records the grid they were cut on.
    What their values were collapsed with is CF `cell_methods`, on the
    variables. An axis carrying no cadence names instants.

    Args:
        time_freq: Cadence — any pandas offset alias, e.g. `"5D"`, `"MS"`,
            `"ME"`. None for an axis of instants, which carries no other field.
        time_closed: Which bucket edge pandas treated as inclusive, always
            resolved to `"left"` or `"right"` rather than left open.
        time_label: Which bucket edge names the bucket in the `time` coord,
            resolved the same way.
        time_origin: Timestamp the edge grid was phased from, or a strategy
            name like `"start_day"`.
        time_offset: Shift added on top of `time_origin`.

    Raises:
        ValueError: A bucket field is set without a `time_freq` to cut
            buckets.

    Examples:
        A monthly axis labels each bucket by its own month start, and the
        bucket runs to the next one:

        >>> monthly = TimeSpec.from_resample("MS")
        >>> monthly.time_freq, monthly.time_closed, monthly.time_label
        ('MS', 'left', 'left')
        >>> monthly.bounds(labels)[0]
        array(['2024-01-01T00:00:00.000000', '2024-02-01T00:00:00.000000'], ...)
    """

    NAME: ClassVar[str] = "timespec"

    time_freq: Freq | None = None
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
            ValueError: `time_freq` is None while another field describes
                buckets.
        """
        if self.time_freq is not None:
            return self
        bucket_fields = sorted(
            name
            for name in type(self).model_fields
            if name != "time_freq" and getattr(self, name) is not None
        )
        if bucket_fields:
            raise ValueError(
                f"{bucket_fields} describe buckets but time_freq is None, so the "
                f"axis names instants; set a time_freq or drop them"
            )
        return self

    @classmethod
    def instants(cls) -> Self:
        """Build the spec of an axis whose labels are instants, not buckets.

        Returns:
            Spec carrying no cadence, which a raster without its own reads as.

        Examples:
            >>> TimeSpec.instants().time_freq is None
            True
        """
        return cls()

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge a time specification, refusing disagreements.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Merged model, and the attr keys it could not keep.

        Raises:
            TypeError: An object carries a different model.
            ValueError: A time field disagrees or `models` is empty.
        """
        return cls._merge_fields(models, must_agree=cls.model_fields)

    def bounds(self, labels: np.ndarray) -> np.ndarray:
        """Half-open `[start, end)` edges of the bucket each label names.

        Each label sits on one edge of its bucket and the other is one
        `time_freq` step away, `time_label` saying which. A label on an axis
        of instants spans the one microsecond it names.

        Args:
            labels: The `time` coordinate.

        Returns:
            `(len(labels), 2)` `datetime64[us]` array, row `i` holding label
            `i`'s bucket start and next edge.

        Raises:
            ValueError: pandas doesn't know this spec's own `time_freq`.
        """
        index = pd.DatetimeIndex(labels)
        if self.time_freq is None:
            edges = index.to_numpy("datetime64[us]")
            return np.stack([edges, edges + np.timedelta64(1, "us")], axis=1)

        # time_offset already sits in `labels`; consecutive edges stay one step apart.
        step = to_offset(self.time_freq)
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
            ValueError: pandas doesn't know this spec's own `time_freq`.

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
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | dt = "start_day",
        offset: str | timedelta | None = None,
    ) -> TimeSpec:
        """Record the bucket grid a resample call cuts, on pandas' own defaults.

        Args:
            freq: Any pandas offset alias, e.g. `"5D"`, `"MS"`.
            closed: Which bucket edge is inclusive. None resolves as pandas
                does: a period-end alias such as `"ME"` closes on the right,
                every other alias on the left.
            label: Which bucket edge names the bucket, resolved the same way.
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
            time_freq=alias,
            time_closed=closed,
            time_label=label,
            time_origin=origin,
            time_offset=offset,
        )

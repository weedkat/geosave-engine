"""TimeSpan: the declared (start, end) window a raster was asked to cover. See TimeSpan for details."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime as dt
from typing import ClassVar

from pydantic import model_validator

from geosave_engine.geodata.extensions.base import GeoExtension
from geosave_engine.geodata.utils.datetime import AnchorDatetime, parse_daterange


class TimeSpan(GeoExtension):
    """Caller-declared `(start, end)` window, paired off its two edges.

    The counterpart to an array's own `time` coord labels: this only has
    to *contain* them, not equal them (see `GeoAnchor._validate_time`).

    Args:
        start_datetime: Range start. None only when `end_datetime` is too.
        end_datetime: Range end. None only when `start_datetime` is too.

    Raises:
        ValueError: Exactly one edge is set, or start follows end.
    """

    NAMESPACE: ClassVar[str] = "timespan"

    start_datetime: dt | None = None
    end_datetime: dt | None = None

    @model_validator(mode="after")
    def _require_whole_span(self) -> TimeSpan:
        """Reject a half-set or reversed time span.

        Raises:
            ValueError: Exactly one edge is set or start follows end.
        """
        start, end = self.start_datetime, self.end_datetime
        if (start is None) != (end is None):
            raise ValueError(f"a time span needs both edges, got start={start!r} end={end!r}")
        if start is not None and end is not None:
            comparable_start = start.astimezone(UTC).replace(tzinfo=None) if start.tzinfo is not None else start
            comparable_end = end.astimezone(UTC).replace(tzinfo=None) if end.tzinfo is not None else end
            if comparable_start > comparable_end:
                raise ValueError(f"time span start must not follow its end, got start={start!r} end={end!r}")
        return self

    @classmethod
    def from_input(cls, value: AnchorDatetime | None) -> TimeSpan:
        """Parse datetime input into a declared span.

        Args:
            value: Datetime string, `(start, end)` pair, or None for a
                timeless span.

        Returns:
            New TimeSpan, both edges None when `value` is None.

        Raises:
            ValueError: `value` is a string that can't be parsed.
        """
        if value is None:
            return cls()
        start, end = parse_daterange(value)
        return cls(start_datetime=start, end_datetime=end)

    @classmethod
    def combine(cls, values: Sequence[GeoExtension | None]) -> None:
        """Never propagate a declared span onto a composed array.

        The composed array's own time labels rarely still fit inside any
        one input's declared window, so the result re-derives its span
        from its own labels instead of inheriting one.

        Args:
            values: This namespace's value from each array being composed.

        Returns:
            None, always.
        """
        return None

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def unix_to_rfc3339(timestamp: float) -> str:
    """Convert a Unix timestamp (seconds since epoch) to an RFC 3339 string.

    Example::

        unix_to_rfc3339(1704067200.0)  # "2024-01-01T00:00:00Z"
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def datetime_to_rfc3339(dt: datetime) -> str:
    """Format a `datetime` as an RFC 3339 string.

    Timezone-naive instances are assumed to be UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def datetime_range_buffer(
    dt: datetime | str,
    delta_before: timedelta | None = None,
    delta_after: timedelta | None = None,
) -> str:
    """Return a STAC datetime range string centred on `dt`.

    Both deltas are optional; omitting one anchors that side to `dt` itself.

    Example::

        datetime_range_buffer(datetime(2023, 6, 1), delta_after=timedelta(days=1))
        # → "2023-06-01T00:00:00Z/2023-06-02T00:00:00Z"
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    start = dt - (delta_before or timedelta())
    end = dt + (delta_after or timedelta())
    return f"{datetime_to_rfc3339(start)}/{datetime_to_rfc3339(end)}"

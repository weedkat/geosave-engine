"""Datetime ranges and raster time-axis bucketing."""

from __future__ import annotations

import re
from datetime import datetime as dt
from datetime import timedelta, timezone
from typing import Literal

from pandas.tseries.frequencies import to_offset

DateRange = tuple[dt, dt]
AnchorDatetime = str | tuple[str, str] | DateRange

# Cadence a time axis is bucketed at — any pandas offset alias; the Literal is just IDE suggestions.
type Freq = Literal["D", "W", "MS", "ME", "QS", "QE", "YS", "YE"] | str

# Offsets pandas buckets closed/labelled on the right by default; everything else goes left.
_END_ANCHORED = frozenset({"W", "ME", "QE", "YE", "BME", "BQE", "BYE"})

# ISO ("2019-05-07T10:30:15") or compact ("20190507T103015") — "-" and ":" separators optional.
_DATETIME_PATTERN = re.compile(
    r"^(?P<year>\d{4})"
    r"(?:-?(?P<month>\d{2})"
    r"(?:-?(?P<day>\d{2})"
    r"(?:[T ](?P<hour>\d{2})"
    r"(?::?(?P<minute>\d{2})"
    r"(?::?(?P<second>\d{2})(?P<fraction>\.\d{1,6})?)?"
    r")?(?P<timezone>Z|[+-]\d{2}:?\d{2})?"
    r")?)?)?$"
)


def _end_anchored(freq: Freq) -> bool:
    """Whether pandas closes and labels `freq`'s buckets on their trailing edge.

    `"W"`, `"ME"`, `"QE"`, `"YE"` and their business variants anchor to the
    period end; every other offset anchors to the start.

    Args:
        freq: Any pandas offset alias, e.g. `"5D"`, `"ME"`.

    Returns:
        True for a period-end offset.

    Raises:
        ValueError: pandas doesn't know `freq`.
    """
    return to_offset(freq_offset(freq)).rule_code.split("-")[0] in _END_ANCHORED


def edge_rules(
    alias: str,
    closed: Literal["left", "right"] | None,
    label: Literal["left", "right"] | None,
) -> tuple[Literal["left", "right"], Literal["left", "right"]]:
    """Resolve which bucket edge is inclusive and which one labels the bucket.

    pandas picks these per frequency, so they are resolved once here rather
    than left open — otherwise a recorded grid cannot be reproduced.

    Args:
        alias: pandas offset alias, e.g. `"5D"`, `"ME"`.
        closed: Caller's choice, or None to take pandas' own for `alias`.
        label: Caller's choice, or None to take pandas' own for `alias`.

    Returns:
        `(closed, label)`, both resolved.

    Raises:
        ValueError: pandas doesn't know `alias`.

    Examples:
        >>> edge_rules("MS", None, None)  # month start anchors left
        ('left', 'left')
        >>> edge_rules("ME", None, None)  # month end anchors right
        ('right', 'right')
    """
    edge: Literal["left", "right"] = "right" if _end_anchored(alias) else "left"
    return closed or edge, label or edge


def naive_utc(value: dt) -> dt:
    """Drop the timezone off a datetime, shifting to UTC when it had one.

    Args:
        value: Aware or naive datetime.

    Returns:
        The same instant with no tzinfo, comparable against a `time` coord's
        own labels, which numpy holds without a timezone.

    Examples:
        >>> naive_utc(dt(2025, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=2))))
        datetime.datetime(2025, 6, 1, 10, 0)
    """
    return (
        value.astimezone(timezone.utc).replace(tzinfo=None)
        if value.tzinfo is not None
        else value
    )


def _parse_timezone(raw: str | None) -> timezone | None:
    """Read a UTC offset off the pattern's timezone group, None for none."""
    if raw is None:
        return None
    if raw == "Z":
        return timezone.utc
    sign = 1 if raw[0] == "+" else -1
    digits = raw[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:4])))


def _parse_daterange(value: str) -> DateRange:
    """Read the inclusive period one ISO or compact timestamp covers.

    Raises:
        ValueError: `value` doesn't match the pattern.
    """
    match = _DATETIME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported datetime value: {value!r}")

    year = int(match.group("year"))
    month = int(match.group("month") or 1)
    day = int(match.group("day") or 1)
    hour = int(match.group("hour") or 0)
    minute = int(match.group("minute") or 0)
    second = int(match.group("second") or 0)
    fraction = match.group("fraction")
    microsecond = round(float(fraction) * 1_000_000) if fraction else 0
    start = dt(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        tzinfo=_parse_timezone(match.group("timezone")),
    )

    if match.group("month") is None:
        end = start.replace(year=year + 1)
    elif match.group("day") is None:
        next_month = (
            start.replace(year=year + 1, month=1)
            if month == 12
            else start.replace(month=month + 1)
        )
        end = next_month
    elif match.group("hour") is None:
        end = start + timedelta(days=1)
    elif match.group("minute") is None:
        end = start + timedelta(hours=1)
    elif match.group("second") is None:
        end = start + timedelta(minutes=1)
    elif fraction is None:
        end = start + timedelta(seconds=1)
    else:
        fraction_digits = len(fraction) - 1  # includes the leading "."
        end = start + timedelta(microseconds=10 ** (6 - fraction_digits))

    return start, end - timedelta(microseconds=1)


def parse_daterange(value: AnchorDatetime) -> DateRange:
    """Read one inclusive `(start, end)` range off a datetime input.

    A partial timestamp covers the whole period it names, so `"2019-05"` is
    the whole of May. A pair of datetimes is already a range and passes
    through untouched.

    Args:
        value: ISO or compact timestamp, a `"<start>/<end>"` interval, a pair
            of either, or a pair of datetimes.

    Returns:
        First and last covered instant, the last at microsecond precision.

    Raises:
        ValueError: A string operand doesn't match the pattern.

    Examples:
        >>> parse_daterange("2019-05-07")
        (datetime.datetime(2019, 5, 7, 0, 0),
         datetime.datetime(2019, 5, 7, 23, 59, 59, 999999))
        >>> parse_daterange("2019-05/2019-06")
        (datetime.datetime(2019, 5, 1, 0, 0),
         datetime.datetime(2019, 6, 30, 23, 59, 59, 999999))
    """
    match value:
        case (dt(), dt()):
            return value
        case (str(), str()):
            left, right = value
            return _parse_daterange(left)[0], _parse_daterange(right)[1]
        case str() if "/" in value:
            left, right = value.split("/", 1)
            return _parse_daterange(left)[0], _parse_daterange(right)[1]
        case str():
            return _parse_daterange(value)


def freq_offset(freq: Freq) -> str:
    """Read the canonical pandas offset alias one cadence buckets on.

    Args:
        freq: Any pandas offset alias — `"D"`, `"5D"`, `"W"`, `"ME"`
            (month end), `"MS"` (month start), `"QE"`, `"YE"`.

    Returns:
        The alias pandas resamples and steps the bucket grid with. One
        cadence has one spelling here, whatever spelling came in.

    Raises:
        ValueError: pandas doesn't know `freq`.

    Examples:
        >>> freq_offset("5d"), freq_offset("W")
        ('5D', 'W-SUN')
    """
    try:
        return to_offset(freq).freqstr  # type: ignore[union-attr]
    except ValueError as e:
        raise ValueError(
            f"Unknown freq {freq!r} — needs a pandas offset alias, e.g. 'D', '5D', 'ME'"
        ) from e


def _compact_token(value: dt, depth: int) -> str:
    """Compact token for `value`, keeping fields up to `depth` (1 year .. 7 microsecond)."""
    parts = [
        f"{value.year:04d}",
        f"{value.month:02d}",
        f"{value.day:02d}",
        f"T{value.hour:02d}",
        f"{value.minute:02d}",
        f"{value.second:02d}",
        f".{value.microsecond:06d}",
    ]
    return "".join(parts[:depth])


def _min_depth(value: dt, bound: Literal[0, 1]) -> int:
    """Coarsest depth whose token parses back to `value` as `bound` (0 start, 1 end)."""
    for depth in range(1, 8):
        if _parse_daterange(_compact_token(value, depth))[bound] == value:
            return depth
    return 7


def format_instant(value: dt) -> str:
    """Compact filename token for one instant, as ESA stamps its granules.

    Args:
        value: Instant at whole-second precision.

    Returns:
        Token shaped `YYYYMMDDTHHMMSS`.

    Raises:
        ValueError: `value` carries sub-second precision, which neither a
            filename nor a GeoTIFF datetime tag holds.

    Examples:
        >>> format_instant(dt(2025, 6, 1, 10, 30, 31))
        '20250601T103031'
    """
    if value.microsecond:
        raise ValueError(
            f"{value} carries sub-second precision, which a filename and "
            f"TIFFTAG_DATETIME both round away; resample the time axis first"
        )
    return _compact_token(value, 6)


def format_stem_dates(value: DateRange) -> str:
    """Compact filename token(s) naming the period a range covers.

    Args:
        value: Inclusive `(start, end)` range, microsecond precision.

    Returns:
        One token where the whole range parses back out of it, else
        `"<start>-<end>"`.

    Examples:
        A whole day is one token; two days need both ends:

        >>> format_stem_dates((dt(2019, 5, 7), dt(2019, 5, 7, 23, 59, 59, 999999)))
        '20190507'
        >>> format_stem_dates((dt(2019, 5, 7), dt(2019, 5, 9, 23, 59, 59, 999999)))
        '20190507-20190509'
    """
    start, end = value
    # coarsest token that parses back to this exact range wins
    for depth in range(1, 8):
        token = _compact_token(start, depth)
        if parse_daterange(token) == (start, end):
            return token
    start_token = _compact_token(start, _min_depth(start, 0))
    end_token = _compact_token(end, _min_depth(end, 1))
    return f"{start_token}-{end_token}"

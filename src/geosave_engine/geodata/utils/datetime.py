"""Datetime range parsing, and the time-axis bucketing a raster is read through."""
from __future__ import annotations

import re
from datetime import datetime as dt
from datetime import timedelta, timezone
from functools import lru_cache
from typing import Literal

import pandas as pd
from pandas.tseries.frequencies import to_offset

DateRange = tuple[dt, dt]
AnchorDatetime = str | tuple[str, str] | DateRange

_SECONDS_PER_DAY = 86_400.0
# A century — long enough that a range's own endpoints can't skew the mean bucket length.
_BUCKET_REFERENCE_SPAN = (pd.Timestamp("1970-01-01"), pd.Timestamp("2070-01-01"))

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

# Trailing "-<compact>" or "-<compact>-<compact>" date on a filename stem, any "_extra" after it ignored.
_STEM_DATE_PATTERN = re.compile(
    r"[-_](?P<start>\d{8}(?:T\d{6})?)(?:[-_](?P<end>\d{8}(?:T\d{6})?))?[^/]*$"
)

def edge_rules(
    alias: str,
    closed: Literal["left", "right"] | None,
    label: Literal["left", "right"] | None,
) -> tuple[Literal["left", "right"], Literal["left", "right"]]:
    """Resolve which bucket edge is inclusive and which one labels the bucket.

    pandas picks these per frequency, so they are resolved once here rather
    than left implicit — otherwise a recorded grid cannot be reproduced.

    Args:
        alias: pandas offset alias, e.g. `"5D"`, `"ME"`.
        closed: Caller's choice, or None to take pandas' own for `alias`.
        label: Caller's choice, or None to take pandas' own for `alias`.

    Returns:
        `(closed, label)`, both resolved.

    Raises:
        ValueError: pandas doesn't know `alias`.
    """
    side: Literal["left", "right"] = (
        "right" if to_offset(alias).rule_code.split("-")[0] in _END_ANCHORED else "left"
    )
    return closed or side, label or side


def bucket_labels(
    window: DateRange,
    freq: Freq,
    *,
    closed: Literal["left", "right"] | None = None,
    label: Literal["left", "right"] | None = None,
) -> pd.DatetimeIndex:
    """Every bucket label one window covers at one cadence.

    The grid comes from the window alone, so two rasters aligned to the same
    window and cadence land on identical labels no matter what either one
    observed.

    Args:
        window: Inclusive `(start, end)` the grid spans.
        freq: Any pandas offset alias, e.g. `"5D"`, `"MS"`.
        closed: Which bucket edge is inclusive. None takes pandas' default.
        label: Which bucket edge labels the bucket. None takes pandas' default.

    Returns:
        Bucket labels in ascending order, one per bucket the window covers.

    Raises:
        ValueError: pandas doesn't know `freq`, or `window` ends before it starts.

    Examples:
        >>> bucket_labels((dt(2024, 1, 1), dt(2024, 3, 31)), "MS")
        DatetimeIndex(['2024-01-01', '2024-02-01', '2024-03-01'], dtype='datetime64[ns]', freq=None)
    """
    start, end = window
    if end < start:
        raise ValueError(f"window ends before it starts: {start} – {end}")
    alias = freq_offset(freq)
    closed, label = edge_rules(alias, closed, label)
    edges = pd.Series(0, index=pd.DatetimeIndex([start, end]))
    return edges.resample(alias, closed=closed, label=label, origin=start).count().index


def naive_utc(value: dt) -> dt:
    """Datetime with no tzinfo, shifted to UTC when it had one.

    Args:
        value: Aware or naive datetime.

    Returns:
        The same instant, tz-naive — comparable against a `time` coord's own
        labels, which numpy holds without a timezone.
    """
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo is not None else value


def _parse_timezone(raw: str | None) -> timezone | None:
    """"Z" -> UTC; "+02:00" / "+0200" -> that offset; None -> None."""
    if raw is None:
        return None
    if raw == "Z":
        return timezone.utc
    sign = 1 if raw[0] == "+" else -1
    digits = raw[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:4])))


def _parse_daterange(value: str) -> DateRange:
    """One ISO/compact string -> the inclusive period it covers.

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
    start = dt(year, month, day, hour, minute, second, microsecond, tzinfo=_parse_timezone(match.group("timezone")))

    if match.group("month") is None:
        end = start.replace(year=year + 1)
    elif match.group("day") is None:
        next_month = start.replace(year=year + 1, month=1) if month == 12 else start.replace(month=month + 1)
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
    """Parse datetime input into one inclusive (start, end) range.

    Examples:
        "2019-05-07"                                  -> (2019-05-07 00:00:00, 2019-05-07 23:59:59.999999)
        "2019-05/2019-06"                              -> (2019-05-01 00:00:00, 2019-06-30 23:59:59.999999)
        ("2019-05", "2019-06-15")                      -> (2019-05-01 00:00:00, 2019-06-15 23:59:59.999999)
        (dt(2019, 5, 7), dt(2019, 5, 7))                -> (dt(2019, 5, 7), dt(2019, 5, 7)) — already resolved, passed through

    Raises:
        ValueError: A string operand doesn't match the pattern.
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
    """Canonical pandas offset alias one freq label buckets on.

    Args:
        freq: Any pandas offset alias — `"D"`, `"5D"`, `"W"`, `"ME"`
            (month end), `"MS"` (month start), `"QE"`, `"YE"`.

    Returns:
        The alias pandas resamples and steps the bucket grid with, e.g.
        `"5D"`. Canonical, so an accepted spelling normalizes.

    Raises:
        ValueError: pandas doesn't know `freq`.
    """
    try:
        return to_offset(freq).freqstr  # type: ignore[union-attr] — to_offset only returns None for None input
    except ValueError as e:
        raise ValueError(f"Unknown freq {freq!r} — needs a pandas offset alias, e.g. 'D', '5D', 'ME'") from e


@lru_cache(maxsize=None)
def bucket_days(freq: Freq) -> float:
    """How many days one bucket of `freq` spans.

    Measured by stepping a fixed century rather than one offset, so which
    edge labels a bucket can't skew it: `"MS"` and `"ME"` agree to 0.08%
    while `"D"` and `"W"` sit 600% apart.

    Args:
        freq: Any pandas offset alias, e.g. `"5D"`, `"ME"`.

    Returns:
        Mean bucket length in days.

    Raises:
        ValueError: pandas doesn't know `freq`.

    Examples:
        >>> round(bucket_days("ME"), 2)
        30.44
    """
    start, end = _BUCKET_REFERENCE_SPAN
    steps = len(pd.date_range(start, end, freq=freq_offset(freq)))
    return (end - start).total_seconds() / steps / _SECONDS_PER_DAY


def is_finer(freq: Freq, than: Freq, *, tolerance: float = 0.05) -> bool:
    """Whether `freq` buckets more finely than `than`, beyond `tolerance`.

    Args:
        freq: Frequency to test.
        than: Frequency to compare against.
        tolerance: Relative gap two frequencies may differ by and still
            count as one size. The default reads `"MS"` and `"ME"` as the
            same month, far below the gap between neighbouring cadences.

    Returns:
        True when `freq`'s buckets are smaller than `than`'s by more than
        `tolerance`.

    Raises:
        ValueError: pandas doesn't know either frequency.

    Examples:
        >>> is_finer("D", than="ME")
        True
        >>> is_finer("ME", than="MS")
        False
    """
    return bucket_days(freq) < bucket_days(than) * (1.0 - tolerance)


def extract_stem_dates(stem: str, pattern: re.Pattern[str] = _STEM_DATE_PATTERN) -> str:
    """Date match from a filename stem, normalized to a single value or a
    "start/end" interval string — ready to hand to parse_daterange, no
    datetime parsing done here.

    Examples:
        "tile-20190507"                   -> "20190507"
        "tile-20190507-20190509"          -> "20190507/20190509"
        "dw_..._22.14_20190923T103015_consensus" -> "20190923T103015"

    Args:
        stem: Filename stem to search.
        pattern: Compiled regex with `start`/`end` named groups (`end`
            optional). Default matches this codebase's own trailing
            `-YYYYMMDD[THHMMSS]` convention — pass a source-specific one
            for anything else (e.g. Copernicus `.SAFE` names embed the
            acquisition date mid-string, not as a trailing suffix).

    Raises:
        ValueError: `stem` doesn't match `pattern`.
    """
    match = pattern.search(stem)
    if match is None:
        raise ValueError(f"Filename doesn't match date pattern {pattern.pattern!r}: {stem!r}")
    start, end = match.group("start"), match.group("end")
    return start if end is None else f"{start}/{end}"


def _compact_token(value: dt, depth: int) -> str:
    """Compact token for `value`, keeping fields up to `depth` (1=year .. 7=microsecond)."""
    parts = [
        f"{value.year:04d}", f"{value.month:02d}", f"{value.day:02d}",
        f"T{value.hour:02d}", f"{value.minute:02d}", f"{value.second:02d}", f".{value.microsecond:06d}",
    ]
    return "".join(parts[:depth])


def _min_depth(value: dt, side: int) -> int:
    """Coarsest depth whose token round-trips `value` back exactly, on `side` (0=start, 1=end)."""
    for depth in range(1, 8):
        if _parse_daterange(_compact_token(value, depth))[side] == value:
            return depth
    return 7


def format_stem_dates(value: DateRange) -> str:
    """Compact filename-suffix token(s) for a range — inverse of extract_stem_dates.

    Args:
        value: Inclusive `(start, end)` range, microsecond precision.

    Returns:
        One token when the whole range parses back out of it, else
        `"<start>-<end>"`.

    Examples:
        (2019-05-07 00:00:00, 2019-05-07 23:59:59.999999) -> "20190507"
        (2019-05-07 00:00:00, 2019-05-09 23:59:59.999999) -> "20190507-20190509"
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

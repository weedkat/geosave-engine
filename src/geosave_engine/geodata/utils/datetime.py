import re
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

DateRange = tuple[dt, dt]
DatePrecision = Literal[
    "year", "month", "day", "hour", "minute", "second", "microsecond"
]
TemporalGranularity = Literal["scene", "day", "month", "year"]
TemporalReduce = Literal["first", "last", "median", "mean"]

_ISO_DATETIME_PATTERN = re.compile(
    r"^(?P<year>\d{4})"
    r"(?:-(?P<month>\d{2})"
    r"(?:-(?P<day>\d{2})"
    r"(?:[T ](?P<hour>\d{2})"
    r"(?::(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?P<fraction>\.\d{1,6})?)?"
    r")?(?P<timezone>Z|[+-]\d{2}:?\d{2})?"
    r")?)?)?$"
)
_GEOTIFF_DATE_SUFFIX_PATTERN = re.compile(r"-(?P<start>\d{8})(?:-(?P<end>\d{8}))?(?:_[^./]*)?$")


def _range_end(start: dt, precision: DatePrecision, fraction_digits: int = 0) -> dt:
    """Find inclusive end for one datetime precision."""
    if precision == "year":
        next_period = start.replace(year=start.year + 1)
    elif precision == "month":
        next_period = (
            start.replace(year=start.year + 1, month=1)
            if start.month == 12
            else start.replace(month=start.month + 1)
        )
    elif precision == "day":
        next_period = start + timedelta(days=1)
    elif precision == "hour":
        next_period = start + timedelta(hours=1)
    elif precision == "minute":
        next_period = start + timedelta(minutes=1)
    elif precision == "second":
        next_period = start + timedelta(seconds=1)
    else:
        next_period = start + timedelta(microseconds=10 ** (6 - fraction_digits))
    return next_period - timedelta(microseconds=1)


def _parse_one(value: str | dt) -> DateRange:
    """Parse one value into its inclusive precision range."""
    if isinstance(value, dt):
        return (value, value)

    if re.fullmatch(r"\d{8}", value):
        start = dt.strptime(value, "%Y%m%d")
        return (start, _range_end(start, "day"))

    match = _ISO_DATETIME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported datetime value: {value!r}")

    groups = match.groupdict()
    precision: DatePrecision
    if groups["fraction"] is not None:
        precision = "microsecond"
        fraction_digits = len(groups["fraction"]) - 1
    else:
        fraction_digits = 0
        precision = cast(
            DatePrecision,
            next(
                field
                for field in ("second", "minute", "hour", "day", "month", "year")
                if groups[field] is not None
            ),
        )

    parsed_value = value.replace("Z", "+00:00")
    if precision == "year":
        start = dt(int(groups["year"]), 1, 1)
    elif precision == "month":
        start = dt(int(groups["year"]), int(groups["month"]), 1)
    else:
        start = dt.fromisoformat(parsed_value)
    return (start, _range_end(start, precision, fraction_digits))


def parse_datetime_range(value: str | dt | tuple[str | dt, str | dt]) -> DateRange:
    """Parse datetime input into one inclusive range.

    Reduced-precision strings cover their whole stated period. Datetime
    objects remain exact instants. Explicit interval endpoints expand to
    the start precision of the first value and end precision of the second.

    Args:
        value: Datetime, reduced ISO string, interval string, or endpoint pair.

    Returns:
        Inclusive `(start, end)` datetime range.

    Raises:
        ValueError: If input is invalid or the range ends before it starts.
    """
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Date range must contain two values, got {len(value)}")
        start = _parse_one(value[0])[0]
        end = _parse_one(value[1])[1]
    elif isinstance(value, str) and "/" in value:
        parts = value.split("/", 1)
        start = _parse_one(parts[0])[0]
        end = _parse_one(parts[1])[1]
    else:
        start, end = _parse_one(value)

    try:
        is_reversed = end < start
    except TypeError as error:
        raise ValueError(
            "Date range endpoints must use matching timezone awareness"
        ) from error
    if is_reversed:
        raise ValueError(
            f"Date range end {end.isoformat()} is before start {start.isoformat()}"
        )
    return (start, end)


def parse_datetime(value: str | dt) -> dt:
    """Parse one datetime and return its range start.

    Args:
        value: Datetime or supported datetime string.

    Returns:
        Parsed range start.
    """
    return parse_datetime_range(value)[0]


def date_range_from_path(path: str | Path) -> DateRange:
    """Extract standard datetime range from a GeoTIFF filename.

    Filename stem must end in ``-YYYYMMDD`` or ``-YYYYMMDD-YYYYMMDD``,
    optionally followed by a ``_suffix`` (e.g. ``-20190923_consensus``) —
    common in raw data releases that tag a file's provenance after its date.

    Args:
        path: GeoTIFF path or filename.

    Returns:
        Inclusive parsed datetime range.

    Raises:
        ValueError: If filename has no standard date suffix.
    """
    filename = Path(path)
    match = _GEOTIFF_DATE_SUFFIX_PATTERN.search(filename.stem)
    if match is None:
        raise ValueError(
            "GeoTIFF filename must end with '-YYYYMMDD' or "
            f"'-YYYYMMDD-YYYYMMDD': {filename.name!r}"
        )

    start = match.group("start")
    end = match.group("end")
    return parse_datetime_range(start if end is None else (start, end))

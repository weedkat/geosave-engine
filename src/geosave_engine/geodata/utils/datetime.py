"""Datetime range parsing."""
import re
from datetime import datetime as dt
from datetime import timedelta, timezone

DateRange = tuple[dt, dt]
AnchorDatetime = str | tuple[str, str] | DateRange

# ISO ("2019-05-07T10:30:15") and compact ("20190507T103015") — "-" and ":"
# separators are optional, unambiguous either way since every field is a
# fixed digit width.
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

# Trailing "-<compact>" or "-<compact>-<compact>" date suffix on a filename
# stem, optionally followed by "_extra" (e.g. "-20190923_consensus" — common
# in raw data releases that tag a file's provenance after its date).
_STEM_DATE_PATTERN = re.compile(
    r"-(?P<start>\d{8}(?:T\d{6})?)(?:-(?P<end>\d{8}(?:T\d{6})?))?(?:_[^./]*)?$"
)

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


def extract_stem_dates(stem: str) -> str:
    """Trailing compact date suffix from a filename stem, normalized to a
    single value or a "start/end" interval string — ready to hand to
    parse_daterange, no datetime parsing done here.

    Examples:
        "tile-20190507"                   -> "20190507"
        "tile-20190507-20190509"          -> "20190507/20190509"
        "dw_..._22.14-20190923_consensus" -> "20190923"

    Raises:
        ValueError: `stem` has no such date suffix.
    """
    match = _STEM_DATE_PATTERN.search(stem)
    if match is None:
        raise ValueError(f"Filename must end with a '-YYYYMMDD' or '-YYYYMMDDTHHMMSS' date suffix: {stem!r}")
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

    Tries coarser depths first and verifies each one by actually re-parsing
    it — a field being at its minimum (e.g. day=1) doesn't by itself mean
    that's the range's real precision, so guessing depth from zeroed fields
    alone is unreliable (a single day landing on the 1st of a month looks
    identical, field-wise, to whole-month precision). Microsecond depth is
    what lets a zero-width exact instant (start == end, e.g. a real STAC
    scene timestamp) round-trip at all — anything coarser always expands to
    at least a 1-second-wide range.

    Examples:
        (2019-05-07 00:00:00, 2019-05-07 23:59:59.999999) -> "20190507"
        (2019-05-07 00:00:00, 2019-05-09 23:59:59.999999) -> "20190507-20190509"
    """
    start, end = value
    for depth in range(1, 8):
        token = _compact_token(start, depth)
        if parse_daterange(token) == (start, end):
            return token
    start_token = _compact_token(start, _min_depth(start, 0))
    end_token = _compact_token(end, _min_depth(end, 1))
    return f"{start_token}-{end_token}"

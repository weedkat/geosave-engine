import re
from datetime import datetime as dt

DEFAULT_DATE_PATTERN: str = r"(?<!\d)(\d{8})(?!\d)"
DEFAULT_DATE_FORMAT: str = "%Y%m%d"


def parse_datetime(s: str | dt) -> dt:
    """Parse a datetime string into a ``datetime`` object.

    Accepts ISO 8601 variants (``"2024-01-15"``, ``"2024-01-15T10:30:00"``,
    ``"2024-01-15 10:30:00"``) and compact date strings (``"20240115"``).

    Args:
        s: Datetime string to parse.
    """
    if isinstance(s, dt):
        return s
    try:
        return dt.fromisoformat(s)
    except ValueError:
        return dt.strptime(s, "%Y%m%d")


def date_from_path(path: str, date_format: str, date_pattern: str) -> dt:
    """Extract a datetime from a file path using regex and strptime.

    Uses the last match so time-series postfixes (``_{YYYYMMDD}.tif``) take
    precedence over the anchor date embedded earlier in the stem.

    Args:
        path: File path or filename to search.
        date_format: strptime format string for the matched date group.
        date_pattern: Regex pattern with one capturing group for the date string.
    """
    matches = re.findall(date_pattern, path)
    if not matches:
        raise ValueError(f"No date found in path '{path}' using pattern '{date_pattern}'")
    return dt.strptime(matches[-1], date_format)

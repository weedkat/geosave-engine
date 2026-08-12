"""Discovery helpers shared by glob-based PyTorch dataset classes (e.g. StackDataset)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def extract_key(name: str, pattern: str | None) -> str:
    """Extract a sample key from a globbed file/folder name.

    Args:
        name: Full name, extension/suffix included (`Path.name`, not
            `Path.stem`) — a custom `pattern` needs the extension present
            to anchor against it (e.g. `r"(\\d+)\\.tif$"`).
        pattern: Regex to extract the key. None strips the last extension
            (same as `Path.stem`) and uses the rest as-is. If given, uses
            the first capture group, or the whole match if the pattern has
            no groups.

    Returns:
        The extracted key.

    Raises:
        ValueError: `pattern` given but doesn't match `name`.
    """
    if pattern is None:
        return Path(name).stem
    match = re.search(pattern, name)
    if match is None:
        raise ValueError(f"key_pattern {pattern!r} did not match {name!r}")
    return match.group(1) if match.groups() else match.group(0)


def filter_by_split(samples: dict[str, Any], split: str | Path | list[str] | None) -> dict[str, Any]:
    """Narrow `samples` to the keys listed in `split`, one key per line.

    Call this on the cheapest dict you have (e.g. paths, before opening
    anything) — filtering happens before whatever's expensive about turning
    a key into a real sample, not after. Keyed internally only, for split
    matching — a dataset's public index doesn't expose these keys.

    Args:
        samples: Discovered samples, keyed by `extract_key`'s output.
        split: Text file of keys to keep or a list of keys. None returns `samples` as-is.

    Returns:
        `samples` filtered to keys present in `split` (missing ones logged,
        not raised — a split file can legitimately be stricter than what's
        on disk).
    """
    if split is None:
        return samples
    if isinstance(split, list):
        wanted = set(split)
    else:
        wanted = {line.strip() for line in Path(split).read_text().splitlines() if line.strip()}
    missing = wanted - samples.keys()
    if missing:
        log.warning("split file %s lists keys not found in samples: %s", split, sorted(missing))
    return {key: value for key, value in samples.items() if key in wanted}

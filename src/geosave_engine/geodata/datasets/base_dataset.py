from __future__ import annotations

import abc
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from geosave_engine.geodata.datasets.intersection_dataset import IntersectionDataset

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
    a key into a real sample, not after. Free function, not a `BaseDataset`
    method — not every subclass has a split concept (`IntersectionDataset`
    doesn't), so this stays opt-in rather than baked into the base.

    Args:
        samples: Discovered samples, any subclass's own shape.
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


class BaseDataset(Dataset, abc.ABC):
    """Base for datasets whose samples are keyed by a string.

    Subclasses call `self.reindex(keys)` once their key set is final and
    implement `render(key)`/`to_row(key)` to build a sample/manifest row.
    Neither has a default body on purpose — a fallback that guesses
    `self.samples[key]` would silently produce the wrong shape for any
    subclass that forgot to override it, instead of failing at
    construction. `samples`/`split` aren't part of this contract at all —
    they're a common convention (most subclasses build a `self.samples`
    dict and take a `split` param, using the free `filter_by_split`
    function above), not something the base declares or requires.
    """

    def reindex(self, keys: Iterable[str]) -> None:
        """Set the key index. Call once, after discovery is final.

        Args:
            keys: Every sample key this dataset has.
        """
        self._index: tuple[str, ...] = tuple(keys)
        if not self._index:
            log.warning("%s: no samples found", type(self).__name__)

    @property
    def keys(self) -> list[str]:
        """Sample keys, in discovery order."""
        return list(self._index)

    @property
    def fields(self) -> list[str]:
        """Dict keys one `render()` call returns — what a sample has.

        Calls `render` on the first key to find out, so this isn't free
        for a `render` that does real work (e.g. `GeoDataset` loading
        pixels). Override for a cheaper peek if one's available.
        """
        if getattr(self, "_index", None) is None or not self._index:
            return []
        return list(self.render(self._index[0]))

    def __len__(self) -> int:
        return len(self._index)

    @abc.abstractmethod
    def render(self, key: str) -> Any:
        """Build one sample's value for `key`.

        Args:
            key: One of `self.keys`.
        """

    def __getitem__(self, index: int) -> Any:
        """Render the sample at `index`.

        Args:
            index: Row index into `self.keys`.
        """
        key = self._index[index]
        return self.render(key)

    @abc.abstractmethod
    def to_row(self, key: str) -> dict[str, Any]:
        """Build one manifest row for `key` — `to_pandas`'s per-sample unit.

        Args:
            key: One of `self.keys`.
        """

    def to_pandas(self) -> pd.DataFrame:
        """Snapshot every sample's `to_row` into one table, keyed by `sample_id`.

        Returns:
            One row per key: `sample_id` column plus whatever `to_row` returns.

        Raises:
            ValueError: A `to_row` result itself has a `"sample_id"` field —
                it would silently collide with the id column added here.
        """
        rows = []
        for key in self.keys:
            row = self.to_row(key)
            if "sample_id" in row:
                raise ValueError(
                    f"{type(self).__name__}.to_row for key {key!r} returned a 'sample_id' "
                    f"field, which collides with to_pandas' own id column"
                )
            rows.append({"sample_id": key, **row})
        return pd.DataFrame(rows)

    def to_parquet(self, path: str | Path) -> None:
        """Snapshot every sample to one parquet file.

        Args:
            path: Output `.parquet` file.
        """
        self.to_pandas().to_parquet(path)

    def __and__(self, other: BaseDataset) -> IntersectionDataset:
        """`a & b` — shorthand for `IntersectionDataset(a, b)`.

        Args:
            other: Another `BaseDataset` to intersect with.
        """
        if not isinstance(other, BaseDataset):
            raise TypeError(f"Expected BaseDataset, got {type(other)}")

        from geosave_engine.geodata.datasets.intersection_dataset import IntersectionDataset

        return IntersectionDataset(self, other)
5  
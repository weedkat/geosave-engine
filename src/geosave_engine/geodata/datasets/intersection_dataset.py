from __future__ import annotations

import logging
from typing import Any

from geosave_engine.geodata.datasets.base_dataset import BaseDataset

log = logging.getLogger(__name__)


class IntersectionDataset(BaseDataset):
    """Merge several `BaseDataset`s down to their common sample keys.

    Args:
        *datasets: Two or more `BaseDataset`s to merge.

    Raises:
        ValueError: Fewer than two datasets given.
    """

    def __init__(self, *datasets: BaseDataset) -> None:
        if len(datasets) < 2:
            raise ValueError(f"IntersectionDataset needs at least 2 datasets, got {len(datasets)}")
        self.datasets = datasets

        primary, *rest = datasets
        common = set(primary.keys)
        for ds in rest:
            missing = common - set(ds.keys)
            if missing:
                log.warning(
                    "%s is missing sample keys present in others, dropped from the intersection: %s",
                    type(ds).__name__,
                    sorted(missing),
                )
            common &= set(ds.keys)

        self.reindex(key for key in primary.keys if key in common)

    def render(self, key: str) -> dict[str, Any]:
        """Merge each dataset's `render(key)`.

        Args:
            key: One of `self.keys`.

        Raises:
            ValueError: Two datasets return an overlapping field name.
        """
        sample: dict[str, Any] = {}
        for ds in self.datasets:
            part = ds.render(key)
            collision = sample.keys() & part.keys()
            if collision:
                raise ValueError(
                    f"{type(ds).__name__}.render field(s) {sorted(collision)} collide with an "
                    f"earlier dataset's — rename one (e.g. NonGeoDataset's layer_name)"
                )
            sample.update(part)
        return sample

    def to_row(self, key: str) -> dict[str, Any]:
        """Merge each dataset's `to_row(key)`.

        Args:
            key: One of `self.keys`.

        Raises:
            ValueError: Two datasets return an overlapping field name.
        """
        row: dict[str, Any] = {}
        for ds in self.datasets:
            part = ds.to_row(key)
            collision = row.keys() & part.keys()
            if collision:
                raise ValueError(
                    f"{type(ds).__name__}.to_row field(s) {sorted(collision)} collide with an "
                    f"earlier dataset's"
                )
            row.update(part)
        return row

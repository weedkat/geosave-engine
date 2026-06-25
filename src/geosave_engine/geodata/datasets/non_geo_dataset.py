from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class NonGeoDataset(Dataset, abc.ABC):
    """Base for non-spatial, pre-chipped tensor datasets.

    For benchmark-style data that is already aligned and chipped to a fixed size:
    plain arrays read from disk and returned as tensor dicts. Unlike
    :class:`~geosave_engine.geodata.datasets.geo_dataset.GeoDataset`, there is no
    GeoTile, geobox, or CRS — and no :class:`GeoTileSampler`; ordering is left to
    the DataLoader's plain ``shuffle`` / ``sampler``.

    Use for image classification or regression where spatial referencing is not
    needed past ingest. Subclasses implement ``__len__`` and ``__getitem__``.

    Args:
        root: Directory holding the pre-chipped samples.
    """

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self.root = Path(root)

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset."""

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample as a ``dict[str, Tensor]``."""

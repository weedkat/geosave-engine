"""CocoDataset is an unimplemented skeleton — guard the stub contract until it's built."""
from __future__ import annotations

import pytest

from geosave_engine.geodata.datasets.coco_dataset import CocoDataset


def test_construction_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        CocoDataset(tmp_path / "annotations.json")

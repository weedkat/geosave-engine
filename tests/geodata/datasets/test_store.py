"""Unit tests for StoreDataset: litdata StreamingDataset over a SampleStore.

Exercises real litdata optimize()/write — multiprocessing, several seconds
per test, marked slow (deselected by default, see pyproject.toml).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.datasets import StoreDataset
from geosave_engine.geodata.spatial import GeoAnchor, GeoStack, GeoTag

pytestmark = pytest.mark.slow

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _tile(names, value=1, dtype="uint16", bbox=BBOX):
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    arr = np.full((len(names), gb.height, gb.width), value, dtype=dtype)
    d = datetime(2024, 1, 15)
    return GeoAnchor(geobox=gb, geotag=GeoTag(datetime=(d, d))).to_geotile(arr, list(names))


def _stacks(n=2):
    return [
        GeoStack(
            rgb=_tile(("r", "g", "b"), value=i, dtype="float32"),
            mask=_tile(("mask",), value=i % 2, dtype="uint8"),
        )
        for i in range(n)
    ]


class TestRead:
    def test_len(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "store", _stacks(3), chunk_size=2, num_workers=1)
        assert len(StoreDataset(str(store.path))) == 3

    def test_sel_bands(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "store", _stacks(2), chunk_size=2, num_workers=1)
        ds = StoreDataset(str(store.path), sel_bands={"rgb": ["b", "r"]})
        assert ds[0]["rgb"].shape[0] == 2

    def test_dtype_override(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "store", _stacks(2), chunk_size=2, num_workers=1)
        ds = StoreDataset(str(store.path), dtype_override={"mask": torch.int64})
        assert ds[0]["mask"].dtype == torch.int64

    def test_getitem_and_iter_both_apply_transform(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "store", _stacks(2), chunk_size=2, num_workers=1)
        ds = StoreDataset(str(store.path), sel_bands={"rgb": ["r"]})
        assert ds[0]["rgb"].shape[0] == 1
        assert next(iter(ds))["rgb"].shape[0] == 1

    def test_arrays_become_writable_tensors(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "store", _stacks(1), chunk_size=2, num_workers=1)
        item = StoreDataset(str(store.path))[0]
        assert isinstance(item["rgb"], torch.Tensor)
        assert isinstance(item["mask"], torch.Tensor)
        item["rgb"] += 1  # would warn/UB on a non-writable (read-only-buffer) tensor


class TestSelBandsErrors:
    def test_unknown_band_name_raises(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "store", _stacks(1), chunk_size=2, num_workers=1)
        ds = StoreDataset(str(store.path), sel_bands={"rgb": ["not_a_band"]})
        with pytest.raises(ValueError, match="not in stored bands"):
            ds[0]


class TestManifest:
    def test_to_pandas_matches_to_parquet(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "store", _stacks(2), chunk_size=2, num_workers=1)
        df = StoreDataset(str(store.path)).to_pandas()

        parquet_path = store.to_parquet(tmp_path / "manifest.parquet")
        pdf = pd.read_parquet(parquet_path)

        assert sorted(df.columns) == sorted(pdf.columns)
        assert "rgb" not in df.columns
        assert "index" in df.columns

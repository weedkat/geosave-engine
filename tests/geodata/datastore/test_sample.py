"""Unit tests for SampleStore: litdata optimize()/StreamingDataset wrapper.

Config-validation tests are fast (no I/O). Write-based tests hit real
litdata optimize() (multiprocessing, several seconds), marked slow.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.datastore import SampleStore
from geosave_engine.geodata.spatial import GeoAnchor, GeoStack, GeoTag

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _tile(names, value=1, dtype="uint16", bbox=BBOX):
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    arr = np.full((len(names), gb.height, gb.width), value, dtype=dtype)
    d = datetime(2024, 1, 15)
    return GeoAnchor(geobox=gb, geotag=GeoTag(datetime=(d, d))).to_geotile(arr, list(names))


def _stacks(n=2):
    return [GeoStack(rgb=_tile(("r", "g", "b"), value=i, dtype="float32")) for i in range(n)]


class TestConfig:
    def test_requires_exactly_one_of_chunk_size_or_bytes(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one"):
            SampleStore(str(tmp_path / "s"))
        with pytest.raises(ValueError, match="exactly one"):
            SampleStore(str(tmp_path / "s"), chunk_size=10, chunk_bytes=10)


@pytest.mark.slow
class TestWriteRead:
    def test_len_fields_getitem(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "s", _stacks(2), chunk_size=2, num_workers=1)
        assert len(store) == 2
        assert set(store.fields) == {"geobox", "geotags", "rgb"}
        assert store[0]["geotags"]["rgb"]["bands"] == ["r", "g", "b"]

    def test_rewrite_with_different_config_raises(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "s", _stacks(1), chunk_size=2, num_workers=1)
        with pytest.raises(ValueError, match="different config"):
            SampleStore(str(store.path), chunk_size=4, num_workers=1)

    def test_to_parquet_drops_array_fields(self, tmp_path, write_sample_store):
        store = write_sample_store(tmp_path / "s", _stacks(2), chunk_size=2, num_workers=1)
        path = store.to_parquet(tmp_path / "manifest.parquet")
        df = pd.read_parquet(path)
        assert len(df) == 2
        assert "rgb" not in df.columns
        assert "index" in df.columns


class TestRepr:
    def test_not_written_yet_does_not_raise(self, tmp_path):
        store = SampleStore(str(tmp_path / "s"), chunk_size=2)
        assert "SampleStore" in repr(store)  # would raise if __repr__ propagated the read failure

    @pytest.mark.slow
    def test_shows_layers_and_never_leaks_storage_options(self, tmp_path, write_sample_store):
        store = write_sample_store(
            tmp_path / "s", _stacks(1), chunk_size=2, num_workers=1,
            storage_options={"aws_secret_access_key": "FAKESECRETVALUE"},
        )
        text = repr(store)
        assert "rgb" in text
        assert "FAKESECRETVALUE" not in text

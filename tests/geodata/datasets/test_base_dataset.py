"""Unit tests for BaseDataset (via a toy concrete subclass), extract_key, filter_by_split."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from geosave_engine.geodata.datasets.base_dataset import BaseDataset, extract_key, filter_by_split
from geosave_engine.geodata.datasets.intersection_dataset import IntersectionDataset


class _ToyDataset(BaseDataset):
    def __init__(self, samples: dict[str, int]):
        self.samples = samples
        self.reindex(samples)

    def render(self, key: str) -> dict:
        return {"value": self.samples[key]}

    def to_row(self, key: str) -> dict:
        return {"value": self.samples[key]}


class TestBaseDataset:
    def test_len_and_keys_reflect_reindex(self):
        ds = _ToyDataset({"a": 1, "b": 2})
        assert len(ds) == 2
        assert ds.keys == ["a", "b"]

    def test_getitem_renders_by_index(self):
        ds = _ToyDataset({"a": 1, "b": 2})
        assert ds[1] == {"value": 2}

    def test_fields_peeks_first_key(self):
        ds = _ToyDataset({"a": 1})
        assert ds.fields == ["value"]

    def test_fields_empty_when_no_samples(self):
        ds = _ToyDataset({})
        assert ds.fields == []

    def test_reindex_warns_on_empty(self, caplog):
        with caplog.at_level(logging.WARNING):
            _ToyDataset({})
        assert "no samples found" in caplog.text

    def test_to_pandas_one_row_per_key(self):
        ds = _ToyDataset({"a": 1, "b": 2})
        df = ds.to_pandas()
        assert list(df["sample_id"]) == ["a", "b"]
        assert list(df["value"]) == [1, 2]

    def test_to_pandas_raises_on_sample_id_collision(self):
        class _Bad(BaseDataset):
            def __init__(self):
                self.reindex(["a"])

            def render(self, key):
                return {}

            def to_row(self, key):
                return {"sample_id": "oops"}

        with pytest.raises(ValueError, match="sample_id"):
            _Bad().to_pandas()

    def test_to_parquet_writes_readable_file(self, tmp_path):
        ds = _ToyDataset({"a": 1, "b": 2})
        out = tmp_path / "manifest.parquet"
        ds.to_parquet(out)
        assert list(pd.read_parquet(out)["value"]) == [1, 2]

    def test_and_operator_builds_intersection_dataset(self):
        a = _ToyDataset({"x": 1})
        b = _ToyDataset({"x": 2})
        merged = a & b
        assert isinstance(merged, IntersectionDataset)
        assert merged.keys == ["x"]

    def test_and_operator_rejects_non_basedataset(self):
        ds = _ToyDataset({"x": 1})
        with pytest.raises(TypeError):
            ds & 5


class TestExtractKey:
    def test_default_strips_extension(self):
        assert extract_key("tile_001.tif", None) == "tile_001"

    def test_pattern_uses_first_group(self):
        assert extract_key("tile_001.tif", r"(\d+)\.tif$") == "001"

    def test_pattern_without_groups_uses_whole_match(self):
        assert extract_key("tile_001.tif", r"tile_\d+") == "tile_001"

    def test_pattern_not_matching_raises(self):
        with pytest.raises(ValueError, match="key_pattern"):
            extract_key("tile_001.tif", r"nope_\d+")


class TestFilterBySplit:
    def test_none_split_returns_as_is(self):
        samples = {"a": 1, "b": 2}
        assert filter_by_split(samples, None) is samples

    def test_split_file_narrows_to_listed_keys(self, tmp_path):
        split = tmp_path / "split.txt"
        split.write_text("a\n\nb\n")
        assert filter_by_split({"a": 1, "b": 2, "c": 3}, split) == {"a": 1, "b": 2}

    def test_split_key_missing_from_samples_logs_not_raises(self, tmp_path, caplog):
        split = tmp_path / "split.txt"
        split.write_text("a\nghost\n")
        with caplog.at_level(logging.WARNING):
            result = filter_by_split({"a": 1}, split)
        assert result == {"a": 1}
        assert "ghost" in caplog.text

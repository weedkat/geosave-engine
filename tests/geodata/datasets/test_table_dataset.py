"""Unit tests for TableDataset: csv/parquet read, id grouping, numeric-vs-plain rendering."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from geosave_engine.geodata.datasets.table_dataset import TableDataset


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "table.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestTableDataset:
    def test_single_row_per_id_gives_scalars(self, tmp_path):
        path = _write_csv(tmp_path, [{"id": "a", "score": 1.5, "label": "cat"}])
        ds = TableDataset(path, id_col="id")
        sample = ds[0]
        assert torch.equal(sample["score"], torch.tensor(1.5))
        assert sample["label"] == "cat"

    def test_multi_row_per_id_gives_lists(self, tmp_path):
        path = _write_csv(
            tmp_path,
            [
                {"id": "a", "score": 1.0, "label": "cat"},
                {"id": "a", "score": 2.0, "label": "dog"},
            ],
        )
        ds = TableDataset(path, id_col="id")
        sample = ds[0]
        assert torch.equal(sample["score"], torch.tensor([1.0, 2.0]))
        assert sample["label"] == ["cat", "dog"]

    def test_len_is_distinct_id_count(self, tmp_path):
        path = _write_csv(tmp_path, [{"id": "a", "v": 1}, {"id": "a", "v": 2}, {"id": "b", "v": 3}])
        ds = TableDataset(path, id_col="id")
        assert len(ds) == 2

    def test_sel_col_restricts_forwarded_columns(self, tmp_path):
        path = _write_csv(tmp_path, [{"id": "a", "keep": 1, "drop": 2}])
        ds = TableDataset(path, id_col="id", sel_col=["keep"])
        assert set(ds[0]) == {"keep"}

    def test_sel_col_unknown_column_raises(self, tmp_path):
        path = _write_csv(tmp_path, [{"id": "a", "v": 1}])
        with pytest.raises(ValueError, match="sel_col"):
            TableDataset(path, id_col="id", sel_col=["ghost"])

    def test_missing_id_col_raises(self, tmp_path):
        path = _write_csv(tmp_path, [{"id": "a", "v": 1}])
        with pytest.raises(ValueError, match="id_col"):
            TableDataset(path, id_col="ghost")

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "table.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="TABLE_EXTENSIONS|path must be one of"):
            TableDataset(path, id_col="id")

    def test_split_filters_ids(self, tmp_path):
        path = _write_csv(tmp_path, [{"id": "a", "v": 1}, {"id": "b", "v": 2}])
        split = tmp_path / "split.txt"
        split.write_text("a\n")
        ds = TableDataset(path, id_col="id", split=split)
        assert ds.keys == ["a"]

    def test_to_row_keeps_plain_values_not_tensors(self, tmp_path):
        path = _write_csv(tmp_path, [{"id": "a", "v": 1.0}])
        ds = TableDataset(path, id_col="id")
        assert ds.to_row("a") == {"v": 1.0}

    def test_reads_parquet(self, tmp_path):
        path = tmp_path / "table.parquet"
        pd.DataFrame([{"id": "a", "v": 1.0}]).to_parquet(path)
        ds = TableDataset(path, id_col="id")
        assert torch.equal(ds[0]["v"], torch.tensor(1.0))

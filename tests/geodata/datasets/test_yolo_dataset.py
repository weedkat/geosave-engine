"""Unit tests for YoloDataset: label parsing, key extraction, split, error paths."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from geosave_engine.geodata.datasets.yolo_dataset import YoloDataset


def _write_label(root: Path, name: str, lines: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("\n".join(lines))


class TestYoloDataset:
    def test_parses_boxes_for_each_label(self, tmp_path):
        _write_label(tmp_path, "img1.txt", ["0 0.5 0.5 0.2 0.2", "1 0.1 0.1 0.05 0.05"])
        ds = YoloDataset(tmp_path)
        sample = ds[0]
        assert torch.equal(sample["class_id"], torch.tensor([0, 1], dtype=torch.int64))
        assert sample["box"].shape == (2, 4)
        assert torch.allclose(sample["box"][0], torch.tensor([0.5, 0.5, 0.2, 0.2]))

    def test_empty_label_file_gives_zero_length_correctly_shaped_tensors(self, tmp_path):
        _write_label(tmp_path, "empty.txt", [])
        ds = YoloDataset(tmp_path)
        sample = ds[0]
        assert sample["class_id"].shape == (0,)
        assert sample["box"].shape == (0, 4)

    def test_malformed_line_raises_value_error(self, tmp_path):
        _write_label(tmp_path, "bad.txt", ["0 0.5 0.5"])
        with pytest.raises(ValueError, match="expected 'class_id cx cy w h'"):
            YoloDataset(tmp_path)

    def test_discovers_labels_nested_under_root(self, tmp_path):
        _write_label(tmp_path / "sub", "a.txt", ["0 0.1 0.1 0.1 0.1"])
        _write_label(tmp_path / "sub2", "b.txt", ["0 0.1 0.1 0.1 0.1"])
        ds = YoloDataset(tmp_path)
        assert len(ds) == 2

    def test_key_pattern_extracts_custom_stem(self, tmp_path):
        _write_label(tmp_path, "labels_042.txt", ["0 0.1 0.1 0.1 0.1"])
        ds = YoloDataset(tmp_path, key_pattern=r"(\d+)\.txt$")
        assert ds.keys == ["042"]

    def test_split_filters_by_stem(self, tmp_path):
        _write_label(tmp_path, "a.txt", ["0 0.1 0.1 0.1 0.1"])
        _write_label(tmp_path, "b.txt", ["0 0.1 0.1 0.1 0.1"])
        split = tmp_path / "split.txt"
        split.write_text("a\n")
        ds = YoloDataset(tmp_path, split=split)
        assert ds.keys == ["a"]

    def test_custom_field_names(self, tmp_path):
        _write_label(tmp_path, "a.txt", ["0 0.1 0.1 0.1 0.1"])
        ds = YoloDataset(tmp_path, class_id_name="cls", box_name="bbox")
        assert set(ds[0]) == {"cls", "bbox"}

    def test_to_row_returns_plain_lists_not_tensors(self, tmp_path):
        _write_label(tmp_path, "a.txt", ["0 0.1 0.1 0.1 0.1"])
        ds = YoloDataset(tmp_path)
        row = ds.to_row("a")
        assert row == {"class_id": [0], "box": [[0.1, 0.1, 0.1, 0.1]]}

"""Unit tests for IntersectionDataset: key intersection, field merge, collisions."""
from __future__ import annotations

import logging

import pytest

from geosave_engine.geodata.datasets.base_dataset import BaseDataset
from geosave_engine.geodata.datasets.intersection_dataset import IntersectionDataset


class _ToyDataset(BaseDataset):
    def __init__(self, samples: dict[str, dict]):
        self.samples = samples
        self.reindex(samples)

    def render(self, key: str) -> dict:
        return dict(self.samples[key])

    def to_row(self, key: str) -> dict:
        return dict(self.samples[key])


class TestIntersectionDataset:
    def test_requires_at_least_two_datasets(self):
        with pytest.raises(ValueError, match="at least 2"):
            IntersectionDataset(_ToyDataset({"a": {"x": 1}}))

    def test_keeps_only_common_keys_in_primary_order(self, caplog):
        a = _ToyDataset({"a": {"x": 1}, "b": {"x": 2}})
        b = _ToyDataset({"b": {"y": 1}, "c": {"y": 2}})
        with caplog.at_level(logging.WARNING):
            ds = IntersectionDataset(a, b)
        assert ds.keys == ["b"]
        assert "a" in caplog.text or "c" in caplog.text

    def test_render_merges_fields_across_datasets(self):
        a = _ToyDataset({"k": {"image": 1}})
        b = _ToyDataset({"k": {"label": 2}})
        ds = IntersectionDataset(a, b)
        assert ds[0] == {"image": 1, "label": 2}

    def test_render_raises_on_field_collision(self):
        a = _ToyDataset({"k": {"image": 1}})
        b = _ToyDataset({"k": {"image": 2}})
        ds = IntersectionDataset(a, b)
        with pytest.raises(ValueError, match="collide"):
            ds[0]

    def test_to_row_merges_and_detects_collision(self):
        a = _ToyDataset({"k": {"image": 1}})
        b = _ToyDataset({"k": {"label": 2}})
        ds = IntersectionDataset(a, b)
        assert ds.to_row("k") == {"image": 1, "label": 2}

        c = _ToyDataset({"k": {"label": 3}})
        collide = IntersectionDataset(b, c)
        with pytest.raises(ValueError, match="collide"):
            collide.to_row("k")

    def test_and_operator_chains_across_three_datasets(self):
        a = _ToyDataset({"k": {"a": 1}})
        b = _ToyDataset({"k": {"b": 2}})
        c = _ToyDataset({"k": {"c": 3}})
        merged = a & b & c
        assert merged[0] == {"a": 1, "b": 2, "c": 3}

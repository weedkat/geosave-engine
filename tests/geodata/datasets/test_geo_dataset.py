"""Unit tests for GeoDataset: anchor-folder discovery, tensor rendering.

No network — tiles are synthetic zarr stores written to a tmp folder.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.tile import GEOSTACK_SUFFIX, GeoAnchor, GeoTile, GeoStack
from geosave_engine.geodata.datasets import GeoDataset, stack_samples

UTM = "EPSG:32633"


def _write_anchor(root: Path, idx: int, bbox, layers: dict[str, tuple[str, ...]]) -> None:
    """Write one anchor folder under root, with the given layers (name -> band names)."""
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    tiles = {}
    for name, band_names in layers.items():
        arr = np.zeros((len(band_names), gb.height, gb.width), dtype="uint16")
        tiles[name] = GeoAnchor(geobox=gb, datetime=datetime(2023, 2, 1)).with_np(arr, list(band_names))
    GeoStack(**tiles).save(root / f"t{idx}{GEOSTACK_SUFFIX}")


def _dataset_root(root: Path, n_anchors: int = 2) -> Path:
    """Write n_anchors anchor folders (s2 + label layers) under root."""
    for i in range(n_anchors):
        x0 = 500000 + i * 320
        bbox = (x0, 5000000, x0 + 320, 5000320)
        _write_anchor(root, i, bbox, {"s2": ("B02", "B03"), "label": ("label",)})
    return root


class TestDiscovery:
    def test_finds_every_anchor_folder(self, tmp_path):
        ds = GeoDataset(_dataset_root(tmp_path, n_anchors=2))
        assert sorted(ds.layers) == ["label", "s2"]
        assert len(ds) == 2

    def test_getitem_renders_tensors(self, tmp_path):
        ds = GeoDataset(_dataset_root(tmp_path))
        sample = ds[0]
        assert isinstance(sample["s2"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)
        assert tuple(sample["s2"].shape) == (2, 32, 32)   # 2 bands, H, W
        assert tuple(sample["label"].shape) == (1, 32, 32)

    def test_anchor_always_present(self, tmp_path):
        ds = GeoDataset(_dataset_root(tmp_path))
        sample = ds[0]
        assert isinstance(sample["anchors"], dict)
        layer = next(iter(sample["anchors"]))
        assert isinstance(sample["anchors"][layer], GeoAnchor)
        assert not isinstance(sample["anchors"][layer], GeoTile)

    def test_single_layer(self, tmp_path):
        _write_anchor(tmp_path, 0, (500000, 5000000, 500320, 5000320), {"s2": ("B02",)})
        ds = GeoDataset(tmp_path)
        assert ds.layers == ["s2"]
        assert len(ds) == 1

    def test_required_layers_excludes_incomplete_anchors(self, tmp_path):
        _write_anchor(tmp_path, 0, (500000, 5000000, 500320, 5000320), {"s2": ("B02",), "label": ("label",)})
        _write_anchor(tmp_path, 1, (500320, 5000000, 500640, 5000320), {"s2": ("B02",)})  # no label
        ds = GeoDataset(tmp_path, required_layers=["s2", "label"])
        assert len(ds) == 1

    def test_finds_anchors_nested_arbitrarily_deep(self, tmp_path):
        """Anchor folders can be grouped in nested subdirectories (e.g. mirroring raw
        data provenance), not just flat directly under root — discovery is by
        .geostack marker via rglob, any depth."""
        _write_anchor(tmp_path / "Experts" / "EH" / "1", 0, (500000, 5000000, 500320, 5000320), {"s2": ("B02",)})
        _write_anchor(tmp_path / "Non_expert" / "WorkForce" / "EH" / "1", 0,
                      (500320, 5000000, 500640, 5000320), {"s2": ("B02",)})
        ds = GeoDataset(tmp_path)
        assert len(ds) == 2


class TestRender:
    def test_stack_samples_batches_per_layer_key(self, tmp_path):
        ds = GeoDataset(_dataset_root(tmp_path), sel_bands={"s2": ["B02"]})
        batch = stack_samples([ds[0], ds[1]])
        assert tuple(batch["s2"].shape) == (2, 1, 32, 32)   # B, C(sel B02), H, W
        assert tuple(batch["label"].shape) == (2, 1, 32, 32)
        assert len(batch["s2"]) == 2

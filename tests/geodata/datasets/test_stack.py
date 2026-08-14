"""Unit tests for StackDataset: GeoStack zarr discovery and rendering.

No network — all tiles are synthetic, built from a projected geobox.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.datasets import StackDataset
from geosave_engine.geodata.spatial import GeoAnchor, GeoStack, GeoTag

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _tile(names, value=1, dtype="uint16", bbox=BBOX):
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    arr = np.full((len(names), gb.height, gb.width), value, dtype=dtype)
    d = datetime(2024, 1, 15)
    return GeoAnchor(geobox=gb, geotag=GeoTag(datetime=(d, d))).to_geotile(arr, list(names))


def _write_stack(path, with_mask=True):
    rgb = _tile(("r", "g", "b"), dtype="float32")
    if not with_mask:
        GeoStack(rgb=rgb).to_zarr(path)
        return
    mask = _tile(("mask",), dtype="uint8")
    GeoStack(rgb=rgb, mask=mask).to_zarr(path)


class TestDiscovery:
    def test_finds_zarr_at_any_depth(self, tmp_path):
        _write_stack(tmp_path / "a" / "0.zarr")
        _write_stack(tmp_path / "b" / "c" / "1.zarr")
        _write_stack(tmp_path / "2.zarr")
        ds = StackDataset(tmp_path)
        assert len(ds) == 3

    def test_required_layers_excludes_missing_anchors(self, tmp_path):
        _write_stack(tmp_path / "with_mask.zarr", with_mask=True)
        _write_stack(tmp_path / "no_mask.zarr", with_mask=False)
        ds = StackDataset(tmp_path, required_layers=["mask"])
        assert len(ds) == 1

    def test_required_layers_keeps_every_layer_not_just_required(self, tmp_path):
        _write_stack(tmp_path / "s.zarr", with_mask=True)
        ds = StackDataset(tmp_path, required_layers=["mask"])
        assert sorted(ds[0].keys()) == ["geobox", "geotags", "mask", "rgb"]

    def test_empty_root(self, tmp_path):
        assert len(StackDataset(tmp_path)) == 0


class TestRender:
    def test_getitem_matches_render(self, tmp_path):
        _write_stack(tmp_path / "s.zarr")
        ds = StackDataset(tmp_path)
        item = ds[0]
        rendered = ds.render(0)
        assert item.keys() == rendered.keys()
        assert torch.equal(item["rgb"], rendered["rgb"])

    def test_sel_bands(self, tmp_path):
        _write_stack(tmp_path / "s.zarr")
        ds = StackDataset(tmp_path, sel_bands={"rgb": ["b", "r"]})
        assert ds[0]["rgb"].shape[0] == 2

    def test_dtype_override(self, tmp_path):
        _write_stack(tmp_path / "s.zarr")
        ds = StackDataset(tmp_path, dtype_override={"mask": torch.int64})
        assert ds[0]["mask"].dtype == torch.int64

    def test_sample_carries_geobox_and_geotags(self, tmp_path):
        _write_stack(tmp_path / "s.zarr")
        item = StackDataset(tmp_path)[0]
        assert sorted(item["geobox"].keys()) == ["affine", "centroid", "crs", "shape"]
        assert sorted(item["geotags"].keys()) == ["mask", "rgb"]


class TestManifest:
    def test_to_row_is_relative_path(self, tmp_path):
        _write_stack(tmp_path / "a" / "0.zarr")
        ds = StackDataset(tmp_path)
        row = ds.to_row(0)
        assert row["path"] == "a/0.zarr"
        assert row["index"] == 0
        assert sorted(row["geotags"].keys()) == ["mask", "rgb"]

    def test_to_pandas_one_row_per_sample_in_index_order(self, tmp_path):
        _write_stack(tmp_path / "a" / "0.zarr")
        _write_stack(tmp_path / "b" / "1.zarr")
        ds = StackDataset(tmp_path)
        df = ds.to_pandas()
        assert len(df) == 2
        assert list(df["path"]) == [ds.to_row(i)["path"] for i in range(len(ds))]

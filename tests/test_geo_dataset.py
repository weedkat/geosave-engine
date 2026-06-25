"""Unit tests for GeoDataset: file-scan discovery + geopandas spatial grouping.

No network — tiles are synthetic zarr stores written to a tmp folder, then
discovered via ``from_dir``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

from geosave_engine.geodata.core.geotile import GeoTile
from geosave_engine.geodata.datasets import GeoDataset, stack_samples

UTM = "EPSG:32633"


def _write_tile(layer_dir: Path, idx: int, bbox, names) -> None:
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    ds = xr.Dataset({n: xr_zeros(gb, dtype="uint16") for n in names}).rio.write_crs(UTM)
    GeoTile(geobox=gb, datetime=datetime(2023, 2, 1), data=ds).to_zarr(layer_dir / f"t{idx}.zarr")


def _dataset_root(root: Path, n_anchors: int = 2) -> Path:
    """Write ``n_anchors`` co-located image+label anchors under ``root``."""
    for i in range(n_anchors):
        x0 = 500000 + i * 320
        bbox = (x0, 5000000, x0 + 320, 5000320)
        _write_tile(root / "s2", i, bbox, ("B02", "B03"))
        _write_tile(root / "label", i, bbox, ("label",))
    return root


class TestGrouping:
    def test_co_located_groups(self, tmp_path):
        ds = GeoDataset.from_dir(_dataset_root(tmp_path, n_anchors=2))
        assert sorted(ds.layers) == ["label", "s2"]
        assert len(ds) == 2                       # 2 groups, not 4 (edge-touch pairs dropped)

    def test_getitem_renders_tensors(self, tmp_path):
        ds = GeoDataset.from_dir(_dataset_root(tmp_path))
        sample = ds[0]
        assert isinstance(sample["s2"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)
        assert tuple(sample["s2"].shape) == (2, 32, 32)   # 2 bands, H, W
        assert tuple(sample["label"].shape) == (1, 32, 32)

    def test_extra_meta_override(self, tmp_path):
        class DS(GeoDataset):
            def extra_meta(self, tiles):
                ref = next(iter(tiles.values()))
                return {"bbox": torch.tensor(ref.bbox)}

        ds = DS.from_dir(_dataset_root(tmp_path))
        sample = ds[0]
        assert "bbox" in sample
        assert isinstance(sample["bbox"], torch.Tensor)

    def test_single_layer(self, tmp_path):
        _write_tile(tmp_path / "s2", 0, (500000, 5000000, 500320, 5000320), ("B02",))
        ds = GeoDataset.from_dir(tmp_path)
        assert ds.layers == ["s2"]
        assert len(ds) == 1


class TestRender:
    def test_stack_samples_batches_per_output_key(self, tmp_path):
        class DS(GeoDataset):
            output_key = {"s2": "image", "label": "mask"}
            sel_bands = {"s2": ["B02"]}

        ds = DS.from_dir(_dataset_root(tmp_path))
        batch = stack_samples([ds[0], ds[1]])
        assert tuple(batch["image"].shape) == (2, 1, 32, 32)   # B, C(sel B02), H, W
        assert tuple(batch["mask"].shape) == (2, 1, 32, 32)
        assert len(batch["image"]) == 2

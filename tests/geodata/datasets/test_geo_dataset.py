"""Unit tests for GeoDataset: one GeoTIFF/Zarr file per sample, CRS preserved.

No network — tiles are synthetic, written via GeoTile.to_geotiff/to_zarr to a tmp folder.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.datasets import GeoDataset
from geosave_engine.geodata.tile import GeoAnchor, GeoTag, GeoTile

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)


def _tile(band_names: tuple[str, ...] = ("B02", "B03"), dt: datetime = datetime(2023, 2, 1)) -> GeoTile:
    gb = GeoBox.from_bbox(BBOX, crs=UTM, resolution=10, anchor="edge")
    arr = np.zeros((len(band_names), gb.height, gb.width), dtype="uint16")
    return GeoAnchor(geobox=gb, geotag=GeoTag(datetime=(dt, dt))).to_geotile(arr, list(band_names))


class TestZarr:
    def test_discovers_and_renders_tensor_with_anchor(self, tmp_path):
        _tile().to_zarr(tmp_path / "tile1.zarr")
        ds = GeoDataset(tmp_path, ".zarr")
        assert len(ds) == 1
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert tuple(sample["image"].shape) == (2, 32, 32)
        assert isinstance(sample["anchor"], GeoAnchor)
        assert not isinstance(sample["anchor"], GeoTile)

    def test_layer_name_customizes_dict_key(self, tmp_path):
        _tile().to_zarr(tmp_path / "tile1.zarr")
        ds = GeoDataset(tmp_path, ".zarr", layer_name="s2")
        assert set(ds[0]) == {"s2", "anchor"}

    def test_sel_bands_restricts_channels(self, tmp_path):
        _tile().to_zarr(tmp_path / "tile1.zarr")
        ds = GeoDataset(tmp_path, ".zarr", sel_bands=["B02"])
        assert tuple(ds[0]["image"].shape) == (1, 32, 32)

    def test_dtype_override_casts_tensor(self, tmp_path):
        _tile().to_zarr(tmp_path / "tile1.zarr")
        ds = GeoDataset(tmp_path, ".zarr", dtype_override=torch.float32)
        assert ds[0]["image"].dtype == torch.float32

    def test_discovers_nested_stores(self, tmp_path):
        _tile().to_zarr(tmp_path / "sub" / "a.zarr")
        _tile().to_zarr(tmp_path / "sub2" / "b.zarr")
        ds = GeoDataset(tmp_path, ".zarr")
        assert len(ds) == 2

    def test_split_filters_by_stem(self, tmp_path):
        _tile().to_zarr(tmp_path / "a.zarr")
        _tile().to_zarr(tmp_path / "b.zarr")
        split = tmp_path / "split.txt"
        split.write_text("a\n")
        ds = GeoDataset(tmp_path, ".zarr", split=split)
        assert ds.keys == ["a"]

    def test_to_row_returns_path_relative_to_root(self, tmp_path):
        _tile().to_zarr(tmp_path / "sub" / "a.zarr")
        ds = GeoDataset(tmp_path, ".zarr")
        assert ds.to_row("a") == {"path": str(Path("sub") / "a.zarr")}


class TestGeoTiff:
    def test_discovers_and_renders_via_filename_date_suffix(self, tmp_path):
        _tile().to_geotiff(tmp_path / "tile1-20230201.tif")
        ds = GeoDataset(tmp_path, ".tif")
        assert len(ds) == 1
        sample = ds[0]
        assert tuple(sample["image"].shape) == (2, 32, 32)
        assert isinstance(sample["anchor"], GeoAnchor)

    def test_key_pattern_strips_date_suffix(self, tmp_path):
        _tile().to_geotiff(tmp_path / "tile1-20230201.tif")
        ds = GeoDataset(tmp_path, ".tif", key_pattern=r"^(.+)-\d{8}\.tif$")
        assert ds.keys == ["tile1"]


def test_invalid_extension_raises(tmp_path):
    with pytest.raises(ValueError, match="GEO_RASTER_EXTENSIONS|extension must be one of"):
        GeoDataset(tmp_path, ".png")

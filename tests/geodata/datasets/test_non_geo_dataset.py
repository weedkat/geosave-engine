"""Unit tests for NonGeoDataset: plain-raster discovery, rendering, dtype/key options."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import xarray as xr

from geosave_engine.geodata.datasets.non_geo_dataset import NonGeoDataset


def _write_tif(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count, dtype=array.dtype
    ) as dst:
        dst.write(array)


def _write_zarr(path: Path, array: np.ndarray) -> None:
    """Write a plain zarr store, no CRS/geobox — array shape (C, H, W)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({f"band_{i}": (("y", "x"), band) for i, band in enumerate(array)})
    ds.to_zarr(path)


class TestNonGeoDataset:
    def test_discovers_and_renders_tensor(self, tmp_path):
        _write_tif(tmp_path / "img1.tif", np.zeros((3, 8, 8), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".tif")
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert tuple(sample["image"].shape) == (3, 8, 8)

    def test_layer_name_customizes_dict_key(self, tmp_path):
        _write_tif(tmp_path / "img1.tif", np.zeros((1, 4, 4), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".tif", layer_name="s2")
        assert set(ds[0]) == {"s2"}

    def test_dtype_override_casts_tensor(self, tmp_path):
        _write_tif(tmp_path / "img1.tif", np.zeros((1, 4, 4), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".tif", dtype_override=torch.float32)
        assert ds[0]["image"].dtype == torch.float32

    def test_invalid_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="RASTER_EXTENSIONS|extension must be one of"):
            NonGeoDataset(tmp_path, ".bmp")

    def test_key_pattern_extracts_custom_stem(self, tmp_path):
        _write_tif(tmp_path / "img_042.tif", np.zeros((1, 4, 4), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".tif", key_pattern=r"(\d+)\.tif$")
        assert ds.keys == ["042"]

    def test_split_filters_by_stem(self, tmp_path):
        _write_tif(tmp_path / "a.tif", np.zeros((1, 2, 2), dtype="uint8"))
        _write_tif(tmp_path / "b.tif", np.zeros((1, 2, 2), dtype="uint8"))
        split = tmp_path / "split.txt"
        split.write_text("a\n")
        ds = NonGeoDataset(tmp_path, ".tif", split=split)
        assert ds.keys == ["a"]

    def test_discovers_nested_files(self, tmp_path):
        _write_tif(tmp_path / "sub" / "a.tif", np.zeros((1, 2, 2), dtype="uint8"))
        _write_tif(tmp_path / "sub2" / "b.tif", np.zeros((1, 2, 2), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".tif")
        assert len(ds) == 2

    def test_to_row_returns_path_relative_to_root(self, tmp_path):
        _write_tif(tmp_path / "sub" / "a.tif", np.zeros((1, 2, 2), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".tif")
        assert ds.to_row("a") == {"path": str(Path("sub") / "a.tif")}


class TestZarrLiteral:
    def test_discovers_and_renders_tensor_no_crs(self, tmp_path):
        _write_zarr(tmp_path / "img1.zarr", np.zeros((3, 8, 8), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".zarr")
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert tuple(sample["image"].shape) == (3, 8, 8)

    def test_dtype_override_casts_tensor(self, tmp_path):
        _write_zarr(tmp_path / "img1.zarr", np.zeros((1, 4, 4), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".zarr", dtype_override=torch.float32)
        assert ds[0]["image"].dtype == torch.float32

    def test_discovers_nested_stores(self, tmp_path):
        _write_zarr(tmp_path / "sub" / "a.zarr", np.zeros((1, 2, 2), dtype="uint8"))
        _write_zarr(tmp_path / "sub2" / "b.zarr", np.zeros((1, 2, 2), dtype="uint8"))
        ds = NonGeoDataset(tmp_path, ".zarr")
        assert len(ds) == 2

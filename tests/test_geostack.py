"""Unit tests for GeoStack: folder-of-GeoTile save/load round trip, no spec.

No network — all tiles are synthetic, built from a projected geobox.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.tile import GeoAnchor, GeoTile, GeoStack

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _geobox(bbox=BBOX) -> GeoBox:
    return GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")


def _tile(names: tuple[str, ...], value: int = 1, dtype: str = "uint16", bbox=BBOX) -> GeoTile:
    gb = _geobox(bbox)
    arr = np.full((len(names), gb.height, gb.width), value, dtype=dtype)
    return GeoAnchor(geobox=gb, datetime=datetime(2024, 1, 15)).with_np(arr, list(names))


class TestConstruction:
    def test_kwargs_build(self):
        stack = GeoStack(
            sentinel_2_l1c=_tile(("B02", "B03", "B04")),
            cloud_mask=_tile(("cloud_mask",), dtype="uint8"),
        )
        assert set(stack.tiles) == {"sentinel_2_l1c", "cloud_mask"}
        assert stack.tiles["sentinel_2_l1c"].num_bands == 3
        assert stack.tiles["cloud_mask"].num_bands == 1

    def test_single_tile_skips_align(self):
        stack = GeoStack(a=_tile(("b1",)))
        assert stack.tiles["a"].width == 32

    def test_overlapping_tiles_auto_align_to_intersection(self):
        # second tile shifted 100m (10px) right — 22px overlap on x
        a = _tile(("b1",), bbox=BBOX)
        b = _tile(("b2",), bbox=(500100, 5000000, 500420, 5000320))
        stack = GeoStack(a=a, b=b)
        assert stack.tiles["a"].width == 22
        assert stack.tiles["b"].width == 22
        assert stack.tiles["a"].bbox == stack.tiles["b"].bbox

    def test_non_overlapping_tiles_raise(self):
        a = _tile(("b1",), bbox=BBOX)
        b = _tile(("b2",), bbox=(600000, 5000000, 600320, 5000320))
        with pytest.raises(ValueError, match="no spatial overlap"):
            GeoStack(a=a, b=b)


class TestSaveLoadRoundTrip:
    def test_multi_layer_round_trip(self, tmp_path):
        stack = GeoStack(
            sentinel_2_l1c=_tile(("B02", "B03", "B04"), value=7, dtype="uint16"),
            cloud_mask=_tile(("cloud_mask",), value=1, dtype="uint8"),
            ndvi=_tile(("ndvi",), value=42, dtype="float32"),
        )
        path = stack.save(tmp_path / "13.000000_52.000000_20240115_10m.geostack")

        loaded = GeoStack.load(path, load_data=True)

        assert set(loaded.tiles) == {"sentinel_2_l1c", "cloud_mask", "ndvi"}

        s2 = loaded.tiles["sentinel_2_l1c"]
        assert s2.num_bands == 3
        assert s2.bands == ("B02", "B03", "B04")
        assert s2.data.shape == (3, 32, 32)
        assert (s2.data.values == 7).all()

        mask = loaded.tiles["cloud_mask"]
        assert mask.num_bands == 1
        assert (mask.data.values == 1).all()

        ndvi = loaded.tiles["ndvi"]
        assert ndvi.data.dtype == np.float32

    def test_one_zarr_store_per_layer(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)), b=_tile(("b2",)), c=_tile(("b3",)))
        path = stack.save(tmp_path / "anchor.geostack")
        assert (path / "a.zarr").exists()
        assert (path / "b.zarr").exists()
        assert (path / "c.zarr").exists()

    def test_save_requires_geostack_suffix(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        with pytest.raises(ValueError, match="Expected a .geostack path"):
            stack.save(tmp_path / "anchor")

    def test_load_requires_geostack_suffix(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        path = stack.save(tmp_path / "anchor.geostack")
        renamed = path.rename(tmp_path / "anchor")
        with pytest.raises(ValueError, match="Expected a .geostack path"):
            GeoStack.load(renamed)

    def test_geobox_preserved(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)), b=_tile(("b2", "b3")))
        path = stack.save(tmp_path / "t.geostack")
        loaded = GeoStack.load(path)
        for tile in loaded.tiles.values():
            assert tile.crs == "EPSG:32633"
            assert tile.bbox == pytest.approx(BBOX)
            assert tile.resolution == 10

    def test_datetime_preserved(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        path = stack.save(tmp_path / "t.geostack")
        loaded = GeoStack.load(path)
        assert loaded.tiles["a"].start == datetime(2024, 1, 15)
        assert loaded.tiles["a"].end == datetime(2024, 1, 15)

    def test_metadata_preserved(self, tmp_path):
        tile = _tile(("b1",)).with_metadata({"nodata": 255, "description": "test layer"})
        stack = GeoStack(a=tile)
        path = stack.save(tmp_path / "t.geostack")
        loaded = GeoStack.load(path)
        assert loaded.tiles["a"].metadata == {"nodata": 255, "description": "test layer"}

    def test_required_layers_subset(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)), b=_tile(("b2",)))
        path = stack.save(tmp_path / "t.geostack")
        loaded = GeoStack.load(path, required_layers=["a"])
        assert set(loaded.tiles) == {"a"}

    def test_required_layers_missing_raises(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        path = stack.save(tmp_path / "t.geostack")
        with pytest.raises(KeyError, match="dynamicworld"):
            GeoStack.load(path, required_layers=["dynamicworld"])


class TestToTensor:
    def test_default_keys(self):
        stack = GeoStack(sentinel_2_l1c=_tile(("B02", "B03")), cloud_mask=_tile(("cloud_mask",)))
        sample = stack.to_tensor()
        assert isinstance(sample["sentinel_2_l1c"], torch.Tensor)
        assert tuple(sample["sentinel_2_l1c"].shape) == (2, 32, 32)
        assert tuple(sample["cloud_mask"].shape) == (1, 32, 32)

    def test_sel_bands(self):
        stack = GeoStack(sentinel_2_l1c=_tile(("B02", "B03", "B04")))
        sample = stack.to_tensor(sel_bands={"sentinel_2_l1c": ["B03"]})
        assert tuple(sample["sentinel_2_l1c"].shape) == (1, 32, 32)

    def test_dtype_override(self):
        stack = GeoStack(cloud_mask=_tile(("cloud_mask",), dtype="uint8"))
        sample = stack.to_tensor(dtype_override={"cloud_mask": torch.bool})
        assert sample["cloud_mask"].dtype == torch.bool

    def test_context_fn(self):
        stack = GeoStack(a=_tile(("b1",)))

        def context(tiles):
            return {"stem": next(iter(tiles.values())).stem}

        sample = stack.to_tensor(context_fn=context)
        assert "context" in sample
        assert "stem" in sample["context"]

    def test_no_context_key_when_context_fn_none(self):
        stack = GeoStack(a=_tile(("b1",)))
        sample = stack.to_tensor()
        assert "context" not in sample

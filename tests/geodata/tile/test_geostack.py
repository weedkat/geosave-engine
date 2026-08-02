"""Unit tests for GeoStack: folder-of-GeoTile save/load round trip, no spec.

No network — all tiles are synthetic, built from a projected geobox.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.tile import GeoAnchor, GeoTag, GeoTile, GeoStack

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _geobox(bbox=BBOX) -> GeoBox:
    return GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")


def _tile(names: tuple[str, ...], value: int = 1, dtype: str = "uint16", bbox=BBOX) -> GeoTile:
    gb = _geobox(bbox)
    arr = np.full((len(names), gb.height, gb.width), value, dtype=dtype)
    d = datetime(2024, 1, 15)
    return GeoAnchor(geobox=gb, geotag=GeoTag(datetime=(d, d))).to_geotile(arr, list(names))


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
        path = stack.save(tmp_path / "13.000000_52.000000_20240115_10m.zarr")

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

    def test_one_zarr_store_with_one_group_per_layer(self, tmp_path):
        import zarr

        stack = GeoStack(a=_tile(("b1",)), b=_tile(("b2",)), c=_tile(("b3",)))
        path = stack.save(tmp_path / "anchor.zarr")
        assert path == tmp_path / "anchor.zarr"
        assert path.is_dir()  # one store, not a folder of per-layer stores
        assert not (path / "a.zarr").exists()  # not the old one-store-per-layer shape
        assert sorted(zarr.open_group(path, mode="r").group_keys()) == ["a", "b", "c"]

    def test_save_overwrites_removed_layer_not_left_stale(self, tmp_path):
        import zarr

        path = tmp_path / "anchor.zarr"
        GeoStack(a=_tile(("b1",)), b=_tile(("b2",))).save(path)
        GeoStack(a=_tile(("b1",))).save(path)
        assert sorted(zarr.open_group(path, mode="r").group_keys()) == ["a"]

    def test_save_requires_zarr_suffix(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        with pytest.raises(ValueError, match="Expected a .zarr path"):
            stack.save(tmp_path / "anchor")

    def test_load_requires_zarr_suffix(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        path = stack.save(tmp_path / "anchor.zarr")
        renamed = path.rename(tmp_path / "anchor")
        with pytest.raises(ValueError, match="Expected a .zarr path"):
            GeoStack.load(renamed)

    def test_load_rejects_store_with_no_groups(self, tmp_path):
        # a plain root-level store (e.g. written by GeoTile.to_zarr, not GeoStack.save)
        import xarray as xr

        plain = xr.Dataset({"x": (("y", "x"), np.zeros((2, 2)))})
        path = tmp_path / "plain.zarr"
        plain.to_zarr(path, mode="w")
        with pytest.raises(ValueError, match="no zarr groups"):
            GeoStack.load(path)

    def test_geobox_preserved(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)), b=_tile(("b2", "b3")))
        path = stack.save(tmp_path / "t.zarr")
        loaded = GeoStack.load(path)
        for tile in loaded.tiles.values():
            assert tile.crs == "EPSG:32633"
            assert tile.bbox == pytest.approx(BBOX)
            assert tile.resolution == 10

    def test_datetime_preserved(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        path = stack.save(tmp_path / "t.zarr")
        loaded = GeoStack.load(path)
        assert loaded.tiles["a"].start == datetime(2024, 1, 15)
        assert loaded.tiles["a"].end == datetime(2024, 1, 15)

    def test_metadata_preserved(self, tmp_path):
        tile = _tile(("b1",)).rebase(metadata={"nodata": 255, "description": "test layer"})
        stack = GeoStack(a=tile)
        path = stack.save(tmp_path / "t.zarr")
        loaded = GeoStack.load(path)
        assert loaded.tiles["a"].metadata == {"nodata": 255, "description": "test layer"}

    def test_required_layers_subset(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)), b=_tile(("b2",)))
        path = stack.save(tmp_path / "t.zarr")
        loaded = GeoStack.load(path, required_layers=["a"])
        assert set(loaded.tiles) == {"a"}

    def test_required_layers_missing_raises(self, tmp_path):
        stack = GeoStack(a=_tile(("b1",)))
        path = stack.save(tmp_path / "t.zarr")
        with pytest.raises(KeyError, match="dynamicworld"):
            GeoStack.load(path, required_layers=["dynamicworld"])

    def test_different_band_counts_and_dtypes_per_layer(self, tmp_path):
        # different band count, different dtype per layer — no forced homogeneity
        stack = GeoStack(
            s1=_tile(("VV", "VH"), dtype="float32"),
            s2=_tile(("B02", "B03", "B04", "B08"), dtype="uint16"),
            label=_tile(("label",), dtype="uint8"),
        )
        path = stack.save(tmp_path / "t.zarr")
        loaded = GeoStack.load(path, load_data=True)
        assert loaded.tiles["s1"].bands == ("VV", "VH")
        assert loaded.tiles["s1"].data.dtype == np.float32
        assert loaded.tiles["s2"].bands == ("B02", "B03", "B04", "B08")
        assert loaded.tiles["s2"].data.dtype == np.uint16
        assert loaded.tiles["label"].bands == ("label",)
        assert loaded.tiles["label"].data.dtype == np.uint8

    def test_mixed_time_and_no_time_layers_survive_independently(self, tmp_path):
        gb = _geobox()
        d = datetime(2024, 1, 15)
        multi_step = GeoAnchor(geobox=gb, geotag=GeoTag(datetime=(d, d))).to_geotile(
            np.stack(
                [
                    np.full((1, gb.height, gb.width), 1, dtype="uint16"),
                    np.full((1, gb.height, gb.width), 2, dtype="uint16"),
                    np.full((1, gb.height, gb.width), 3, dtype="uint16"),
                ]
            ),
            ["s2"],
            times=[datetime(2024, 1, 1), datetime(2024, 1, 15), datetime(2024, 2, 1)],
        )
        single_step = _tile(("label",), value=9, dtype="uint8")
        stack = GeoStack(s2=multi_step, label=single_step)

        path = stack.save(tmp_path / "t.zarr")
        loaded = GeoStack.load(path, load_data=True)

        s2 = loaded.tiles["s2"]
        assert s2.has_time
        assert s2.times == (datetime(2024, 1, 1), datetime(2024, 1, 15), datetime(2024, 2, 1))
        assert s2.data.shape == (3, 1, 32, 32)
        assert (s2.data.isel(time=1).values == 2).all()
        assert s2.start == datetime(2024, 1, 1)
        assert s2.end == datetime(2024, 2, 1)

        label = loaded.tiles["label"]
        assert not label.has_time
        assert label.start == datetime(2024, 1, 15)
        assert label.end == datetime(2024, 1, 15)
        assert (label.data.values == 9).all()

    def test_polygon_and_plot_meta_preserved(self, tmp_path):
        from odc.geo.geom import polygon

        poly = polygon([(500000, 5000000), (500320, 5000000), (500320, 5000320), (500000, 5000000)], crs=UTM)
        tile = _tile(("label",), dtype="uint8").rebase(
            plot_meta={"class_map": {0: "water", 1: "trees"}, "color_map": {0: "#0000ff", 1: "#00ff00"}}
        )
        tile = tile.rebase(polygon=poly)
        stack = GeoStack(a=tile)
        path = stack.save(tmp_path / "t.zarr")
        loaded = GeoStack.load(path)

        assert loaded.tiles["a"].polygon is not None
        assert loaded.tiles["a"].polygon.crs == poly.crs
        assert loaded.tiles["a"].plot_meta.class_map == {0: "water", 1: "trees"}
        assert loaded.tiles["a"].plot_meta.color_map == {0: "#0000ff", 1: "#00ff00"}


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

    def test_anchor_always_present(self):
        stack = GeoStack(a=_tile(("b1",)))
        sample = stack.to_tensor()
        assert "anchors" in sample
        assert isinstance(sample["anchors"], dict)
        assert isinstance(sample["anchors"]["a"], GeoAnchor)
        assert not isinstance(sample["anchors"]["a"], GeoTile)

    def test_anchor_matches_reference_tile_identity(self):
        tile = _tile(("b1",))
        stack = GeoStack(a=tile)
        sample = stack.to_tensor()
        assert sample["anchors"]["a"].stem == tile.stem

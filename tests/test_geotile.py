"""Unit tests for GeoTile: round-trips, spatial ops, and tensor rendering.

No network — all tiles are synthetic, built from a projected geobox.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros, xr_coords

from geosave_engine.geodata.core import GeoTile, align, mosaic, remap

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _item(item_id: str) -> pystac.Item:
    """Minimal valid pystac Item for provenance round-trip tests."""
    return pystac.Item(
        id=item_id,
        geometry={"type": "Point", "coordinates": [0.0, 0.0]},
        bbox=[0.0, 0.0, 0.0, 0.0],
        datetime=datetime(2023, 2, 1, tzinfo=timezone.utc),
        properties={},
    )


def _tile(
    bbox=BBOX,
    *,
    names=("red", "green"),
    times=None,
    dt=datetime(2023, 2, 1),
    meta=None,
    stac=(),
) -> GeoTile:
    """Synthetic tile: one band per name, optional time axis."""
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    base_coords = dict(xr_coords(gb))
    n = len(names)
    if times is None:
        arr = np.stack([np.full((gb.height, gb.width), i, dtype="uint16") for i in range(n)])
        da = xr.DataArray(arr, dims=("band", "y", "x"), coords={**base_coords, "band": list(names)})
    else:
        time_coord = [np.datetime64(t) for t in times]
        nt = len(time_coord)
        arr = np.stack([
            np.stack([np.full((gb.height, gb.width), i, dtype="uint16") for _ in range(nt)])
            for i in range(n)
        ], axis=1)  # (time, band, y, x)
        da = xr.DataArray(arr, dims=("time", "band", "y", "x"),
                          coords={**base_coords, "band": list(names), "time": time_coord})
    return GeoTile(
        geobox=gb,
        datetime=dt,
        data=da,
        metadata=meta or {"foo": "bar"},
        stac=[_item(s) for s in stac],
    )


class TestHeader:
    def test_from_bbox_is_header(self):
        t = GeoTile.from_bbox(BBOX, crs=UTM, resolution=10, datetime="2023-02-01")
        assert t.data is None
        assert t.bands == ()
        assert t.num_bands == 0
        assert not t.has_time

    def test_with_data_rejects_non_dataarray(self):
        t = GeoTile.from_bbox(BBOX, crs=UTM, resolution=10, datetime="2023-02-01")
        gb = GeoBox.from_bbox(BBOX, crs=UTM, resolution=10, anchor="edge")
        with pytest.raises(TypeError):
            t.with_data(xr.Dataset({"a": xr_zeros(gb)}))  # Dataset, not DataArray


class TestGeotiffRoundtrip:
    @pytest.mark.parametrize("writer", ["to_geotiff", "to_cog"])
    def test_roundtrip_preserves_names_meta_geobox(self, tmp_path, writer):
        t = _tile(meta={"foo": "bar"})
        path = getattr(t, writer)(tmp_path / "scene" / "red_20230201.tif")
        r = GeoTile.from_geotiff(path)
        assert r.bands == ("red", "green")
        assert r.metadata == {"foo": "bar"}
        assert r.geobox == t.geobox

    def test_band_selection(self, tmp_path):
        path = _tile().to_geotiff(tmp_path / "red_20230201.tif")
        r = GeoTile.from_geotiff(path, bands=("green",))
        assert r.bands == ("green",)

    def test_datetime_attr_wins_over_filename(self, tmp_path):
        t = _tile(dt=datetime(2024, 5, 6))
        path = t.to_geotiff(tmp_path / "scene_20230201.tif")
        renamed = path.with_name("renamed.tif")
        path.rename(renamed)

        r = GeoTile.from_geotiff(renamed)
        assert r.datetime == t.datetime

    def test_datetime_falls_back_to_stem_when_missing(self, monkeypatch, tmp_path):
        t = _tile()
        ds = t.data.to_dataset(dim="band")
        ds.attrs.pop("datetime", None)
        ds.attrs.pop("metadata", None)

        monkeypatch.setattr("geosave_engine.geodata.core.geotile.rioxarray.open_rasterio", lambda *args, **kwargs: ds)

        r = GeoTile.from_geotiff(tmp_path / "scene_20230201.tif")
        assert r.datetime == datetime(2023, 2, 1)

    def test_datetime_missing_everywhere_raises(self, monkeypatch, tmp_path):
        t = _tile()
        ds = t.data.to_dataset(dim="band")
        ds.attrs.pop("datetime", None)
        ds.attrs.pop("metadata", None)

        monkeypatch.setattr("geosave_engine.geodata.core.geotile.rioxarray.open_rasterio", lambda *args, **kwargs: ds)

        with pytest.raises(ValueError, match="Could not determine datetime"):
            GeoTile.from_geotiff(tmp_path / "scene.tif")

    def test_time_series_to_geotiff_raises(self, tmp_path):
        t = _tile(names=("red",), times=["2023-01-01", "2023-02-01"])
        with pytest.raises(ValueError):
            t.to_geotiff(tmp_path / "x_20230201.tif")


class TestZarrRoundtrip:
    def test_timeseries_roundtrip(self, tmp_path):
        t = _tile(names=("red",), times=["2023-01-01", "2023-02-01"], meta={"a": 1})
        store = t.to_zarr(tmp_path / "cube.zarr")
        r = GeoTile.from_zarr(store)
        assert r.bands == ("red",)
        assert r.has_time and len(r.times) == 2
        assert r.metadata == {"a": 1}
        assert r.datetime == t.datetime
        assert r.geobox == t.geobox

    def test_datetime_falls_back_to_stem_when_missing(self, monkeypatch, tmp_path):
        t = _tile(names=("red",))
        ds = t.data.to_dataset(dim="band")
        ds.attrs.pop("datetime", None)
        ds.attrs.pop("metadata", None)

        monkeypatch.setattr("geosave_engine.geodata.core.geotile.xr.open_zarr", lambda *args, **kwargs: ds)

        r = GeoTile.from_zarr(tmp_path / "cube_20230201.zarr")
        assert r.datetime == datetime(2023, 2, 1)


class TestMetadata:
    def test_append_conflict_raises(self):
        t = _tile(names=("red",), meta={"name": "s2"})
        with pytest.raises(ValueError):
            t.with_metadata({"name": "other"})        # append default → clash raises

    def test_replace_overwrites(self):
        t = _tile(names=("red",), meta={"name": "s2"})
        assert t.with_metadata({"name": "other"}, replace=True).metadata["name"] == "other"

    def test_append_disjoint_merges(self):
        t = _tile(names=("red",), meta={"a": 1}).with_metadata({"b": 2})
        assert t.metadata == {"a": 1, "b": 2}


class TestStac:
    def test_sidecar_roundtrip_zarr(self, tmp_path):
        t = _tile(names=("red",), stac=("scene_a", "scene_b"))
        store = t.to_zarr(tmp_path / "cube.zarr", save_stac=True)
        assert (tmp_path / "cube.stac.json").exists()
        r = GeoTile.from_zarr(store)
        assert [i.id for i in r.stac] == ["scene_a", "scene_b"]

    def test_sidecar_roundtrip_cog(self, tmp_path):
        t = _tile(names=("red",), stac=("scene_a",))
        p = t.to_cog(tmp_path / "x_20230201.tif", save_stac=True)
        assert (tmp_path / "x_20230201.stac.json").exists()
        assert [i.id for i in GeoTile.from_geotiff(p).stac] == ["scene_a"]

    def test_no_sidecar_when_opted_out(self, tmp_path):
        t = _tile(names=("red",), stac=("scene_a",))
        t.to_zarr(tmp_path / "cube.zarr")            # save_stac=False (default)
        assert not (tmp_path / "cube.stac.json").exists()
        assert GeoTile.from_zarr(tmp_path / "cube.zarr").stac == []

    def test_with_stac_dedups_by_id(self):
        t = _tile(names=("red",)).with_stac([_item("a"), _item("b")]).with_stac([_item("a")])
        assert [i.id for i in t.stac] == ["a", "b"]


class TestSpatial:
    def test_mosaic_preserves_time_and_extent(self):
        left = _tile((500000, 5000000, 500320, 5000320), names=("red",),
                     times=["2023-01-01", "2023-02-01"])
        right = _tile((500320, 5000000, 500640, 5000320), names=("red",),
                      times=["2023-01-01", "2023-02-01"])
        m = mosaic([left, right])
        assert m.has_time and len(m.times) == 2
        assert (m.width, m.height) == (64, 32)

    def test_mosaic_different_times_raises(self):
        a = _tile(names=("red",), times=["2023-01-01", "2023-02-01"])
        b = _tile((500320, 5000000, 500640, 5000320), names=("red",),
                  times=["2023-03-01", "2023-04-01"])
        with pytest.raises(ValueError):
            mosaic([a, b])

    def test_align_narrows_geobox_lazily(self):
        a = _tile((500000, 5000000, 500320, 5000320), names=("red",))
        b = _tile((500160, 5000160, 500480, 5000480), names=("nir",))
        a2, b2 = align(a, b)
        assert a2.geobox == b2.geobox             # common intersection
        assert (a2.width, a2.height) == (16, 16)
        assert a2.data is a.data                  # data untouched — read happens on to_tensor


class TestTensor:
    def test_to_numpy_shape_with_time(self):
        t = _tile(names=("red", "green"), times=["2023-01-01", "2023-02-01"])
        out = t.to_numpy()                    # (time, band, y, x)
        assert isinstance(out, np.ndarray)
        assert tuple(out.shape) == (2, 2, 32, 32)

    def test_to_tensor_shape_with_time(self):
        t = _tile(names=("red", "green"), times=["2023-01-01", "2023-02-01"])
        out = t.to_tensor()                       # (time, band, y, x)
        assert tuple(out.shape) == (2, 2, 32, 32)

    def test_to_numpy_matches_to_tensor(self):
        t = _tile(names=("red", "green", "blue"))
        assert np.array_equal(t.to_numpy(["blue", "red"]), t.to_tensor(["blue", "red"]).numpy())

    def test_to_tensor_shape_no_time(self):
        out = _tile(names=("red", "green")).to_tensor()   # (band, y, x)
        assert tuple(out.shape) == (2, 32, 32)

    def test_to_tensor_band_select_and_order(self):
        out = _tile(names=("red", "green", "blue")).to_tensor(["blue", "red"])
        assert tuple(out.shape) == (2, 32, 32)    # selected, in given order

    def test_patch_reads_only_window(self):
        t = _tile(names=("red",))
        patch = t.with_geobox(t.geobox[0:16, 0:16])
        out = patch.to_tensor()                   # (band, y, x)
        assert tuple(out.shape) == (1, 16, 16)


class TestRemap:
    def test_remap_values(self):
        t = _tile(names=("label",))  # all zeros
        out = remap(t, {0: 5}).to_tensor()        # (band, y, x)
        assert (out == 5).all()


class TestRealData:
    def test_from_geotiff_real_dw_tif(self, dw_tif_path):
        t = GeoTile.from_geotiff(dw_tif_path, load_data=True)
        assert t.data is not None
        assert t.num_bands >= 1
        out = t.to_tensor()                       # (band, y, x)
        assert out.shape[0] == t.num_bands

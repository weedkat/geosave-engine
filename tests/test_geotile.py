"""Unit tests for GeoTile: round-trips, spatial ops, and tensor rendering.

No network — all tiles are synthetic, built from a projected geobox.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pystac
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros, xr_coords

from geosave_engine.geodata.tile import GeoAnchor, GeoTile, align, mosaic, remap

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
    def test_from_bbox_returns_anchor_not_tile(self):
        a = GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=10, datetime="2023-02-01")
        assert isinstance(a, GeoAnchor)
        assert not isinstance(a, GeoTile)

    def test_with_data_rejects_non_dataarray(self):
        a = GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=10, datetime="2023-02-01")
        gb = GeoBox.from_bbox(BBOX, crs=UTM, resolution=10, anchor="edge")
        with pytest.raises(TypeError):
            a.with_data(xr.Dataset({"a": xr_zeros(gb)}))  # Dataset, not DataArray

    def test_with_data_returns_tile(self):
        a = GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=10, datetime="2023-02-01")
        gb = GeoBox.from_bbox(BBOX, crs=UTM, resolution=10, anchor="edge")
        t = a.with_data(xr_zeros(gb).expand_dims(band=["b1"]))
        assert isinstance(t, GeoTile)
        assert t.bands == ("b1",)

    def test_date_string_expands_and_stem_contains_range(self):
        a = GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=10, datetime="2023-02-01")

        assert a.datetime == (
            datetime(2023, 2, 1),
            datetime(2023, 2, 1, 23, 59, 59, 999999),
        )
        assert a.stem.endswith("_20230201T000000_20230201T235959.999999_10m")


class TestFromGeojson:
    """from_geojson yields lazily, one anchor per feature — not a pre-built list."""

    def _write(self, tmp_path, n_features: int):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [13.0 + i, 52.0]}, "properties": {}}
                for i in range(n_features)
            ],
        }
        path = tmp_path / "aoi.geojson"
        path.write_text(json.dumps(geojson))
        return path

    def test_returns_iterator_not_list(self, tmp_path):
        path = self._write(tmp_path, 2)
        result = GeoAnchor.from_geojson(path, datetime="2023-02-01")
        assert not isinstance(result, list)
        assert iter(result) is result

    def test_yields_one_anchor_per_feature(self, tmp_path):
        path = self._write(tmp_path, 3)
        anchors = list(GeoAnchor.from_geojson(path, datetime="2023-02-01"))
        assert len(anchors) == 3
        assert all(isinstance(a, GeoAnchor) for a in anchors)


class TestGeotiffRoundtrip:
    @pytest.mark.parametrize("writer", ["to_geotiff", "to_cog"])
    def test_roundtrip_preserves_names_meta_geobox(self, tmp_path, writer):
        t = _tile(meta={"foo": "bar"})
        path = getattr(t, writer)(tmp_path / "scene" / "red_20230201.tif")
        r = GeoTile.from_geotiff(path, datetime=t.datetime)
        assert r.bands == ("red", "green")
        assert r.metadata == {"foo": "bar"}
        assert r.geobox == t.geobox

    def test_band_selection(self, tmp_path):
        path = _tile().to_geotiff(tmp_path / "red_20230201.tif")
        r = GeoTile.from_geotiff(path, datetime=datetime(2023, 2, 1), bands=("green",))
        assert r.bands == ("green",)

    def test_datetime_is_caller_supplied_not_read_from_file(self, tmp_path):
        """from_geotiff never reads a date from tags or the filename — caller decides."""
        t = _tile(dt=datetime(2024, 5, 6))
        path = t.to_geotiff(tmp_path / "scene_20230201.tif")  # embeds 2024-05-06, named 2023-02-01

        r = GeoTile.from_geotiff(path, datetime=datetime(2019, 1, 1))
        assert r.datetime == (datetime(2019, 1, 1), datetime(2019, 1, 1))

    def test_time_series_to_geotiff_raises(self, tmp_path):
        t = _tile(names=("red",), times=["2023-01-01", "2023-02-01"])
        with pytest.raises(ValueError):
            t.to_geotiff(tmp_path / "x_20230201.tif")

    def test_nodata_preserved(self, tmp_path):
        t = _tile(names=("label",)).with_nodata(255)
        path = t.to_geotiff(tmp_path / "x_20230201.tif")
        r = GeoTile.from_geotiff(path, datetime=t.datetime)
        assert r.nodata == 255

    def test_no_nodata_stays_none(self, tmp_path):
        t = _tile(names=("label",))
        path = t.to_geotiff(tmp_path / "x_20230201.tif")
        r = GeoTile.from_geotiff(path, datetime=t.datetime)
        assert r.nodata is None


class TestZarrRoundtrip:
    def test_timeseries_roundtrip(self, tmp_path):
        t = _tile(names=("red",), times=["2023-01-01", "2023-02-01"], meta={"a": 1})
        store = t.to_zarr(tmp_path / "cube.zarr")
        r = GeoTile.from_zarr(store)
        assert r.bands == ("red",)
        assert r.has_time and len(r.times) == 2
        assert r.metadata == {"a": 1}
        # datetime derives from the time coordinate on reload, not the
        # original construction anchor -- start/end become the actual
        # observed min/max, independent of t.datetime's own anchor value.
        assert r.start == datetime(2023, 1, 1) and r.end == datetime(2023, 2, 1)
        assert r.geobox == t.geobox

    def test_nodata_preserved(self, tmp_path):
        t = _tile(names=("label",)).with_nodata(255)
        store = t.to_zarr(tmp_path / "cube.zarr")
        r = GeoTile.from_zarr(store)
        assert r.nodata == 255

    def test_missing_datetime_attr_raises(self, monkeypatch, tmp_path):
        t = _tile(names=("red",))
        ds = t.data.to_dataset(dim="band")
        ds.attrs.pop("datetime", None)
        ds.attrs.pop("metadata", None)

        monkeypatch.setattr("geosave_engine.geodata.tile.geotile.xr.open_zarr", lambda *args, **kwargs: ds)

        with pytest.raises(ValueError, match="has no time dimension and no 'datetime' attr"):
            GeoTile.from_zarr(tmp_path / "cube.zarr")


class TestNodata:
    def test_default_is_none(self):
        assert _tile(names=("red",)).nodata is None

    def test_with_nodata_sets_value(self):
        t = _tile(names=("red",)).with_nodata(255)
        assert t.nodata == 255

    def test_with_nodata_does_not_mutate_original(self):
        original = _tile(names=("red",))
        original.with_nodata(255)
        assert original.nodata is None

    def test_with_nodata_none_clears(self):
        t = _tile(names=("red",)).with_nodata(255).with_nodata(None)
        assert t.nodata is None


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
        r = GeoTile.from_geotiff(p, datetime=datetime(2023, 2, 1))
        assert [i.id for i in r.stac] == ["scene_a"]

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
        t = GeoTile.from_geotiff(dw_tif_path, datetime=datetime(2019, 2, 23), load_data=True)
        assert t.num_bands >= 1
        out = t.to_tensor()                       # (band, y, x)
        assert out.shape[0] == t.num_bands

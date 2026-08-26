"""Round trips through every writer, and the laziness the whole design rests on.

Reads run under `-W error` because a warning during open is a real defect
here: it means something we wrote into a store isn't something the format
recognises. Laziness is asserted explicitly — losing it is silent, and only
shows up as an out-of-memory much later.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from geosave_engine.geodata.spatial import GeoRaster, GeoStack

from .conftest import ORIGIN_X, is_lazy, make_lazy_raster, make_raster, make_vector


@pytest.fixture(autouse=True)
def warnings_are_errors():
    """A warning from a reader means we wrote something the format doesn't accept."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path


class TestGeoTiffRoundTrip:
    def test_pixels_bands_and_nodata(self, store):
        raster = make_raster()
        back = GeoRaster.open(raster.to_geotiff(store / "x.tif", progress=False))
        assert back.bands == ("B04", "B08")
        assert back.nodata == 0
        np.testing.assert_array_equal(back.data.values, raster.data.values)

    def test_header_survives(self, store):
        raster = make_raster().rebase(
            tags={"source": "survey"}, render={"rgb_bands": ("B04", "B08", "B04")}
        )
        back = GeoRaster.open(raster.to_geotiff(store / "x.tif", progress=False))
        assert back.tags == {"source": "survey"}
        assert back.render is not None and back.render.rgb_bands == ("B04", "B08", "B04")

    def test_cf_only_attrs_do_not_leak_in(self, store):
        """A GeoTIFF has no CF legend, so CF adapter attrs must not ride along."""
        import rasterio

        source = GeoRaster.open(make_raster().to_zarr(store / "src.zarr", progress=False))
        with rasterio.open(source.to_geotiff(store / "x.tif", progress=False)) as handle:
            tags = handle.tags()
        assert not {"Conventions", "_geosave_band_order", "flag_values"} & set(tags)

    def test_a_time_dimension_needs_one_step_named(self, store):
        raster = make_raster(
            times=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
            dtype="float32",
            time=("2024-01-01", "2024-02-01"),
        )
        with pytest.raises(ValueError, match="holds no time axis"):
            raster.to_geotiff(store / "x.tif", progress=False)

    def test_naming_a_step_on_a_timeless_raster_is_rejected(self, store):
        with pytest.raises(ValueError, match="has no time dim"):
            make_raster().to_geotiff(store / "x.tif", time="2024-01-15", progress=False)


class TestZarrRoundTrip:
    def test_pixels_bands_and_header(self, store):
        raster = make_raster().rebase(tags={"source": "survey"})
        back = GeoRaster.open(raster.to_zarr(store / "x.zarr", progress=False))
        assert back.bands == ("B04", "B08")
        assert back.tags == {"source": "survey"}
        np.testing.assert_array_equal(back.data.values, raster.data.values)

    def test_reopens_lazily(self, store):
        back = GeoRaster.open(make_raster().to_zarr(store / "x.zarr", progress=False))
        assert is_lazy(back)

    def test_time_axis_survives(self, store):
        raster = make_raster(
            times=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
            dtype="float32",
            time=("2024-01-01", "2024-02-01"),
        )
        back = GeoRaster.open(raster.to_zarr(store / "x.zarr", progress=False))
        assert back.data.dims == ("time", "band", "y", "x")
        np.testing.assert_array_equal(back.data.time.values, raster.data.time.values)

    def test_band_narrowing_prunes_a_now_broken_render_reference(self, store):
        raster = make_raster().rebase(render={"rgb_bands": ("B04", "B08", "B04")})
        path = raster.to_zarr(store / "x.zarr", progress=False)
        back = GeoRaster.open(path, bands=("B04",))
        assert back.bands == ("B04",)
        assert back.render is not None and back.render.rgb_bands is None

    def test_band_narrowing_keeps_a_reference_it_still_covers(self, store):
        raster = make_raster(bands=("B02", "B04", "B08")).rebase(render={"rgb_bands": ("B04", "B02", "B02")})
        path = raster.to_zarr(store / "x.zarr", progress=False)
        back = GeoRaster.open(path, bands=("B04", "B02"))
        assert back.render is not None and back.render.rgb_bands == ("B04", "B02", "B02")

    def test_clearing_render_clears_the_cf_flag_mirror_on_disk(self, store):
        """The mirror is rebuilt per write, so a stale copy must not survive."""
        import zarr

        raster = make_raster().rebase(render={"class_map": {0: "bg", 1: "palm"}})
        reopened = GeoRaster.open(raster.to_zarr(store / "a.zarr", progress=False))
        path = reopened.rebase(render=None).to_zarr(store / "b.zarr", progress=False)
        attrs = dict(zarr.open_group(str(path), mode="r").attrs)
        assert not [key for key in attrs if key.startswith("flag_")]


class TestNetcdfRoundTrip:
    def test_pixels_and_bands(self, store):
        raster = make_raster()
        back = GeoRaster.open(raster.to_netcdf(store / "x.nc", progress=False))
        assert back.bands == ("B04", "B08")
        np.testing.assert_array_equal(back.data.values, raster.data.values)

    def test_groups_stay_independent(self, store):
        path = store / "x.nc"
        make_raster(bands=("B04",)).to_netcdf(path, group="image", progress=False)
        make_raster(bands=("dem",), dtype="int16", nodata=-9999).to_netcdf(
            path, group="dem", progress=False
        )
        assert GeoRaster.open(path, group="image").bands == ("B04",)
        assert GeoRaster.open(path, group="dem").nodata == -9999


class TestVectorSidecar:
    def test_written_beside_the_store_never_inside(self, store):
        """Zarr's member scan rejects any foreign object inside the hierarchy."""
        path = make_raster(vector=make_vector()).to_zarr(store / "x.zarr", progress=False)
        assert not list(path.glob("*.parquet"))
        assert (store / "x.vector.parquet").exists()

    def test_round_trips(self, store):
        raster = make_raster(vector=make_vector())
        back = GeoRaster.open(raster.to_zarr(store / "x.zarr", progress=False))
        assert back.vector is not None and len(back.vector) == 1

    def test_grouped_writes_keep_separate_files(self, store):
        path = store / "x.zarr"
        make_raster(bands=("B04",), vector=make_vector()).to_zarr(path, group="image", progress=False)
        make_raster(bands=("dem",)).to_zarr(path, group="dem", progress=False)
        assert GeoRaster.open(path, group="image").vector is not None
        assert GeoRaster.open(path, group="dem").vector is None

    def test_clearing_the_vector_removes_a_stale_sidecar(self, store):
        raster = make_raster(vector=make_vector())
        raster.to_zarr(store / "x.zarr", progress=False)
        raster.rebase(vector=None).to_zarr(store / "x.zarr", progress=False)
        assert not (store / "x.vector.parquet").exists()

    def test_a_named_vector_path_must_exist(self, store):
        path = make_raster().to_zarr(store / "x.zarr", progress=False)
        with pytest.raises(FileNotFoundError):
            GeoRaster.open(path, vector_path=store / "nope.parquet")


class TestStackStore:
    def test_one_group_per_layer(self, store):
        stack = GeoStack(
            image=make_raster(bands=("B04", "B08"), vector=make_vector()),
            label=make_raster(bands=("cls",), dtype="uint8", nodata=255),
        )
        back = GeoStack.open(stack.to_zarr(store / "s.zarr", progress=False), reference_layer="image")
        assert sorted(back) == ["image", "label"]

    def test_per_layer_dtype_and_nodata_survive(self, store):
        stack = GeoStack(
            image=make_raster(bands=("B04",), dtype="uint16", nodata=0),
            label=make_raster(bands=("cls",), dtype="uint8", nodata=255),
        )
        back = GeoStack.open(stack.to_zarr(store / "s.zarr", progress=False))
        assert back["image"].dtype == np.dtype("uint16") and back["image"].nodata == 0
        assert back["label"].dtype == np.dtype("uint8") and back["label"].nodata == 255

    def test_reopens_lazily(self, store):
        stack = GeoStack(image=make_raster(bands=("B04",)))
        back = GeoStack.open(stack.to_zarr(store / "s.zarr", progress=False))
        assert all(is_lazy(back[name]) for name in back)

    def test_the_reference_layers_vector_comes_back(self, store):
        stack = GeoStack(image=make_raster(bands=("B04",), vector=make_vector()))
        back = GeoStack.open(stack.to_zarr(store / "s.zarr", progress=False))
        assert back.vector is not None and len(back.vector) == 1

    def test_a_non_zarr_path_is_rejected(self, store):
        with pytest.raises(ValueError, match="Expected a .zarr store"):
            GeoStack.open(store / "x.tif")

    def test_a_store_with_no_layer_group_is_rejected(self, store):
        path = make_raster().to_zarr(store / "flat.zarr", progress=False)
        with pytest.raises(ValueError, match="no layer groups"):
            GeoStack.open(path)


class TestLaziness:
    """Every transform on an unbounded surface must leave the pixels on disk."""

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda r: r.astype("float32"), id="astype"),
            pytest.param(lambda r: r.rename_bands({"B04": "red"}), id="rename_bands"),
            pytest.param(lambda r: r.rebase(tags={"a": "b"}), id="rebase"),
            pytest.param(lambda r: r.reproject("EPSG:32633", resolution=20), id="reproject"),
            pytest.param(lambda r: r.astype("int32", nodata=-1).remap({1: 2}), id="remap"),
            pytest.param(lambda r: r.merge_spatial(make_lazy_raster(x0=ORIGIN_X + 640)), id="merge_spatial"),
            pytest.param(
                lambda r: r.concat(make_lazy_raster().rename_bands({"B04": "B08"}), dim="band"),
                id="concat_band",
            ),
        ],
    )
    def test_transform_keeps_the_pixels_on_disk(self, operation):
        assert is_lazy(operation(make_lazy_raster()))

    def test_open_then_tiles_never_reads(self, store):
        path = make_raster().to_geotiff(store / "x.tif", chunk_px=32, progress=False)
        back = GeoRaster.open(path)
        assert is_lazy(back)
        assert is_lazy(next(iter(back.tiles(tile_size_px=32))).to_raster())

    def test_a_mosaic_of_many_pieces_stays_lazy(self):
        pieces = [make_lazy_raster(width=32, x0=ORIGIN_X + 320 * index) for index in range(6)]
        merged = pieces[0].merge_spatial(*pieces[1:])
        assert is_lazy(merged)
        assert merged.anchor.shape == (64, 32 * 6)

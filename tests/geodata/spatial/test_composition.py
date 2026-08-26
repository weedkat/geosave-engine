"""Strict composition — concat, mosaic, reproject.

Composition never infers. Every mismatch a caller could fix explicitly is an
error naming the fix, because silently reprojecting, resampling, cropping or
promoting dtypes produces a plausible-looking wrong answer.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from geosave_engine.geodata.spatial import GeoMosaic

from .conftest import ORIGIN_X, is_lazy, make_anchor, make_lazy_raster, make_raster


def dated(day: int, band: str = "B04"):
    """Single-step raster whose one timestamp is 2024-01-`day`."""
    return make_raster(
        bands=(band,),
        dtype="float32",
        nodata=0,
        times=[datetime(2024, 1, day)],
        time=f"2024-01-{day:02d}",
    )


class TestConcatBand:
    def test_joins_bands_in_argument_order(self):
        joined = make_raster(bands=("B04",)).concat(make_raster(bands=("B08",)), dim="band")
        assert joined.bands == ("B04", "B08")

    def test_rejects_a_repeated_band_name(self):
        with pytest.raises(ValueError, match="land in more than one raster"):
            make_raster(bands=("B04",)).concat(make_raster(bands=("B04",)), dim="band")

    def test_rejects_a_different_grid(self):
        with pytest.raises(ValueError, match="one exact grid"):
            make_raster(bands=("B04",)).concat(
                make_raster(bands=("B08",), width=32, height=32), dim="band"
            )

    def test_rejects_a_different_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            make_raster(bands=("B04",)).concat(
                make_raster(bands=("B08",), dtype="int16"), dim="band"
            )

    def test_rejects_a_different_nodata(self):
        with pytest.raises(ValueError, match="nodata"):
            make_raster(bands=("B04",)).concat(
                make_raster(bands=("B08",), nodata=255), dim="band"
            )

    def test_clears_the_tiling_stamp(self):
        tiled = next(iter(make_raster(bands=("B04",)).tiles(tile_size_px=64))).to_raster()
        assert tiled.tiling is not None
        assert tiled.concat(make_raster(bands=("B08",)), dim="band").tiling is None


class TestConcatTime:
    def test_orders_by_timestamp_not_argument_order(self):
        joined = dated(20).concat(dated(1), dim="time")
        labels = [np.datetime64(value, "D") for value in joined.data.time.values]
        assert labels == sorted(labels)

    def test_span_is_rederived_from_the_merged_labels(self):
        joined = dated(1).concat(dated(20), dim="time")
        assert joined.timespan is not None
        assert joined.timespan[0].day == 1 and joined.timespan[1].day == 20

    def test_rejects_a_timeless_input(self):
        with pytest.raises(ValueError, match="requires every raster to have a time dimension"):
            dated(1).concat(make_raster(bands=("B04",), dtype="float32"), dim="time")

    def test_rejects_a_duplicate_timestamp(self):
        with pytest.raises(ValueError, match="land in more than one raster"):
            dated(1).concat(dated(1), dim="time")

    def test_rejects_differing_bands(self):
        with pytest.raises(ValueError, match="bands"):
            dated(1, "B04").concat(dated(20, "B08"), dim="time")


class TestMosaic:
    def test_adjacent_footprints_union(self):
        left = make_raster(bands=("B04",), width=32)
        right = make_raster(bands=("B04",), width=32, x0=ORIGIN_X + 320)
        merged = left.merge_spatial(right)
        assert merged.anchor.shape == (64, 64)
        assert merged.dtype == np.dtype("uint16")

    def test_gaps_between_footprints_hold_nodata(self):
        left = make_raster(bands=("B04",), width=16, nodata=7)
        right = make_raster(bands=("B04",), width=16, x0=ORIGIN_X + 480)
        merged = left.merge_spatial(right.rebase(nodata=7))
        assert 7 in np.unique(merged.data.values)

    def test_north_up_orientation_survives(self):
        left = make_raster(bands=("B04",), width=32)
        right = make_raster(bands=("B04",), width=32, x0=ORIGIN_X + 320)
        merged = left.merge_spatial(right)
        assert merged.data.y.values[0] > merged.data.y.values[-1]

    def test_stays_lazy(self):
        left = make_lazy_raster(width=32)
        right = make_lazy_raster(width=32, x0=ORIGIN_X + 320)
        assert is_lazy(left.merge_spatial(right))

    def test_rejects_a_fractional_grid_offset(self):
        left = make_raster(bands=("B04",), width=32)
        right = make_raster(bands=("B04",), width=32, x0=ORIGIN_X + 325)
        with pytest.raises(ValueError, match="one pixel grid"):
            left.merge_spatial(right)

    def test_rejects_a_different_resolution(self):
        left = make_raster(bands=("B04",), width=32)
        right = make_raster(bands=("B04",), width=32, resolution=20, x0=ORIGIN_X + 320)
        with pytest.raises(ValueError, match="pixel basis"):
            left.merge_spatial(right)

    def test_rejects_a_different_dtype(self):
        left = make_raster(bands=("B04",), width=32)
        right = make_raster(bands=("B04",), width=32, dtype="float32", x0=ORIGIN_X + 320)
        with pytest.raises(ValueError, match="dtype"):
            left.merge_spatial(right)

    def test_needs_a_sentinel_to_fill_the_gaps_with(self):
        with pytest.raises(ValueError, match="nodata"):
            make_raster(nodata=None).merge_spatial()

    def test_a_single_input_still_honours_the_contract(self):
        mosaic = GeoMosaic()
        mosaic.add(make_raster(nodata=None))
        with pytest.raises(ValueError, match="nodata"):
            mosaic.result()

    def test_a_single_input_clears_the_tiling_stamp(self):
        tiled = next(iter(make_raster().tiles(tile_size_px=32))).to_raster()
        mosaic = GeoMosaic()
        mosaic.add(tiled)
        assert mosaic.result().tiling is None

    def test_result_needs_at_least_one_raster(self):
        with pytest.raises(ValueError, match="at least one raster"):
            GeoMosaic().result()

    def test_add_rejects_a_non_raster(self):
        with pytest.raises(TypeError):
            GeoMosaic().add(object())


class TestReproject:
    def test_to_a_coarser_resolution(self):
        out = make_raster(bands=("B04",)).reproject("EPSG:32633", resolution=20)
        assert out.anchor.resolution == 20.0

    def test_clears_the_tiling_stamp(self):
        tiled = next(iter(make_raster(bands=("B04",)).tiles(tile_size_px=32))).to_raster()
        assert tiled.reproject("EPSG:32633", resolution=20).tiling is None

    def test_reproject_like_lands_on_the_exact_grid(self):
        target = make_raster(bands=("B04",))
        source = make_raster(bands=("dem",), resolution=20, width=32, height=32, dtype="int16", nodata=-9999)
        assert source.reproject_like(target).anchor.geobox == target.anchor.geobox

    def test_reproject_like_is_a_no_op_on_the_same_grid(self):
        raster = make_raster(bands=("B04",))
        assert raster.reproject_like(raster) is raster

    def test_ground_the_source_misses_becomes_nodata(self):
        target = make_raster(bands=("B04",))
        offset = make_raster(bands=("dem",), width=32, height=32, x0=ORIGIN_X + 320, dtype="int16", nodata=-9999)
        warped = offset.reproject_like(target)
        assert -9999 in np.unique(warped.data.values)

    def test_integer_pixels_need_a_sentinel_to_fill_with(self):
        target = make_raster(bands=("B04",), width=32, height=32)
        source = make_raster(bands=("dem",), dtype="int16", nodata=None)
        with pytest.raises(ValueError, match="rebase\\(nodata="):
            source.reproject_like(target)

    def test_needs_a_target_or_a_resolution(self):
        with pytest.raises(ValueError, match="needs a target grid/CRS"):
            make_raster().reproject()

    def test_stays_lazy(self):
        assert is_lazy(make_lazy_raster().reproject("EPSG:32633", resolution=20))


class TestResampleTime:
    def test_buckets_steps_and_records_the_spec(self):
        anchor = make_anchor(time=("2024-01-01", "2024-01-31"))
        steps = [datetime(2024, 1, day) for day in (1, 5, 20, 25)]
        raster = anchor.to_raster(
            np.ones((4, 1, 64, 64), dtype="float32"), bands=["B04"], times=steps
        )
        monthly = raster.resample_time("ME", "mean")
        assert monthly.data.sizes["time"] == 1
        assert monthly.timespec is not None

    def test_span_covers_the_whole_bucket_not_just_the_label(self):
        """`ME` bins are `(prev month end, this month end]`, so the span reaches
        back past the first observation rather than starting at its label."""
        anchor = make_anchor(time=("2024-01-01", "2024-01-31"))
        steps = [datetime(2024, 1, day) for day in (1, 25)]
        raster = anchor.to_raster(
            np.ones((2, 1, 64, 64), dtype="float32"), bands=["B04"], times=steps
        )
        monthly = raster.resample_time("ME", "mean")
        assert monthly.timespan is not None
        start, end = monthly.timespan
        assert start <= steps[0] and end >= steps[-1]
        assert (monthly.data.time.values[0]).astype("datetime64[D]") == np.datetime64("2024-01-31")

    def test_needs_a_time_dimension(self):
        with pytest.raises(ValueError, match="needs a raster with a time dim"):
            make_raster().resample_time("ME", "mean")


def test_area_needs_a_projected_crs():
    geographic = make_anchor(crs="EPSG:4326", x0=13.0, y0=52.0, resolution=0.001)
    with pytest.raises(ValueError, match="projected CRS"):
        _ = geographic.area_m2


def test_stem_is_deterministic():
    assert make_anchor().stem == make_anchor().stem


def test_stem_records_extent_time_and_resolution():
    stem = make_anchor(width=64, height=64, resolution=10).stem
    assert "640m" in stem and "10m" in stem and "20240115" in stem

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.core.raster import raster as build_raster
from geosave_engine.geodata.transform.composite import interpolate, reduce, resample

UTM = "EPSG:32633"

# Two observations in January, one in March, so February covers none.
OBSERVED = np.array(["2024-01-05", "2024-01-20", "2024-03-08"], "datetime64[ns]")


def utm_box() -> GeoBox:
    """Build a small projected grid."""
    return GeoBox.from_bbox((300000, 5000000, 300020, 5000020), crs=UTM, resolution=10)


def series(
    *values: int, nodata: int | None = None, dtype: str = "uint16"
) -> xr.Dataset:
    """Build a raster whose every pixel reads `values[i]` at `OBSERVED[i]`."""
    box = utm_box()
    pixels = np.stack([np.full(box.shape, value, dtype) for value in values])
    return build_raster({"red": pixels}, box, nodata=nodata, time=OBSERVED)


def test_the_bucketing_survives_the_reduction() -> None:
    # xarray carries coordinate attrs through a groupby reduce, which is the only
    # reason a reduced axis still says what its labels stand for.
    monthly = reduce(resample(series(100, 900, 500), "ME"), "median")

    bucketing = monthly.gs.attrs.coords["time"].get(attrs.TimeSpec)

    assert bucketing is not None
    assert bucketing.time_freq == "ME"
    assert (bucketing.time_closed, bucketing.time_label) == ("right", "right")


def test_the_result_is_the_kind_the_resampler_holds() -> None:
    scene = series(100, 900, 500)

    assert isinstance(reduce(resample(scene, "MS"), "median"), xr.Dataset)
    assert isinstance(reduce(resample(scene.red, "MS"), "median"), xr.DataArray)


def test_flag_values_refuse_a_reducer_that_ranks_them() -> None:
    # min returns a real class code, which is why it reads as safe and is not:
    # it asserts an order over codes that carry none.
    coded = series(4, 5, 4, dtype="uint8").gs.rebase(
        attrs.Legend(class_map={4: "veg", 5: "water"}), target="red"
    )

    with pytest.raises(ValueError, match="class codes"):
        reduce(resample(coded, "MS"), "min")


def test_a_flag_variable_alone_refuses_the_same_reducer() -> None:
    coded = series(4, 5, 4, dtype="uint8").gs.rebase(
        attrs.Legend(class_map={4: "veg", 5: "water"}), target="red"
    )

    with pytest.raises(ValueError, match="class codes"):
        reduce(resample(coded.red, "MS"), "min")


def test_flag_values_take_a_point_sample() -> None:
    coded = series(4, 5, 4, dtype="uint8").gs.rebase(
        attrs.Legend(class_map={4: "veg", 5: "water"}), target="red"
    )

    sampled = reduce(resample(coded, "MS"), "first")

    assert sampled.red.values[0, 0, 0] == 4
    assert sampled.red.attrs["cell_methods"] == "time: point"


def test_the_nodata_value_is_left_out_of_the_reduction() -> None:
    # January reads its nodata value and 900, so the month observed only 900.
    scene = series(0, 900, 500, nodata=0)

    assert reduce(resample(scene, "MS"), "mean").red.values[0, 0, 0] == 900


def test_keeping_the_nodata_value_blends_it_in() -> None:
    scene = series(0, 900, 500, nodata=0)

    kept = reduce(resample(scene, "MS", skip_nodata=False), "mean")

    assert kept.red.values[0, 0, 0] == 450


def monthly(*values: int) -> xr.Dataset:
    """Collapse `series` into monthly buckets, February observing none."""
    return reduce(resample(series(*values), "MS"), "median")


def test_a_bucket_observing_nothing_holds_its_label_and_absence() -> None:
    buckets = monthly(100, 100, 900)

    # resample cuts every bucket its cadence covers, so nothing is missing from
    # the axis; February is present and absent throughout.
    assert [str(label)[:7] for label in buckets.time.values] == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]
    assert bool(buckets.red.isel(time=1).isnull().all())


def test_nearest_repeats_the_nearest_observed_bucket() -> None:
    filled = interpolate(monthly(100, 100, 900), "nearest")

    # February sits 31 days after January and 29 before March, so March is nearer.
    assert filled.red.values[:, 0, 0].tolist() == [100.0, 900.0, 900.0]


def test_linear_reads_between_the_observed_buckets() -> None:
    filled = interpolate(monthly(100, 100, 900), "linear")

    # Weighted by real elapsed days, not by bucket count.
    assert filled.red.values[1, 0, 0] == pytest.approx(513.333, abs=0.01)


def quarterly() -> xr.Dataset:
    """Collapse four months of readings, each pixel reading 100, 200, 300, 400."""
    box = utm_box()
    stamps = np.array(
        ["2024-01-05", "2024-02-05", "2024-03-05", "2024-04-05"], "datetime64[ns]"
    )
    pixels = np.stack(
        [np.full(box.shape, value, "uint16") for value in (100, 200, 300, 400)]
    )
    return reduce(
        resample(build_raster({"red": pixels}, box, time=stamps), "MS"), "median"
    )


def test_a_partly_observed_bucket_fills_only_the_cells_it_missed() -> None:
    clouded = quarterly()
    clouded.red.values[1, 0, 0] = np.nan

    filled = interpolate(clouded, "nearest")

    # February sits 31 days after January and 29 before March, so March is nearer.
    assert filled.red.values[1, 0, 0] == 300.0
    assert filled.red.values[1, 1, 1] == 200.0


def test_a_cell_observed_once_stays_absent() -> None:
    lonely = quarterly()
    lonely.red.values[:3, 0, 0] = np.nan

    filled = interpolate(lonely, "nearest")

    # Interpolating reads between two observations, and this cell has one.
    assert bool(np.isnan(filled.red.values[:3, 0, 0]).all())
    assert filled.red.values[0, 1, 1] == 100.0


def test_interpolating_one_variable_returns_a_data_array() -> None:
    buckets = reduce(resample(series(100, 100, 900).red, "MS"), "median")

    filled = interpolate(buckets, "nearest")

    assert isinstance(filled, xr.DataArray)
    assert filled.values[:, 0, 0].tolist() == [100.0, 900.0, 900.0]


def test_linear_refuses_to_blend_class_codes() -> None:
    coded = monthly(4, 4, 5).gs.rebase(
        attrs.Legend(class_map={4: "veg", 5: "water"}), target="red"
    )

    with pytest.raises(ValueError, match="class codes"):
        interpolate(coded, "linear")

    # Repeating a code names a class, so the nearest reading is allowed.
    assert interpolate(coded, "nearest").red.values[1, 0, 0] == 5


def test_an_axis_of_instants_refuses_to_be_filled() -> None:
    with pytest.raises(ValueError, match="names instants rather than buckets"):
        interpolate(series(100, 100, 900), "nearest")


def test_a_raster_without_a_time_axis_refuses_to_be_filled() -> None:
    timeless = series(100, 100, 900).isel(time=0, drop=True)

    with pytest.raises(ValueError, match="carries no 'time' coordinate"):
        interpolate(timeless, "nearest")


def test_the_bucketing_survives_the_filling() -> None:
    filled = interpolate(monthly(100, 100, 900), "nearest")

    bucketing = filled.gs.attrs.coords["time"].get(attrs.TimeSpec)

    assert bucketing is not None
    assert bucketing.time_freq == "MS"

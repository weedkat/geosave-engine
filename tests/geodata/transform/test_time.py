from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geosave_engine.geodata.attrs import TimeSpec
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.core.stack import stack as build_stack
from geosave_engine.geodata.transform import time as transform_time

from .conftest import build, geobox


def test_concat_joins_operands_in_order() -> None:
    box = geobox()

    joined = transform_time.concat([build(box), build(box, "2024-02-01")])

    assert joined.sizes["time"] == 4
    assert [str(label)[:10] for label in joined.time.values] == [
        "2024-01-01",
        "2024-01-02",
        "2024-02-01",
        "2024-02-02",
    ]


def test_concat_preserves_dtype_grid_and_agreed_attrs() -> None:
    box = geobox()

    joined = transform_time.concat([build(box), build(box, "2024-02-01")])

    assert joined.red.dtype == np.dtype("uint16")
    assert joined.gs.geobox == box
    assert joined.attrs["license"] == "CC0"
    assert joined.red.attrs["scale_factor"] == pytest.approx(1e-4)


def test_concat_drops_an_attr_the_operands_disagree_on() -> None:
    box = geobox()
    later = build(box, "2024-02-01")
    later.attrs["title"] = "february"

    joined = transform_time.concat([build(box), later])

    assert "title" not in joined.attrs


def test_concat_keeps_dask_arrays_lazy() -> None:
    box = geobox()
    chunks = {"time": 1}

    joined = transform_time.concat(
        [build(box, chunks=chunks), build(box, "2024-02-01", chunks=chunks)]
    )

    assert joined.red.chunks is not None


def test_concat_refuses_no_rasters() -> None:
    with pytest.raises(ValueError, match="at least one raster"):
        transform_time.concat([])


def test_concat_refuses_a_timeless_raster() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="no 'time' coordinate"):
        transform_time.concat([build(box).isel(time=0, drop=True), build(box)])


def test_concat_refuses_descending_operands() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="descend across the operands"):
        transform_time.concat([build(box, "2024-02-01"), build(box)])


def test_concat_refuses_overlapping_labels() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="overlapping time labels"):
        transform_time.concat([build(box), build(box)])


def test_concat_refuses_a_promoted_dtype() -> None:
    box = geobox()
    later = build(box, "2024-02-01")
    later["red"] = later.red.astype("float32")

    with pytest.raises(ValueError, match="different dtype"):
        transform_time.concat([build(box), later])


def test_concat_refuses_a_different_variable_set() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="select or merge them first"):
        transform_time.concat([build(box), build(box, "2024-02-01", labelled=True)])


def test_concat_refuses_a_different_grid() -> None:
    box = geobox()
    other = geobox((400_000.0, 5_100_000.0, 400_320.0, 5_100_320.0))

    with pytest.raises(ValueError, match="reproject or resample"):
        transform_time.concat([build(box), build(other, "2024-02-01")])


def test_concat_refuses_mixing_resampled_and_raw_operands() -> None:
    box = geobox()
    bucketed = transform_time.resample(build(box, times=40), "MS", "first")

    with pytest.raises(ValueError, match="some operands record a resample"):
        transform_time.concat([bucketed, build(box, "2024-03-01")])


def test_concat_refuses_mixing_cadences() -> None:
    box = geobox()
    winter = transform_time.resample(build(box, times=40), "MS", "first")
    spring = transform_time.resample(build(box, "2024-03-01", times=60), "ME", "first")

    with pytest.raises(ValueError, match="bucket every operand at the same cadence"):
        transform_time.concat([winter, spring])


def test_concat_refuses_mixing_reducers_at_the_same_cadence() -> None:
    box = geobox()
    winter = transform_time.resample(build(box, times=40), "MS", "first")
    spring = transform_time.resample(build(box, "2024-03-01", times=60), "MS", "median")

    with pytest.raises(ValueError, match="bucket every operand at the same cadence"):
        transform_time.concat([winter, spring])


def test_concat_of_bucketed_rasters_spans_every_bucket() -> None:
    box = geobox()
    winter = transform_time.resample(build(box, times=40), "MS", "first")
    spring = transform_time.resample(build(box, "2024-03-01", times=60), "MS", "first")

    joined = transform_time.concat([winter, spring])

    assert joined["time_bnds"].shape == (joined.sizes["time"], 2)
    assert [(str(lo)[:7], str(hi)[:7]) for lo, hi in joined["time_bnds"].values] == [
        ("2024-01", "2024-02"),
        ("2024-02", "2024-03"),
        ("2024-03", "2024-04"),
        ("2024-04", "2024-05"),
    ]


def test_resample_buckets_and_records_the_grid() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=40), "ME", "first")

    assert monthly.sizes["time"] == 2
    assert monthly.attrs["freq"] == "ME"
    assert monthly.attrs["time_method"] == "first"
    assert monthly.attrs["time_closed"] == "right"


def test_resample_reports_bucket_spans_through_timespan() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=40), "MS", "first")
    start, end = monthly.gs.timespan

    assert (start.year, start.month, start.day) == (2024, 1, 1)
    assert (end.year, end.month) == (2024, 2)


def test_resample_writes_half_open_time_bounds() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=40), "MS", "first")

    bounds = monthly["time_bnds"]
    assert bounds.dims == ("time", "bnds")
    assert [(str(lo)[:10], str(hi)[:10]) for lo, hi in bounds.values] == [
        ("2024-01-01", "2024-02-01"),
        ("2024-02-01", "2024-03-01"),
    ]


def test_resample_time_bounds_track_a_dropped_bucket() -> None:
    box = geobox()
    sparse = build(box, times=3).assign_coords(
        time=np.array(
            ["2024-01-05", "2024-01-19", "2024-03-11"], dtype="datetime64[ns]"
        )
    )

    monthly = transform_time.resample(sparse, "MS", "first")

    assert [str(label)[:10] for label in monthly.time.values] == [
        "2024-01-01",
        "2024-03-01",
    ]
    assert [(str(lo)[:10], str(hi)[:10]) for lo, hi in monthly["time_bnds"].values] == [
        ("2024-01-01", "2024-02-01"),
        ("2024-03-01", "2024-04-01"),
    ]


def test_resample_marks_blending_reducers_with_cell_methods() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=40), "MS", "mean")

    assert monthly.gs.attrs.root.get(TimeSpec).cell_methods == "time: mean"


def test_resample_leaves_value_picking_reducers_without_cell_methods() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=40), "MS", "first")

    assert "cell_methods" not in monthly["red"].attrs


def test_resample_keeps_packing_when_the_reducer_picks_a_value() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=40), "ME", "first")

    assert monthly.red.dtype == np.dtype("uint16")
    assert monthly.red.attrs["scale_factor"] == pytest.approx(1e-4)


def test_resample_removes_packing_when_the_reducer_blends() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=40), "ME", "median")

    assert monthly.red.dtype == np.dtype("float64")
    assert "scale_factor" not in monthly.red.attrs


def test_resample_refuses_a_second_pass() -> None:
    box = geobox()

    monthly = transform_time.resample(build(box, times=90), "MS", "first")

    with pytest.raises(ValueError, match="already records a resample"):
        transform_time.resample(monthly, "YS", "first")


def test_resample_keeps_dask_arrays_lazy() -> None:
    box = geobox()

    monthly = transform_time.resample(
        build(box, times=40, chunks={"time": 10}), "ME", "first"
    )

    assert monthly.red.chunks is not None


def test_resample_refuses_a_timeless_raster() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="no 'time' coordinate"):
        transform_time.resample(build(box).isel(time=0, drop=True), "ME")


def test_resample_refuses_upsampling() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="it upsamples"):
        transform_time.resample(build(box, times=2), "h", "first")


def test_resample_refuses_blending_a_categorical_variable() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="carry a class map"):
        transform_time.resample(build(box, times=40, labelled=True), "ME", "median")


def test_resample_keeps_a_class_map_through_a_picking_reducer() -> None:
    box = geobox()

    monthly = transform_time.resample(
        build(box, times=40, labelled=True), "ME", "first"
    )

    assert monthly.cls.attrs["class_map"] == {0: "bg", 1: "crop"}


def test_resample_refuses_an_unknown_reducer() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="is not a resample reducer"):
        transform_time.resample(build(box, times=40), "ME", "nope")


def gappy() -> xr.Dataset:
    """Build a raster observing January and March, with February empty."""
    box = geobox()
    return transform_time.concat(
        [build(box, "2024-01-01", times=5), build(box, "2024-03-01", times=5)]
    )


def test_resample_drops_an_empty_bucket_by_default() -> None:
    with pytest.warns(transform_time.DroppedBucketsWarning, match="1 of 3 buckets"):
        monthly = transform_time.resample(gappy(), "ME", "first")

    assert [str(label)[:7] for label in monthly.time.values] == ["2024-01", "2024-03"]
    assert monthly.red.dtype == np.dtype("uint16")
    assert monthly.red.attrs["scale_factor"] == pytest.approx(1e-4)


def test_resample_warns_nothing_when_no_bucket_is_empty() -> None:
    box = geobox()

    with warnings.catch_warnings():
        warnings.simplefilter("error", transform_time.DroppedBucketsWarning)
        transform_time.resample(build(box, times=40), "MS", "first")


def test_interpolate_keeps_an_empty_bucket_as_nodata() -> None:
    dropped = transform_time.resample(gappy(), "ME", "first")

    filled = transform_time.interpolate(dropped, "keep")

    assert filled.sizes["time"] == 3
    assert np.isnan(filled.red.values[1]).all()


def test_interpolate_fills_an_empty_bucket_from_the_nearest() -> None:
    dropped = transform_time.resample(gappy(), "ME", "first")

    filled = transform_time.interpolate(dropped, "nearest")

    assert filled.sizes["time"] == 3
    assert not np.isnan(filled.red.values).any()
    assert filled.red.dtype == np.dtype("uint16")


def test_interpolate_fills_an_empty_bucket_by_interpolation() -> None:
    dropped = transform_time.resample(gappy(), "ME", "median")

    filled = transform_time.interpolate(dropped, "linear")

    assert filled.sizes["time"] == 3
    assert not np.isnan(filled.red.values).any()


def test_interpolate_is_idempotent_on_a_gapless_axis() -> None:
    box = geobox()
    monthly = transform_time.resample(build(box, times=40), "MS", "first")

    filled = transform_time.interpolate(monthly, "nearest")

    assert filled.sizes["time"] == monthly.sizes["time"]
    assert (filled.time.values == monthly.time.values).all()


def test_interpolate_fills_whole_slices_only() -> None:
    raster = gappy()
    raster["red"].values[0, 0, 0] = 0  # a nodata pixel inside an observed bucket
    dropped = transform_time.resample(raster, "ME", "first")

    filled = transform_time.interpolate(dropped, "nearest")

    assert filled.red.values[0, 0, 0] == 0


def test_interpolate_refuses_a_categorical_variable_with_linear() -> None:
    box = geobox()
    labelled = transform_time.concat(
        [
            build(box, "2024-01-01", times=5, labelled=True),
            build(box, "2024-03-01", times=5, labelled=True),
        ]
    )
    dropped = transform_time.resample(labelled, "ME", "first")

    with pytest.raises(ValueError, match="empty='linear' would"):
        transform_time.interpolate(dropped, "linear")


def test_interpolate_refuses_no_timespec() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="no TimeSpec"):
        transform_time.interpolate(build(box), "nearest")


def _yearly(box) -> xr.Dataset:
    """Build a two-year raster with distinguishable values, resampled to yearly."""
    times = pd.date_range("2023-01-01", periods=2, freq="YS")
    values = np.stack([np.full(box.shape, 1.0), np.full(box.shape, 2.0)])
    ds = raster({"dem": values}, box, time=times).gs.write_nodata(0.0)
    return transform_time.resample(ds, "YS", "first")


def test_broadcast_stretches_a_timeless_raster_across_labels() -> None:
    box = geobox()
    timeless = raster({"dem": np.full(box.shape, 5.0)}, box, nodata=0.0)
    labels = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=6, freq="MS"))

    stretched = transform_time.broadcast(timeless, labels)

    assert stretched.sizes["time"] == 6
    assert (stretched["dem"].values == 5.0).all()


def test_broadcast_repeats_each_bucket_across_the_finer_labels_it_covers() -> None:
    yearly = _yearly(geobox())
    labels = pd.DatetimeIndex(pd.date_range("2023-01-01", "2024-12-01", freq="MS"))

    monthly = transform_time.broadcast(yearly, labels)

    assert monthly.sizes["time"] == 24
    assert (monthly["dem"].values[:12] == 1.0).all()
    assert (monthly["dem"].values[12:] == 2.0).all()


def test_broadcast_drops_time_bnds_and_timespec() -> None:
    yearly = _yearly(geobox())
    labels = pd.DatetimeIndex(pd.date_range("2023-01-01", "2023-12-01", freq="MS"))

    monthly = transform_time.broadcast(yearly, labels)

    assert "time_bnds" not in monthly.coords
    assert "freq" not in monthly.attrs


def test_broadcast_keeps_dask_arrays_lazy() -> None:
    box = geobox()
    yearly = _yearly(box).chunk({"time": 1})
    labels = pd.DatetimeIndex(pd.date_range("2023-01-01", "2024-12-01", freq="MS"))

    monthly = transform_time.broadcast(yearly, labels)

    assert monthly["dem"].chunks is not None


def test_broadcast_keeps_a_timeless_raster_lazy() -> None:
    box = geobox()
    timeless = (
        raster({"dem": np.full(box.shape, 5.0)}, box)
        .gs.write_nodata(0.0)
        .chunk({"x": 2})
    )
    labels = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=6, freq="MS"))

    stretched = transform_time.broadcast(timeless, labels)

    assert stretched["dem"].chunks is not None


def test_broadcast_refuses_a_raster_with_no_time_bnds() -> None:
    box = geobox()
    times = pd.date_range("2023-01-01", periods=2, freq="YS")
    raw = raster(
        {"dem": np.stack([np.full(box.shape, 1.0), np.full(box.shape, 2.0)])},
        box,
        time=times,
    ).gs.write_nodata(0.0)
    labels = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=6, freq="MS"))

    with pytest.raises(ValueError, match="no 'time_bnds'"):
        transform_time.broadcast(raw, labels)


def test_broadcast_refuses_labels_outside_the_covered_span() -> None:
    yearly = _yearly(geobox())
    labels = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=3, freq="MS"))

    with pytest.raises(ValueError, match="outside its covered span"):
        transform_time.broadcast(yearly, labels)


def test_broadcast_refuses_a_finer_source() -> None:
    box = geobox()
    daily = transform_time.resample(build(box, times=90), "D", "first")
    labels = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="MS"))

    with pytest.raises(ValueError, match="more finely than labels"):
        transform_time.broadcast(daily, labels)


def test_broadcast_warns_and_drops_an_unneeded_source_bucket() -> None:
    yearly = _yearly(geobox())
    labels = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=12, freq="MS"))

    with pytest.warns(
        transform_time.UnmatchedBucketsWarning, match="left out of the result"
    ):
        monthly = transform_time.broadcast(yearly, labels)

    assert monthly.sizes["time"] == 12
    assert (monthly["dem"].values == 1.0).all()


def _monthly(box, periods: int = 12) -> xr.Dataset:
    """Build a raster with one distinct value per month, resampled to monthly."""
    times = pd.date_range("2023-01-01", periods=periods, freq="MS")
    values = np.stack([np.full(box.shape, float(i)) for i in range(periods)])
    ds = raster({"val": values}, box, time=times).gs.write_nodata(-1.0)
    return transform_time.resample(ds, "MS", "first")


def test_time_window_cuts_overlapping_windows_by_stride() -> None:
    monthly = _monthly(geobox())

    windows = transform_time.time_window(monthly, 3, stride=1)

    assert len(windows) == 10
    assert windows[0]["val"].values[:, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert windows[1]["val"].values[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert windows[-1]["val"].values[:, 0, 0].tolist() == [9.0, 10.0, 11.0]


def test_time_window_defaults_stride_to_slot() -> None:
    monthly = _monthly(geobox())

    windows = transform_time.time_window(monthly, 3)

    assert len(windows) == 4
    assert windows[1]["val"].values[:, 0, 0].tolist() == [3.0, 4.0, 5.0]


def test_time_window_refuses_no_timespec() -> None:
    box = geobox()
    raw = raster(
        {"val": np.full((1, *box.shape), 1.0)},
        box,
        time=pd.date_range("2023-01-01", periods=1, freq="MS"),
    ).gs.write_nodata(-1.0)

    with pytest.raises(ValueError, match="no TimeSpec"):
        transform_time.time_window(raw, 1)


def test_time_window_refuses_a_slot_larger_than_the_axis() -> None:
    monthly = _monthly(geobox())

    with pytest.raises(ValueError, match="larger than the raster's own"):
        transform_time.time_window(monthly, 20)


def test_time_window_refuses_a_gappy_axis() -> None:
    box = geobox()
    times = pd.to_datetime(
        ["2023-01-05", "2023-01-20", "2023-04-02", "2023-04-15", "2023-04-28"]
    )
    ds = raster(
        {"val": np.stack([np.full(box.shape, float(i)) for i in range(5)])},
        box,
        time=times,
    ).gs.write_nodata(-1.0)
    gappy = transform_time.resample(ds, "MS", "first")

    with pytest.raises(ValueError, match="missing bucket"):
        transform_time.time_window(gappy, 2)


def test_time_window_accepts_a_gap_filled_axis() -> None:
    box = geobox()
    times = pd.to_datetime(
        ["2023-01-05", "2023-01-20", "2023-04-02", "2023-04-15", "2023-04-28"]
    )
    ds = raster(
        {"val": np.stack([np.full(box.shape, float(i)) for i in range(5)])},
        box,
        time=times,
    ).gs.write_nodata(-1.0)
    dropped = transform_time.resample(ds, "MS", "first")
    filled = transform_time.interpolate(dropped, "nearest")

    windows = transform_time.time_window(filled, 2)

    assert len(windows) == 2


def _stacked(box) -> xr.DataTree:
    """Build a stack pairing a monthly group with a broadcast yearly one."""
    monthly = _monthly(box)
    yearly = _yearly(box)
    held = transform_time.broadcast(yearly, pd.DatetimeIndex(monthly.time.values))
    return build_stack({"sentinel-2-l2a": monthly, "dem": held})


def test_time_window_stack_anchors_every_group_on_the_same_calendar_position() -> None:
    scene = _stacked(geobox())

    windows = transform_time.time_window_stack(
        scene, {"sentinel-2-l2a": 3, "dem": 1}, stride=1
    )

    assert len(windows) == 10
    for window in windows:
        assert window.gs.groups == ("sentinel-2-l2a", "dem")
        s2_last = window["sentinel-2-l2a"].time.values[-1]
        dem_only = window["dem"].time.values[0]
        assert s2_last == dem_only
        assert window["sentinel-2-l2a"].sizes["time"] == 3
        assert window["dem"].sizes["time"] == 1


def test_time_window_stack_keeps_the_shared_geobox() -> None:
    scene = _stacked(geobox())

    windows = transform_time.time_window_stack(
        scene, {"sentinel-2-l2a": 3, "dem": 1}, stride=1
    )

    assert windows[0].gs.geobox == scene.gs.geobox


def test_time_window_stack_refuses_a_group_missing_from_slot() -> None:
    scene = _stacked(geobox())

    with pytest.raises(ValueError, match="name no slot"):
        transform_time.time_window_stack(scene, {"sentinel-2-l2a": 3}, stride=1)


def test_time_window_stack_refuses_a_non_positive_slot() -> None:
    scene = _stacked(geobox())

    with pytest.raises(ValueError, match="must be positive"):
        transform_time.time_window_stack(
            scene, {"sentinel-2-l2a": 3, "dem": 0}, stride=1
        )


def test_time_window_stack_refuses_a_slot_wider_than_the_shared_axis() -> None:
    scene = _stacked(geobox())

    with pytest.raises(ValueError, match="larger than the shared axis"):
        transform_time.time_window_stack(
            scene, {"sentinel-2-l2a": 30, "dem": 1}, stride=1
        )


def test_time_window_stack_refuses_a_stack_with_no_time_axis() -> None:
    box = geobox()
    static = build_stack(
        {"dem": raster({"dem": np.full(box.shape, 1.0)}, box, nodata=-1.0)}
    )

    with pytest.raises(ValueError, match="carries no time axis"):
        transform_time.time_window_stack(static, {"dem": 1}, stride=1)

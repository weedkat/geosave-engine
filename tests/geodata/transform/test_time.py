from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.raster import raster as build_raster
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.warnings import (
    DroppedInstantsWarning,
    DroppedFramesWarning,
    GeoSaveWarning,
    UncoveredInstantsWarning,
)
from geosave_engine.geodata.transform.time import frames, stack_frames

UTM = "EPSG:32633"

# Optical whenever it is clear, radar on its own repeat cycle.
OPTICAL = np.array(
    ["2024-01-10", "2024-03-08", "2024-03-28", "2024-04-12", "2024-07-03"],
    "datetime64[ns]",
)
RADAR = np.array(["2024-03-05", "2024-04-04", "2024-07-01"], "datetime64[ns]")


def utm_box() -> GeoBox:
    """Build a small projected grid."""
    return GeoBox.from_bbox((300000, 5000000, 300020, 5000020), crs=UTM, resolution=10)


def series(
    observed: np.ndarray, name: str = "red", dtype: str = "uint16"
) -> xr.Dataset:
    """Build a raster observing once per label in `observed`."""
    box = utm_box()
    pixels = np.stack(
        [np.full(box.shape, index, dtype) for index in range(len(observed))]
    )
    return build_raster({name: pixels}, box, time=observed)


def flat(name: str = "elevation", dtype: str = "float32") -> xr.Dataset:
    """Build a raster spanning no time at all."""
    box = utm_box()
    return build_raster({name: np.ones(box.shape, dtype)}, box)


def scene() -> xr.DataTree:
    """Build the stack every pairing rule acts on.

    Jan-10 lies before radar ever flew, Jul-03 lies after it stopped yet still
    answers the last instant, and Apr-04 names the scenes Mar-28 already named.

    Returns:
        Stack of optical, radar and elevation on one grid.
    """
    return stack(
        {
            "s2": series(OPTICAL, "red"),
            "s1": series(RADAR, "vv", "int16"),
            "dem": flat(),
        }
    )


def months(count: int) -> xr.Dataset:
    """Build a raster observing on `count` consecutive month starts."""
    return series(pd.date_range("2024-01-01", periods=count, freq="MS").values)


def dates(raster: xr.Dataset | xr.DataArray) -> list[str]:
    """Read a raster's time labels as `YYYY-MM-DD` strings."""
    return [str(label)[:10] for label in raster.coords["time"].values]


@contextmanager
def quiet() -> Iterator[None]:
    """Ignore warnings a pairing raises beside whatever is under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeoSaveWarning)
        yield


def test_frames_abut_when_no_stride_is_given() -> None:
    quarters = frames(months(12), 3)

    assert len(quarters) == 4
    assert dates(quarters[0]) == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert dates(quarters[3]) == ["2024-10-01", "2024-11-01", "2024-12-01"]


def test_a_stride_below_length_overlaps_frames() -> None:
    halves = frames(months(12), 6, stride=3)

    assert [dates(half)[0] for half in halves] == [
        "2024-01-01",
        "2024-04-01",
        "2024-07-01",
    ]
    assert dates(halves[0])[-1] == "2024-06-01"
    assert dates(halves[1])[0] == "2024-04-01"


def test_a_stride_above_size_skips_the_instants_between() -> None:
    pairs = frames(months(12), 2, stride=5)

    assert [dates(pair) for pair in pairs] == [
        ["2024-01-01", "2024-02-01"],
        ["2024-06-01", "2024-07-01"],
        ["2024-11-01", "2024-12-01"],
    ]


def test_instants_that_fill_no_frame_are_warned_about() -> None:
    with pytest.warns(DroppedInstantsWarning, match="2024-07-01"):
        cut = frames(months(7), 3)

    assert len(cut) == 2
    assert dates(cut[-1])[-1] == "2024-06-01"


def test_a_frame_the_length_of_the_axis_keeps_every_instant() -> None:
    whole = frames(months(5), 5)

    assert len(whole) == 1
    assert dates(whole[0]) == dates(months(5))


def test_variables_that_do_not_span_time_ride_into_every_frame() -> None:
    box = utm_box()
    mixed = build_raster(
        {"red": np.zeros((4, *box.shape), "uint16"), "dem": (np.ones(box.shape), ())},
        box,
        time=pd.date_range("2024-01-01", periods=4, freq="MS").values,
    )

    cut = frames(mixed, 2)

    assert [frame.red.sizes["time"] for frame in cut] == [2, 2]
    assert all("time" not in frame.dem.dims for frame in cut)
    assert all(float(frame.dem.mean()) == 1.0 for frame in cut)


def test_a_frame_stays_lazy_while_its_raster_is() -> None:
    lazy = months(6).chunk({"time": 1})

    first = frames(lazy, 3)[0]

    assert first.red.chunks is not None
    assert np.array_equal(first.red.compute().values, months(6).red.values[:3])


@pytest.mark.parametrize(
    "raster",
    [
        xr.DataArray(np.zeros((2, 2)), dims=("y", "x")),
        xr.DataArray(np.zeros((3, 2, 2)), dims=("time", "y", "x")),
        xr.DataArray(np.zeros((2, 2)), dims=("y", "x")).assign_coords(
            time=pd.Timestamp("2024-01-01")
        ),
    ],
    ids=["no time axis", "time axis without labels", "a scalar time label"],
)
def test_cutting_refuses_a_raster_naming_no_sequence(raster: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="no 'time' dimension coordinate"):
        frames(raster, 2)


def test_cutting_refuses_a_length_or_stride_that_places_no_frame() -> None:
    monthly = months(6)
    with pytest.raises(ValueError, match="holds no observations"):
        frames(monthly, 0)
    with pytest.raises(ValueError, match="never advances"):
        frames(monthly, 2, stride=0)
    with pytest.raises(ValueError, match="does not fit the 6"):
        frames(monthly, 7)


def test_groups_observing_on_their_own_dates_share_one_axis() -> None:
    with quiet():
        cut = stack_frames(scene(), 2, tolerance="10D")

    assert len(cut) == 2
    assert dates(cut[0]["s2"]) == ["2024-03-08", "2024-03-28"]
    assert dates(cut[0]["s1"]) == ["2024-03-05", "2024-04-04"]
    assert dates(cut[1]["s2"]) == ["2024-04-12", "2024-07-03"]
    assert dates(cut[1]["s1"]) == ["2024-04-04", "2024-07-01"]


def test_instants_outside_the_shared_interval_name_no_slot() -> None:
    with pytest.warns(UncoveredInstantsWarning, match=r"\{'s2': 2\}"):
        cut = stack_frames(scene(), 2, tolerance="10D")

    observed = {label for frame in cut for label in dates(frame["s2"])}
    assert "2024-01-10" not in observed


def test_a_scene_outside_the_shared_interval_still_answers_its_edge() -> None:
    with quiet():
        cut = stack_frames(scene(), 2, tolerance="10D")

    assert "2024-07-03" in dates(cut[1]["s2"])


def test_one_scene_may_answer_two_instants_under_its_own_date() -> None:
    with quiet():
        cut = stack_frames(scene(), 2, tolerance="10D")

    answered = [label for frame in cut for label in dates(frame["s1"])]
    assert answered.count("2024-04-04") == 2


def test_every_group_keeps_the_dtype_it_was_given() -> None:
    with quiet():
        first = stack_frames(scene(), 2, tolerance="10D")[0]

    assert first["s2"].red.dtype == np.dtype("uint16")
    assert first["s1"].vv.dtype == np.dtype("int16")
    assert first["dem"].elevation.dtype == np.dtype("float32")


def test_a_group_spanning_no_time_joins_every_frame_whole() -> None:
    with quiet():
        cut = stack_frames(scene(), 2, tolerance="10D")

    for frame in cut:
        assert frame.gs.groups == ("s2", "s1", "dem")
        assert "time" not in frame["dem"].dims
        assert float(frame["dem"].elevation.mean()) == 1.0


def test_strict_refuses_a_frame_covering_an_instant_a_group_missed() -> None:
    with (
        quiet(),
        pytest.raises(ValueError, match=r"\['s1'\] observed nothing within 5 days"),
    ):
        stack_frames(scene(), 2, tolerance="5D")


def test_drop_skips_the_frames_it_cannot_fill_and_says_how_many() -> None:
    with pytest.warns(GeoSaveWarning) as raised:
        cut = stack_frames(scene(), 2, tolerance="5D", mode="drop")

    assert cut == ()
    skipped = [
        warning for warning in raised if warning.category is DroppedFramesWarning
    ]
    assert len(skipped) == 1
    assert "2 of 2 frames" in str(skipped[0].message)


def test_a_gap_exactly_the_tolerance_still_reaches_a_scene() -> None:
    anchor = series(np.array(["2024-01-01", "2024-01-11"], "datetime64[ns]"), "red")
    probe = series(np.array(["2024-01-06", "2024-01-11"], "datetime64[ns]"), "vv")

    with quiet():
        paired = stack_frames(stack({"a": anchor, "b": probe}), 2, tolerance="5D")

    assert len(paired) == 1
    assert dates(paired[0]["b"]) == ["2024-01-06", "2024-01-11"]


def test_a_group_spanning_no_instants_is_refused() -> None:
    box = utm_box()
    empty = build_raster(
        {"red": np.zeros((0, *box.shape), "uint16")},
        box,
        time=np.array([], "datetime64[ns]"),
    )

    with pytest.raises(ValueError, match="spans no instants at all"):
        stack_frames(stack({"a": empty, "b": series(RADAR)}), 1, tolerance="5D")


def test_an_unlabelled_instant_is_refused_rather_than_read_as_disorder() -> None:
    ragged = series(
        np.array(["2024-03-06", "NaT", "2024-03-20"], "datetime64[ns]"), "red"
    )

    with pytest.raises(ValueError, match="unlabelled instant"):
        stack_frames(stack({"a": ragged, "b": series(RADAR)}), 1, tolerance="5D")


def test_cutting_a_stack_refuses_a_mode_it_cannot_apply() -> None:
    with pytest.raises(ValueError, match="not a frame mode"):
        stack_frames(scene(), 2, tolerance="10D", mode="skip")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "tolerance", ["0D", np.timedelta64("NaT")], ids=["zero", "not a time"]
)
def test_cutting_a_stack_refuses_a_tolerance_reaching_nothing(
    tolerance: object,
) -> None:
    with pytest.raises(ValueError, match="reaches no scene"):
        stack_frames(scene(), 2, tolerance=tolerance)  # type: ignore[arg-type]


def test_cutting_a_stack_refuses_one_holding_no_sequence() -> None:
    with pytest.raises(ValueError, match="no group of this stack spans"):
        stack_frames(stack({"dem": flat()}), 1, tolerance="10D")


def test_unchronological_or_repeated_labels_are_refused() -> None:
    shuffled = series(OPTICAL, "red").isel(time=[2, 0, 1, 3, 4])
    with pytest.raises(ValueError, match="not chronological"):
        stack_frames(
            stack({"a": shuffled, "b": series(RADAR, "vv")}), 1, tolerance="5D"
        )

    twice = series(np.array(["2024-03-05", "2024-03-05"], "datetime64[ns]"), "vv")
    with pytest.raises(ValueError, match="same instant more than once"):
        stack_frames(stack({"a": series(OPTICAL), "b": twice}), 1, tolerance="5D")


def test_groups_observing_over_no_shared_interval_are_refused() -> None:
    elsewhere = series(np.array(["2030-01-01", "2030-02-01"], "datetime64[ns]"), "vv")

    with pytest.raises(ValueError, match="no shared interval"):
        stack_frames(stack({"a": series(OPTICAL), "b": elsewhere}), 1, tolerance="10D")


def test_a_group_reaching_no_instant_is_refused() -> None:
    straddling = series(np.array(["2024-01-01", "2024-12-01"], "datetime64[ns]"), "red")
    inside = series(
        np.array(["2024-05-01", "2024-06-01", "2024-07-01"], "datetime64[ns]"), "vv"
    )

    with quiet(), pytest.raises(ValueError, match=r"\['a'\] observed nothing within"):
        stack_frames(stack({"a": straddling, "b": inside}), 2, tolerance="5D")


def test_frames_matches_stack_frames_of_a_lone_group() -> None:
    lone = series(OPTICAL, "red")

    for size, stride in [(1, None), (2, None), (2, 1), (3, 2), (5, None)]:
        with quiet():
            alone = frames(lone, size, stride=stride)
            stacked = stack_frames(
                stack({"only": lone}), size, stride=stride, tolerance="1ns"
            )
        assert len(alone) == len(stacked)
        for plain, grouped in zip(alone, stacked, strict=True):
            assert dates(plain) == dates(grouped["only"])
            assert np.array_equal(plain.red.values, grouped["only"].red.values)


def test_frames_land_wherever_the_stride_reaches() -> None:
    for length in range(1, 17):
        monthly = months(length)
        every = dates(monthly)
        for size in range(1, min(6, length + 1)):
            for stride in (None, 1, 2, 3, 5):
                step = size if stride is None else stride
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DroppedInstantsWarning)
                    cut = frames(monthly, size, stride=stride)

                reached = size + (length - size) // step * step
                assert [dates(frame)[0] for frame in cut] == [
                    every[at] for at in range(0, reached - size + 1, step)
                ]
                assert all(len(dates(frame)) == size for frame in cut)


def test_frames_never_invent_a_date_or_reorder_time() -> None:
    rng = np.random.default_rng(20240914)
    checked = 0
    for _ in range(30):
        groups, observed = {}, {}
        for index in range(int(rng.integers(1, 4))):
            count = int(rng.integers(1, 12))
            offsets = np.sort(rng.choice(np.arange(0, 200), size=count, replace=False))
            labels = np.array(
                [
                    np.datetime64("2024-01-01") + np.timedelta64(int(o), "D")
                    for o in offsets
                ],
                "datetime64[ns]",
            )
            groups[f"g{index}"] = series(labels, f"v{index}")
            observed[f"g{index}"] = set(labels.tolist())

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cut = stack_frames(
                    stack(groups),
                    int(rng.integers(1, 5)),
                    tolerance=f"{int(rng.integers(1, 60))}D",
                    mode="drop",
                )
        except ValueError:
            continue
        checked += 1

        for frame in cut:
            for name in groups:
                labels = frame[name].coords["time"].values
                assert set(labels.tolist()) <= observed[name], "invented a date"
                assert (np.diff(labels.astype("int64")) >= 0).all(), (
                    "time ran backwards"
                )
    assert checked > 15

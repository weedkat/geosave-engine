"""How frames pair time labels, cut from windows that name labels alone."""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geosave_engine.geodata import cuts
from geosave_engine.geodata.warnings import (
    DroppedFramesWarning,
    DroppedInstantsWarning,
    GeoSaveWarning,
    UncoveredInstantsWarning,
)

# Optical whenever it is clear, radar on its own repeat cycle.
OPTICAL = ["2024-01-10", "2024-03-08", "2024-03-28", "2024-04-12", "2024-07-03"]
RADAR = ["2024-03-05", "2024-04-04", "2024-07-01"]


def windows(**groups: list[str | None] | None) -> gpd.GeoDataFrame:
    """Build one whole window whose groups observe on the given dates."""
    times = {
        name: None
        if labels is None
        else [None if label is None else f"{label}T00:00:00" for label in labels]
        for name, labels in groups.items()
    }
    row = {
        "id": "s0",
        "parent": None,
        "stack": "s0",
        "times": times,
        "start_datetime": None,
        "end_datetime": None,
        "crs": "EPSG:4326",
        "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
        "row_off": 0,
        "col_off": 0,
        "height": 2,
        "width": 2,
        "geometry": box(0, 0, 2, 2),
    }
    return gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326")


def scene() -> gpd.GeoDataFrame:
    return windows(s2=OPTICAL, s1=RADAR, dem=None)


def months(count: int) -> gpd.GeoDataFrame:
    labels = pd.date_range("2024-01-01", periods=count, freq="MS")
    return windows(image=[str(label)[:10] for label in labels])


def dates(cut: gpd.GeoDataFrame, group: str) -> list[list[str]]:
    return [[label[:10] for label in times[group]] for times in cut["times"]]


def quiet_frames(*args, **kwargs) -> gpd.GeoDataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeoSaveWarning)
        return cuts.frames(*args, **kwargs)


def test_abutting_frames_cover_each_instant_once() -> None:
    quarters = cuts.frames(months(12), 3, tolerance="1D")

    assert [labels[0] for labels in dates(quarters, "image")] == [
        "2024-01-01",
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
    ]


def test_a_stride_below_the_length_overlaps_frames() -> None:
    halves = cuts.frames(months(12), 6, stride=3, tolerance="1D")

    assert [labels[0][:7] for labels in dates(halves, "image")] == [
        "2024-01",
        "2024-04",
        "2024-07",
    ]


def test_a_stride_above_the_length_skips_instants() -> None:
    pairs = cuts.frames(months(12), 2, stride=5, tolerance="1D")

    assert [labels[0][:7] for labels in dates(pairs, "image")] == [
        "2024-01",
        "2024-06",
        "2024-11",
    ]


def test_instants_that_fill_no_frame_warn() -> None:
    with pytest.warns(DroppedInstantsWarning):
        cut = cuts.frames(months(7), 3, tolerance="1D")

    assert len(cut) == 2


def test_one_frame_may_span_the_whole_axis() -> None:
    assert len(cuts.frames(months(5), 5, tolerance="1D")) == 1


def test_a_length_or_stride_that_cuts_nothing_refuses() -> None:
    with pytest.raises(ValueError):
        cuts.frames(months(6), 0, tolerance="1D")
    with pytest.raises(ValueError):
        cuts.frames(months(6), 2, stride=0, tolerance="1D")
    with pytest.raises(ValueError, match="does not fit"):
        cuts.frames(months(6), 7, tolerance="1D")


def test_instants_outside_the_shared_interval_name_no_slot() -> None:
    with pytest.warns(UncoveredInstantsWarning, match=r"\{'s2': 2\}"):
        cut = cuts.frames(scene(), 2, tolerance="10D")

    observed = {label for labels in dates(cut, "s2") for label in labels}
    assert "2024-01-10" not in observed


def test_a_scene_outside_the_shared_interval_still_answers_its_edge() -> None:
    cut = quiet_frames(scene(), 2, tolerance="10D")

    assert "2024-07-03" in dates(cut, "s2")[1]


def test_one_scene_may_answer_two_instants_under_its_own_date() -> None:
    cut = quiet_frames(scene(), 2, tolerance="10D")

    answered = [label for labels in dates(cut, "s1") for label in labels]
    assert answered.count("2024-04-04") == 2


def test_strict_names_the_group_that_missed_an_instant() -> None:
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match=r"\['s1'\] observed nothing within 5 days"),
    ):
        warnings.simplefilter("ignore", GeoSaveWarning)
        cuts.frames(scene(), 2, tolerance="5D")


def test_drop_says_how_many_frames_it_skipped() -> None:
    with pytest.warns(GeoSaveWarning) as raised:
        cut = cuts.frames(scene(), 2, tolerance="5D", mode="drop")

    assert len(cut) == 0
    skipped = [each for each in raised if each.category is DroppedFramesWarning]
    assert len(skipped) == 1
    assert "2 of 2 frames" in str(skipped[0].message)


def test_a_gap_exactly_the_tolerance_still_reaches_a_scene() -> None:
    paired = quiet_frames(
        windows(a=["2024-01-01", "2024-01-11"], b=["2024-01-06", "2024-01-11"]),
        2,
        tolerance="5D",
    )

    assert dates(paired, "b") == [["2024-01-06", "2024-01-11"]]


def test_a_group_spanning_no_instants_refuses() -> None:
    with pytest.raises(ValueError, match="spans no instants at all"):
        cuts.frames(windows(a=[], b=RADAR), 1, tolerance="5D")


def test_an_unlabelled_instant_refuses_rather_than_reading_as_disorder() -> None:
    ragged = ["2024-03-06", None, "2024-03-20"]

    with pytest.raises(ValueError, match="unlabelled instant"):
        cuts.frames(windows(a=ragged, b=RADAR), 1, tolerance="5D")


def test_a_tolerance_that_is_no_duration_refuses() -> None:
    with pytest.raises(ValueError, match="reaches no scene"):
        cuts.frames(scene(), 2, tolerance=np.timedelta64("NaT"))


def test_unchronological_or_repeated_labels_refuse() -> None:
    shuffled = [OPTICAL[index] for index in (2, 0, 1, 3, 4)]
    with pytest.raises(ValueError, match="not chronological"):
        cuts.frames(windows(a=shuffled, b=RADAR), 1, tolerance="5D")

    twice = ["2024-03-05", "2024-03-05"]
    with pytest.raises(ValueError, match="same instant more than once"):
        cuts.frames(windows(a=OPTICAL, b=twice), 1, tolerance="5D")


def test_groups_observing_over_no_shared_interval_refuse() -> None:
    elsewhere = ["2030-01-01", "2030-02-01"]

    with pytest.raises(ValueError, match="no shared interval"):
        cuts.frames(windows(a=OPTICAL, b=elsewhere), 1, tolerance="10D")


def test_a_group_reaching_no_instant_refuses() -> None:
    straddling = ["2024-01-01", "2024-12-01"]
    inside = ["2024-05-01", "2024-06-01", "2024-07-01"]

    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match=r"\['a'\] observed nothing within"),
    ):
        warnings.simplefilter("ignore", GeoSaveWarning)
        cuts.frames(windows(a=straddling, b=inside), 2, tolerance="5D")

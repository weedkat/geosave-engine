"""Windows are cut along time into frames, each group on its own dates."""

from __future__ import annotations

import pytest

from geosave_engine.geodata import cuts
from geosave_engine.geodata.warnings import DroppedFramesWarning, DroppedInstantsWarning


def test_frames_take_each_groups_own_nearest_dates(items) -> None:
    cut = cuts.frames(cuts.stacks(items), 2, tolerance="10D")

    assert cut["id"].tolist() == ["s0/frame-0", "s0/frame-1"]
    assert cut["parent"].tolist() == ["s0", "s0"]
    first, second = cut["times"]
    assert first["s2"] == ["2024-03-08T00:00:00", "2024-03-28T00:00:00"]
    assert first["s1"] == ["2024-03-05T00:00:00", "2024-04-04T00:00:00"]
    # The frame ends on Jul 1, yet optical answers with the scene it took on Jul 3.
    assert second["s2"] == ["2024-04-12T00:00:00", "2024-07-03T00:00:00"]


def test_a_timeless_group_joins_every_frame_whole(items) -> None:
    cut = cuts.frames(cuts.stacks(items), 2, tolerance="10D")

    assert [times["dem"] for times in cut["times"]] == [None, None]


def test_a_frame_states_what_its_instants_span(items) -> None:
    cut = cuts.frames(cuts.stacks(items), 2, tolerance="10D")

    assert str(cut.iloc[0]["start_datetime"])[:10] == "2024-03-05"
    assert str(cut.iloc[0]["end_datetime"])[:10] == "2024-04-04"


def test_a_frame_keeps_its_windows_pixels(items, grid) -> None:
    cut = cuts.frames(cuts.stacks(items), 2, tolerance="10D")

    assert set(zip(cut["height"], cut["width"], strict=True)) == {tuple(grid.shape)}
    assert cut.crs.to_epsg() == 4326


def test_an_instant_no_group_reaches_refuses_when_strict(items) -> None:
    with pytest.raises(ValueError, match="observed nothing within"):
        cuts.frames(cuts.stacks(items), 2, tolerance="5D")


def test_incomplete_frames_are_dropped_with_a_warning(items) -> None:
    with pytest.warns(DroppedFramesWarning):
        cut = cuts.frames(cuts.stacks(items), 2, tolerance="5D", mode="drop")

    assert len(cut) == 0


def test_instants_that_fill_no_frame_warn(items) -> None:
    with pytest.warns(DroppedInstantsWarning):
        cut = cuts.frames(cuts.stacks(items), 3, tolerance="10D")

    assert cut["id"].tolist() == ["s0/frame-0"]


def test_windows_with_no_dated_group_refuse(items) -> None:
    timeless = cuts.stacks(items[items["collection"].isin(["dem", "label"])])

    with pytest.raises(ValueError, match="states time labels"):
        cuts.frames(timeless, 2, tolerance="10D")


@pytest.mark.parametrize(
    ("options", "message"),
    [({"mode": "skip"}, "not a frame mode"), ({"tolerance": "0D"}, "reaches no scene")],
)
def test_bad_settings_refuse(items, options, message) -> None:
    with pytest.raises(ValueError, match=message):
        cuts.frames(cuts.stacks(items), 2, **{"tolerance": "10D", **options})

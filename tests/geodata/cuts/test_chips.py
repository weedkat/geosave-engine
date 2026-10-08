"""Windows are cut in space into chips of one size."""

from __future__ import annotations

import numpy as np
import pytest

from geosave_engine.geodata import cuts


def test_chips_without_overlap_tile_the_window(items) -> None:
    cut = cuts.chips(cuts.stacks(items), 32)

    # 50 by 70 pixels in 32-pixel chips: two rows of three.
    assert cut["id"].tolist() == [f"s0/chip-{number}" for number in range(6)]
    assert cut["chip"].tolist() == list(range(6))
    assert cut["parent"].unique().tolist() == ["s0"]
    assert cut[["row_off", "col_off"]].to_numpy().tolist() == [
        [0, 0],
        [0, 32],
        [0, 64],
        [32, 0],
        [32, 32],
        [32, 64],
    ]
    assert set(zip(cut["height"], cut["width"], strict=True)) == {(32, 32)}
    assert cut.iloc[0]["halo"] == [[0, 0], [0, 0]]


def test_overlapping_chips_are_laid_over_a_halo(items) -> None:
    cut = cuts.chips(cuts.stacks(items), 32, overlap=8)

    tiler, halo = cuts.layout((50, 70), (32, 32), overlap=8)
    assert len(cut) == len(tiler)
    assert cut.iloc[0]["halo"] == [list(widths) for widths in halo]
    assert (cut.iloc[0]["row_off"], cut.iloc[0]["col_off"]) == (
        -halo[0][0],
        -halo[1][0],
    )
    assert cut.iloc[0]["mode"] == "reflect"


def test_a_chip_states_its_own_grid_and_footprint(items, grid) -> None:
    cut = cuts.chips(cuts.stacks(items), 32)

    chip = grid.translate_pix(32, 32).crop((32, 32))
    row = cut.iloc[4]
    assert row["transform"] == pytest.approx(list(chip.transform)[:6])
    assert row.geometry.equals_exact(chip.extent.to_crs("EPSG:4326").geom, 1e-9)


def test_chips_of_a_frame_keep_its_instants_and_name_it(items) -> None:
    framed = cuts.frames(cuts.stacks(items), 2, tolerance="10D")

    cut = cuts.chips(framed, 32)

    assert cut.iloc[0]["id"] == "s0/frame-0/chip-0"
    assert cut.iloc[0]["parent"] == "s0/frame-0"
    assert cut.iloc[0]["times"] == framed.iloc[0]["times"]
    assert len(cut) == 2 * 6


def test_a_size_states_height_then_width(items) -> None:
    cut = cuts.chips(cuts.stacks(items), (25, 35))

    assert set(zip(cut["height"], cut["width"], strict=True)) == {(25, 35)}
    assert len(cut) == 4


def test_a_layout_is_the_same_every_time() -> None:
    first, halo = cuts.layout((300, 500), (128, 128), overlap=32)
    second, again = cuts.layout((300, 500), (128, 128), overlap=32)

    assert halo == again
    assert len(first) == len(second)
    assert all(
        np.array_equal(first.get_tile_bbox(n)[0], second.get_tile_bbox(n)[0])
        for n in range(len(first))
    )

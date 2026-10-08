"""Chip outputs saved during prediction are merged back into whole rasters."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import cuts, raster

from tests.geodata.conftest import whole_windows


@pytest.fixture
def parent():
    grid = GeoBox.from_bbox(
        (300000, 5000000, 300190, 5000130), "EPSG:32633", resolution=10
    )
    rng = np.random.default_rng(3)
    return raster(
        {
            "a": (("y", "x"), rng.random(grid.shape).astype("float32")),
            "b": (("y", "x"), rng.random(grid.shape).astype("float32")),
        },
        grid,
    )


def _outputs(parent, windows) -> xr.DataArray:
    """Stand in for a model: each chip's output is its own two bands."""
    chips = [
        np.stack(
            [
                cuts.select_pixels(parent, row.to_dict())[name].values
                for name in ("a", "b")
            ]
        )
        for _, row in windows.iterrows()
    ]
    return xr.DataArray(
        np.stack(chips),
        dims=("id", "band", "y", "x"),
        coords={"id": windows["id"].to_numpy()},
        name="logits",
    )


@pytest.mark.parametrize(("taper", "tolerance"), [(None, 0.0), ("hann", 1e-5)])
def test_saved_chips_merge_back_into_the_raster(parent, taper, tolerance) -> None:
    whole = whole_windows({"scene": parent})
    windows = cuts.chips(whole, (6, 8), overlap=2)
    outputs = _outputs(parent, windows)
    shuffled = outputs.isel(id=np.random.default_rng(7).permutation(len(windows)))

    merged = cuts.merge(whole, windows, shuffled, taper=taper)

    assert list(merged) == ["scene"]
    scene = merged["scene"]
    assert scene.gs.geobox == parent.gs.geobox
    np.testing.assert_allclose(
        scene["logits"].values,
        np.stack([parent.a.values, parent.b.values]),
        atol=tolerance,
    )


def test_chips_written_in_two_parts_merge_as_one(parent) -> None:
    whole = whole_windows({"scene": parent})
    windows = cuts.chips(whole, (6, 8), overlap=2)
    outputs = _outputs(parent, windows)
    parts = xr.concat(
        [outputs.isel(id=slice(None, 5)), outputs.isel(id=slice(5, None))], "id"
    )

    merged = cuts.merge(whole, windows, parts)

    np.testing.assert_array_equal(merged["scene"]["logits"].values[0], parent.a.values)


def test_each_parent_is_merged_on_its_own_grid(parent) -> None:
    whole = whole_windows({"north": parent, "south": parent * 2})
    windows = cuts.chips(whole, (6, 8))
    outputs = xr.concat(
        [
            _outputs(parent, windows[windows["parent"] == "north"]),
            _outputs(parent * 2, windows[windows["parent"] == "south"]),
        ],
        "id",
    )

    merged = cuts.merge(whole, windows, outputs)

    assert sorted(merged) == ["north", "south"]
    np.testing.assert_array_equal(
        merged["south"]["logits"].values[1], (parent.b * 2).values
    )


def test_a_parent_missing_a_chip_refuses(parent) -> None:
    whole = whole_windows({"scene": parent})
    windows = cuts.chips(whole, (6, 8))
    outputs = _outputs(parent, windows).isel(id=slice(1, None))

    with pytest.raises(ValueError, match="scene/chip-0"):
        cuts.merge(whole, windows, outputs)

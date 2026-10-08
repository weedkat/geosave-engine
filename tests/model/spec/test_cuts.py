from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from geosave_engine.model.spec import FramesSpec, ChipsSpec


def _grid() -> GeoBox:
    return GeoBox.from_bbox((0, 0, 80, 80), "EPSG:32748", resolution=10)


def test_a_square_size_names_both_sides():
    assert ChipsSpec(size=224).shape == (224, 224)
    assert ChipsSpec(size=[224, 256]).shape == (224, 256)


@pytest.mark.parametrize("size", [0, [224], [224, 0], [1, 2, 3]])
def test_a_size_that_names_no_tile_is_refused(size):
    with pytest.raises(ValueError):
        ChipsSpec(size=size)


def test_a_window_needs_overlapping_tiles():
    with pytest.raises(ValueError, match="overlap"):
        ChipsSpec(size=4, window="hann")


def _windows() -> gpd.GeoDataFrame:
    """One whole window over an 8 by 8 sample with a dated and a timeless group."""
    grid = _grid()
    return gpd.GeoDataFrame(
        [
            {
                "id": "s0",
                "parent": None,
                "stack": "s0",
                "times": {
                    "s2": [
                        f"2024-01-{day}T00:00:00" for day in ("01", "11", "21", "31")
                    ],
                    "dem": None,
                },
                "start_datetime": None,
                "end_datetime": None,
                "crs": "EPSG:32748",
                "transform": list(grid.transform)[:6],
                "row_off": 0,
                "col_off": 0,
                "height": 8,
                "width": 8,
                "geometry": grid.extent.to_crs("EPSG:4326").geom,
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_layout_is_native_and_only_needs_pixel_dimensions():
    from tiler import Tiler

    layout, halo = ChipsSpec(size=4, overlap=2, mode="constant").layout((8, 8))
    assert isinstance(layout, Tiler)
    assert tuple(layout.tile_shape) == (4, 4)
    assert (layout.overlap, layout.mode) == (2, "constant")
    assert halo == [(1, 1), (1, 1)]
    assert layout.get_tile(np.zeros(tuple(layout.data_shape)), 0).shape == (4, 4)


def test_frames_are_cut_as_declared():
    cut = FramesSpec(length=2, tolerance="1D").cut(_windows())

    assert cut["id"].tolist() == ["s0/frame-0", "s0/frame-1"]
    assert [len(times["s2"]) for times in cut["times"]] == [2, 2]
    assert all(times["dem"] is None for times in cut["times"])


def test_chips_are_cut_as_declared():
    cut = ChipsSpec(size=4).cut(_windows())

    assert cut["chip"].tolist() == [0, 1, 2, 3]
    assert set(zip(cut["height"], cut["width"], strict=True)) == {(4, 4)}


@pytest.mark.parametrize(
    "window", ["hann", "bartlett", "barthann", "bohman", "blackman", "overlap-tile"]
)
def test_taper_rejects_single_pixel_overlap_that_leaves_seams(window):
    with pytest.raises(ValueError, match="at least 2"):
        ChipsSpec(size=8, overlap=1, window=window)


@pytest.mark.parametrize(
    "window", ["boxcar", "hamming", "triang", "parzen", "nuttall", "blackmanharris"]
)
def test_positive_edge_windows_allow_single_pixel_overlap(window):
    assert ChipsSpec(size=8, overlap=1, window=window).overlap == 1

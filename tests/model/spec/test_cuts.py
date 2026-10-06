from __future__ import annotations

import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster, stack
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


def test_layout_is_native_and_only_needs_pixel_dimensions():
    from tiler import Tiler

    layout = ChipsSpec(size=4, overlap=2, mode="constant").tiler((8, 8))
    assert isinstance(layout, Tiler)
    assert tuple(layout.tile_shape) == (4, 4)
    assert (layout.overlap, layout.mode) == (2, "constant")
    assert layout.get_tile(np.zeros((8, 8)), 0).shape == (4, 4)


def test_frames_are_cut_as_declared():
    times = np.array(
        ["2024-01-01", "2024-01-11", "2024-01-21", "2024-01-31"], "datetime64[ns]"
    )
    series = raster({"red": np.zeros((4, 8, 8), dtype="float32")}, _grid(), time=times)
    flat = raster({"elevation": np.ones((8, 8), dtype="float32")}, _grid())

    cut = FramesSpec(length=2, tolerance="1D").cut(stack({"s2": series, "dem": flat}))

    assert [frame["s2"].sizes["time"] for frame in cut] == [2, 2]
    assert all("time" not in frame["dem"].dims for frame in cut)


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

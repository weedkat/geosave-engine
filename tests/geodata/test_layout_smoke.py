from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from geosave_engine.geodata.attrs import GeoTIFFTags, Nodata, rebase
from geosave_engine.geodata.errors import DroppedAttrsWarning
from geosave_engine.geodata.utils.io.layout import FlatLayout, Layout, NestedLayout

from .conftest import build_raster


def write_scenes(root: Path, layout: Layout, tweak=lambda ds, position: ds) -> Path:
    """Write a two-instant cube as a tree, one scene at a time.

    Writing scene by scene lets a test state each one's attrs separately, which
    is what a tree read has to reconcile.

    Args:
        root: Directory the tree is written into.
        layout: Layout arranging the leaves.
        tweak: Applied to each scene with its position, before it is written.

    Returns:
        The directory holding the tree.
    """
    cube = build_raster(times=2)
    for position, instant in enumerate(cube.time.values):
        layout.write(tweak(cube.sel(time=[instant]), position), root)
    return root


@pytest.mark.parametrize(
    "layout",
    [
        NestedLayout(),
        NestedLayout(split_bands=False),
        FlatLayout(),
        FlatLayout(split_bands=False),
    ],
    ids=["nested", "nested-joined", "flat", "flat-joined"],
)
def test_every_layout_round_trips_a_cube(tmp_path: Path, layout: Layout) -> None:
    written = build_raster(times=2)

    layout.write(written, tmp_path / "scene")
    restored = layout.read(tmp_path / "scene")

    # A tree records no variable order: one leaf per variable reads back by
    # filename, so only a joined-band layout keeps the cube's own order.
    assert set(restored.data_vars) == set(written.data_vars)
    assert restored.sizes["time"] == 2
    assert restored.gs.geobox == written.gs.geobox
    for name, variable in written.data_vars.items():
        assert np.array_equal(restored[name].values, variable.values)


@pytest.mark.parametrize(
    "layout",
    [NestedLayout(split_bands=False), FlatLayout(split_bands=False)],
    ids=["nested-joined", "flat-joined"],
)
def test_a_joined_band_layout_keeps_the_variable_order(
    tmp_path: Path, layout: Layout
) -> None:
    written = build_raster(times=2)

    layout.write(written, tmp_path / "scene")

    # One leaf holds every band, so band order carries the cube's order.
    assert list(layout.read(tmp_path / "scene").data_vars) == list(written.data_vars)


def test_the_leaves_agree_on_a_tag_and_it_survives(tmp_path: Path) -> None:
    root = write_scenes(
        tmp_path / "scene",
        NestedLayout(),
        lambda ds, position: rebase(ds, GeoTIFFTags(TIFFTAG_ARTIST="geosave")),
    )

    assert NestedLayout().read(root).attrs["TIFFTAG_ARTIST"] == "geosave"


def test_a_tag_the_leaves_state_differently_drops_and_warns(tmp_path: Path) -> None:
    root = write_scenes(
        tmp_path / "scene",
        NestedLayout(),
        lambda ds, position: rebase(
            ds, GeoTIFFTags(TIFFTAG_ARTIST=f"artist{position}")
        ),
    )

    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_ARTIST"):
        restored = NestedLayout().read(root)

    assert "TIFFTAG_ARTIST" not in restored.attrs


def test_leaves_declaring_different_absent_pixels_refuse_to_join(
    tmp_path: Path,
) -> None:
    root = write_scenes(
        tmp_path / "scene",
        NestedLayout(),
        lambda ds, position: rebase(ds, Nodata(_FillValue=position), target="red"),
    )

    # A join cannot pick which stored value means absent, so it refuses.
    with pytest.raises(ValueError, match="objects disagree on nodata"):
        NestedLayout().read(root)


def test_the_instant_leaves_the_tags_once_it_spans_an_axis(tmp_path: Path) -> None:
    written = build_raster(times=2)

    NestedLayout().write(written, tmp_path / "scene")
    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_DATETIME"):
        restored = NestedLayout().read(tmp_path / "scene")

    # Each leaf stamped its own instant; the cube carries them on `time` instead.
    assert "TIFFTAG_DATETIME" not in restored.attrs
    assert np.array_equal(restored.time.values, written.time.values)


def test_a_directory_holding_no_leaf_refuses(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="holds no .tif leaf"):
        NestedLayout().read(empty)


def test_a_cube_spanning_more_than_time_refuses_to_write(tmp_path: Path) -> None:
    spanning = build_raster(times=2).expand_dims({"model": ["a", "b"]})

    with pytest.raises(ValueError, match="a leaf holds one instant of one grid"):
        NestedLayout().write(spanning, tmp_path / "scene")


def test_a_safe_product_refuses_to_be_written(tmp_path: Path) -> None:
    from geosave_engine.geodata.utils.io.layout import SAFELayout

    with pytest.raises(NotImplementedError, match="ESA's own identifiers"):
        SAFELayout().write(build_raster(), tmp_path / "scene")


def test_one_leaf_reads_without_a_directory(tmp_path: Path) -> None:
    written = build_raster()

    NestedLayout().write(written, tmp_path / "dem")
    restored = NestedLayout().read(tmp_path / "dem" / "red")

    assert list(restored.data_vars) == ["red"]
    assert isinstance(restored, xr.Dataset)

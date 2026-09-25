from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import xarray as xr

from geosave_engine.geodata.attrs import CFVariable, GeoTIFFTags, Nodata, rebase
from geosave_engine.geodata.errors import DroppedAttrsWarning
from geosave_engine.geodata.utils.io.layout import read_tree, write_tree

from .conftest import build_raster


def write_scenes(
    root: Path, tweak=lambda ds, position: ds, **write_options: Any
) -> Path:
    """Write a two-instant cube as a tree, one scene at a time.

    Writing scene by scene lets a test state each one's attrs separately, which
    is what a tree read has to reconcile.

    Args:
        root: Directory the tree is written into.
        tweak: Applied to each scene with its position, before it is written.
        **write_options: Passed to `write_tree`.

    Returns:
        The directory holding the tree.
    """
    cube = build_raster(times=2)
    for position, instant in enumerate(cube.time.values):
        write_tree(tweak(cube.sel(time=[instant]), position), root, **write_options)
    return root


@pytest.mark.parametrize(
    ("layout", "split_bands"),
    [("nested", True), ("nested", False), ("flat", True), ("flat", False)],
    ids=["nested-split", "nested-joined", "flat-split", "flat-joined"],
)
def test_every_layout_round_trips_a_cube(
    tmp_path: Path, layout: str, split_bands: bool
) -> None:
    written = build_raster(times=2)

    write_tree(written, tmp_path / "scene", layout=layout, split_bands=split_bands)
    restored = read_tree(tmp_path / "scene")

    # A tree records no variable order: one leaf per variable reads back by
    # filename, so only a joined-band layout keeps the cube's own order.
    assert set(restored.data_vars) == set(written.data_vars)
    assert restored.sizes["time"] == 2
    assert restored.gs.geobox == written.gs.geobox
    for name, variable in written.data_vars.items():
        assert np.array_equal(restored[name].values, variable.values)


def test_map_scale_reaches_every_leaf_without_losing_cube_metadata(
    tmp_path: Path,
) -> None:
    written = rebase(
        build_raster(times=2), CFVariable(units="reflectance"), target=["red", "nir"]
    )
    root = tmp_path / "scene"

    write_tree(written, root, map_scale=10_000)

    leaves = sorted(root.rglob("*.tif"))
    assert len(leaves) == 2
    for leaf in leaves:
        with rasterio.open(leaf) as src:
            tags = src.tags()
            assert float(tags["TIFFTAG_XRESOLUTION"]) == pytest.approx(10.0)
            assert float(tags["TIFFTAG_YRESOLUTION"]) == pytest.approx(10.0)
            assert tags["TIFFTAG_RESOLUTIONUNIT"].startswith("3")

    restored = read_tree(root)
    assert np.array_equal(restored.time.values, written.time.values)
    assert restored.red.attrs["units"] == "reflectance"
    assert restored.nir.attrs["units"] == "reflectance"


@pytest.mark.parametrize("layout", ["nested", "flat"])
def test_a_joined_band_layout_keeps_the_variable_order(
    tmp_path: Path, layout: str
) -> None:
    written = build_raster(times=2)

    write_tree(written, tmp_path / "scene", layout=layout, split_bands=False)

    # One leaf holds every band, so band order carries the cube's order.
    assert list(read_tree(tmp_path / "scene").data_vars) == list(written.data_vars)


def test_each_layout_places_its_leaves_where_it_says(tmp_path: Path) -> None:
    written = build_raster(times=1)
    instant = "".join(
        np.datetime_as_string(written.time.values[0], unit="s")
        .replace("-", "")
        .replace(":", "")
    )

    write_tree(written, tmp_path / "nested", layout="nested", split_bands=True)
    write_tree(written, tmp_path / "flat", layout="flat", split_bands=True)

    assert (tmp_path / "nested" / instant / "red.tif").is_file()
    assert (tmp_path / "flat" / f"{instant}_red.tif").is_file()


def test_a_caller_can_place_leaves_itself(tmp_path: Path) -> None:
    written = build_raster(times=1)

    write_tree(
        written,
        tmp_path / "scene",
        layout=lambda timestamp, variable: Path(variable) / timestamp,
        split_bands=True,
    )

    assert list((tmp_path / "scene" / "red").glob("*.tif"))


def test_an_unknown_layout_name_refuses(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        write_tree(build_raster(), tmp_path / "scene", layout="spiral")


def test_the_leaves_agree_on_a_tag_and_it_survives(tmp_path: Path) -> None:
    root = write_scenes(
        tmp_path / "scene",
        lambda ds, position: rebase(ds, GeoTIFFTags(TIFFTAG_ARTIST="geosave")),
    )

    assert read_tree(root).attrs["TIFFTAG_ARTIST"] == "geosave"


def test_a_tag_the_leaves_state_differently_drops_and_warns(tmp_path: Path) -> None:
    root = write_scenes(
        tmp_path / "scene",
        lambda ds, position: rebase(
            ds, GeoTIFFTags(TIFFTAG_ARTIST=f"artist{position}")
        ),
    )

    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_ARTIST"):
        restored = read_tree(root)

    assert "TIFFTAG_ARTIST" not in restored.attrs


def test_leaves_declaring_different_absent_pixels_join_without_one(
    tmp_path: Path,
) -> None:
    root = write_scenes(
        tmp_path / "scene",
        # One file holds one fill value, so a leaf declares it across its bands.
        lambda ds, position: rebase(
            ds, Nodata(fill_value=position), target=list(ds.data_vars)
        ),
    )

    # A join cannot pick which stored value means absent, so it declares none.
    with pytest.warns(DroppedAttrsWarning, match="red._FillValue"):
        restored = read_tree(root)

    assert "_FillValue" not in restored.red.attrs


def test_the_instant_leaves_the_tags_once_it_spans_an_axis(tmp_path: Path) -> None:
    written = build_raster(times=2)

    write_tree(written, tmp_path / "scene")
    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_DATETIME"):
        restored = read_tree(tmp_path / "scene")

    # Each leaf stamped its own instant; the cube carries them on `time` instead.
    assert "TIFFTAG_DATETIME" not in restored.attrs
    assert np.array_equal(restored.time.values, written.time.values)


def test_a_directory_holding_no_leaf_refuses(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="holds no .tif leaf"):
        read_tree(empty)


def test_a_cube_spanning_more_than_time_refuses_to_write(tmp_path: Path) -> None:
    spanning = build_raster(times=2).expand_dims({"model": ["a", "b"]})

    with pytest.raises(ValueError, match="a leaf holds one instant of one grid"):
        write_tree(spanning, tmp_path / "scene")


def test_one_leaf_reads_without_a_directory(tmp_path: Path) -> None:
    written = build_raster()

    write_tree(written, tmp_path / "dem", split_bands=True)
    restored = read_tree(tmp_path / "dem" / "red")

    assert list(restored.data_vars) == ["red"]
    assert isinstance(restored, xr.Dataset)

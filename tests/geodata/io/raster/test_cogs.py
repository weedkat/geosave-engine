"""A raster arranged as COG files named after its path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.attrs import CFVariable, rebase
from geosave_engine.geodata.io import cogs

from tests.geodata.conftest import build_raster

DAYS = ("20250601T000000", "20250602T000000")


@pytest.mark.parametrize(
    ("times", "split_bands", "expected"),
    [
        (0, False, ["scene.v2.tif"]),
        (0, True, ["scene.v2/red.tif", "scene.v2/nir.tif"]),
        (2, False, [f"scene.v2/scene.v2_{day}.tif" for day in DAYS]),
        (
            2,
            True,
            [
                f"scene.v2/scene.v2_{day}/{band}.tif"
                for day in DAYS
                for band in ("red", "nir")
            ],
        ),
    ],
    ids=["timeless-joined", "timeless-split", "dated-joined", "dated-split"],
)
def test_write_names_every_file_after_its_path(
    tmp_path: Path, times: int, split_bands: bool, expected: list[str]
) -> None:
    written = build_raster(times=times)
    unrelated = tmp_path / "scene.v2" / "unrelated.tif"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"not produced by this export")

    paths = cogs.write(written, tmp_path / "scene.v2", split_bands=split_bands)

    assert paths == tuple(tmp_path / name for name in expected)
    assert all(path.is_file() for path in paths)
    restored = read_raster(paths)
    assert restored.gs.geobox == written.gs.geobox
    for name, variable in written.data_vars.items():
        np.testing.assert_array_equal(restored[name].squeeze(), variable)


def test_one_variable_is_never_split(tmp_path: Path) -> None:
    written = build_raster(times=2)[["red"]]

    paths = cogs.write(written, tmp_path / "scene", split_bands=True)

    assert paths == tuple(tmp_path / f"scene/scene_{day}.tif" for day in DAYS)


def test_a_scalar_instant_is_named_by_the_path(tmp_path: Path) -> None:
    written = build_raster(times=1).isel(time=0)

    (path,) = cogs.write(written, tmp_path / "scene")

    assert path == tmp_path / "scene.tif"
    np.testing.assert_array_equal(read_raster(path).time, written.time)


def test_a_name_ending_in_tif_stays_a_name(tmp_path: Path) -> None:
    assert cogs.write(build_raster(), tmp_path / "image.tif") == (
        tmp_path / "image.tif.tif",
    )


def test_split_band_names_with_dots_remain_distinct(tmp_path: Path) -> None:
    written = build_raster().rename_vars({"red": "band.v1", "nir": "band.v2"})

    paths = cogs.write(written, tmp_path / "scene", split_bands=True)

    assert paths == (tmp_path / "scene/band.v1.tif", tmp_path / "scene/band.v2.tif")


def test_sub_second_instants_keep_their_fraction_in_the_name(tmp_path: Path) -> None:
    instant = np.datetime64("2025-06-01T10:30:31.123456789", "ns")
    written = build_raster(times=1).assign_coords(time=[instant])

    (path,) = cogs.write(written, tmp_path / "scene")

    assert path.name == "scene_20250601T103031_123456789.tif"
    assert read_raster([path]).time.values[0] == instant


def test_joined_bands_keep_the_variable_order(tmp_path: Path) -> None:
    written = build_raster(times=2)

    paths = cogs.write(written, tmp_path / "scene")

    assert list(read_raster(paths).data_vars) == list(written.data_vars)


def test_map_scale_reaches_every_file(tmp_path: Path) -> None:
    written = rebase(
        build_raster(times=2), CFVariable(units="reflectance"), target=["red", "nir"]
    )

    paths = cogs.write(written, tmp_path / "scene", map_scale=10_000)

    for path in paths:
        with rasterio.open(path) as src:
            assert float(src.tags()["TIFFTAG_XRESOLUTION"]) == pytest.approx(10.0)
    assert read_raster(paths).red.attrs["units"] == "reflectance"


def test_an_existing_file_refuses_without_overwrite(tmp_path: Path) -> None:
    written = build_raster(times=1)
    cogs.write(written, tmp_path / "scene")

    with pytest.raises(FileExistsError):
        cogs.write(written, tmp_path / "scene")
    assert cogs.write(written, tmp_path / "scene", overwrite=True)


def test_a_raster_spanning_more_than_time_refuses(tmp_path: Path) -> None:
    spanning = build_raster(times=2).expand_dims({"model": ["a", "b"]})

    with pytest.raises(ValueError, match="a COG holds one instant of one grid"):
        cogs.write(spanning, tmp_path / "scene")


@pytest.mark.parametrize("root", ["scene.v2", "scene.v2/"])
def test_remote_names_keep_their_scheme_and_dots(bucket: str, root: str) -> None:
    written = build_raster(times=2)

    paths = cogs.write(written, f"{bucket}/{root}", split_bands=True)

    assert paths == tuple(
        f"{bucket}/scene.v2/scene.v2_{day}/{band}.tif"
        for day in DAYS
        for band in ("red", "nir")
    )


def test_local_names_are_still_paths(tmp_path: Path) -> None:
    paths = cogs.write(build_raster(times=1), tmp_path / "scene")

    assert paths == (tmp_path / "scene" / "scene_20250601T000000.tif",)


def test_layout_names_each_file_its_scene_and_its_pixels() -> None:
    source = build_raster(times=2)

    files = cogs.layout(source, "samples/forest", split_bands=True)

    assert [file.path for file in files] == [
        "samples/forest/forest_20250601T000000/red.tif",
        "samples/forest/forest_20250601T000000/nir.tif",
        "samples/forest/forest_20250602T000000/red.tif",
        "samples/forest/forest_20250602T000000/nir.tif",
    ]
    assert [file.scene for file in files[::2]] == [
        "forest_20250601T000000",
        "forest_20250602T000000",
    ]
    assert files[0].time == pd.Timestamp("2025-06-01")
    assert files[0].raster.gs.variables == ("red",)
    assert "time" not in files[0].raster.dims


def test_layout_of_a_raster_without_a_time_dimension_is_one_scene() -> None:
    scalar = build_raster(times=1).isel(time=0)

    (file,) = cogs.layout(scalar, "samples/label")

    assert (file.scene, file.path, file.time) == ("label", "samples/label.tif", None)
    assert file.raster.gs.variables == ("red", "nir")


def test_write_writes_exactly_the_layout(tmp_path: Path) -> None:
    source = build_raster(times=2)
    root = tmp_path / "forest"

    written = cogs.write(source, root, split_bands=True)

    planned = cogs.layout(source, root, split_bands=True)
    assert [str(path) for path in written] == [file.path for file in planned]

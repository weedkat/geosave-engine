"""Several saved rasters read as one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave_engine.geodata import io, stack
from geosave_engine.geodata.io.readers import read_raster, read_stack
from geosave_engine.geodata.attrs import GeoTIFFTags, rebase
from geosave_engine.geodata.warnings import DroppedAttrsWarning
from geosave_engine.geodata.io import geotiff, zarr

from tests.geodata.conftest import build_raster


def write_scenes(folder: Path, cube: xr.Dataset, *, split: bool = False) -> list[Path]:
    """Write one COG per scene, or per scene and band, under arbitrary names."""
    paths = []
    for position in range(cube.sizes["time"]):
        scene = cube.isel(time=position)
        parts = [scene[[name]] for name in scene.data_vars] if split else [scene]
        for part in parts:
            name = f"file{len(paths)}.tif"
            paths.append(geotiff.write_cog(part, folder / name))
    return paths


@pytest.mark.parametrize("split", [False, True])
def test_several_files_read_as_one_lazy_raster(tmp_path: Path, split: bool) -> None:
    written = build_raster(times=2)
    paths = write_scenes(tmp_path, written, split=split)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        restored = read_raster(paths, chunks={})

    assert started == []
    assert restored.red.chunks is not None
    assert restored.gs.geobox == written.gs.geobox
    np.testing.assert_array_equal(restored.time, written.time)
    for name in written.data_vars:
        np.testing.assert_array_equal(restored[name], written[name])


def test_a_folder_reads_like_its_files(tmp_path: Path) -> None:
    paths = write_scenes(tmp_path, build_raster(times=2))

    with read_raster(tmp_path) as folder, read_raster(sorted(paths)) as files:
        xr.testing.assert_identical(folder, files)


def test_one_file_in_a_sequence_keeps_a_time_dimension(tmp_path: Path) -> None:
    (path,) = write_scenes(tmp_path, build_raster(times=1))

    assert read_raster([path]).sizes["time"] == 1
    assert "time" not in read_raster(path).dims


def test_files_and_a_store_read_through_one_call(tmp_path: Path) -> None:
    written = build_raster(times=3)
    store = zarr.write(written.isel(time=[0, 1]), tmp_path / "early.zarr")
    late = geotiff.write_cog(written.isel(time=2), tmp_path / "late.tif")

    restored = read_raster([store, late])

    np.testing.assert_array_equal(restored.time, written.time)
    np.testing.assert_array_equal(restored.red, written.red)


def test_a_sub_second_instant_reads_back_exactly(tmp_path: Path) -> None:
    instant = np.datetime64("2025-06-01T10:30:31.123456789", "ns")
    written = build_raster(times=1).assign_coords(time=[instant])
    (path,) = write_scenes(tmp_path, written)

    assert read_raster([path]).time.values[0] == instant


def test_sources_on_two_grids_refuse(tmp_path: Path) -> None:
    here = build_raster(times=1)
    there = here.assign_coords(x=here.x + 1000)
    paths = [
        *write_scenes(tmp_path / "here", here),
        *write_scenes(tmp_path / "there", there),
    ]

    with pytest.raises(ValueError, match="different grids"):
        read_raster(paths)


def test_two_sources_holding_one_instant_refuse(tmp_path: Path) -> None:
    written = build_raster(times=2)
    store = zarr.write(written, tmp_path / "all.zarr")
    first = geotiff.write_cog(written.isel(time=0), tmp_path / "first.tif")

    with pytest.raises(ValueError, match="one variable at one instant"):
        read_raster([first, store])


def test_no_source_refuses() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        read_raster([])


def test_a_folder_holding_no_tiff_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holds no .tif"):
        read_raster(tmp_path)


@pytest.mark.parametrize("failure", [None, "open", "combine"])
def test_a_combined_read_closes_every_file(tmp_path, monkeypatch, failure) -> None:
    for name in ("a.tif", "b.tif"):
        (tmp_path / name).touch()
    closed = []

    def read(path, **options):
        path = Path(path)
        if failure == "open" and path.stem == "b":
            raise ValueError("unreadable file")
        raster = build_raster(times=2).isel(time=0 if path.stem == "a" else 1)
        raster.set_close(lambda: closed.append(path.stem))
        return raster

    monkeypatch.setattr(io.gdal, "read", read)
    if failure == "combine":

        def combine(*args, **kwargs):
            raise ValueError("incompatible files")

        monkeypatch.setattr(xr, "combine_by_coords", combine)
    paths = [tmp_path / "a.tif", tmp_path / "b.tif"]
    if failure:
        with pytest.raises(ValueError):
            read_raster(paths)
        assert sorted(closed) == (["a"] if failure == "open" else ["a", "b"])
    else:
        raster = read_raster(paths)
        assert closed == []
        raster.close()
        assert sorted(closed) == ["a", "b"]


def test_a_tag_the_files_agree_on_survives(tmp_path: Path) -> None:
    cube = rebase(build_raster(times=2), GeoTIFFTags(TIFFTAG_ARTIST="geosave"))

    assert (
        read_raster(write_scenes(tmp_path, cube)).attrs["TIFFTAG_ARTIST"] == "geosave"
    )


def test_a_tag_the_files_state_differently_drops_and_warns(tmp_path: Path) -> None:
    cube = build_raster(times=2)
    paths = [
        geotiff.write_cog(
            rebase(
                cube.isel(time=position),
                GeoTIFFTags(TIFFTAG_ARTIST=f"artist{position}"),
            ),
            tmp_path / f"{position}.tif",
        )
        for position in range(2)
    ]

    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_ARTIST"):
        restored = read_raster(paths)

    assert "TIFFTAG_ARTIST" not in restored.attrs


def test_a_tiff_on_an_hf_bucket_opens_through_the_s3_gateway(monkeypatch) -> None:
    opened: list[str] = []

    def open_(path, *args, **kwargs):
        opened.append(str(path))
        raise OSError("no network in this test")

    monkeypatch.setattr(io.gdal.rasterio, "open", open_)

    with pytest.raises(OSError, match="no network"):
        read_raster("hf://buckets/me/samples/forest/a.tif")
    assert opened == ["s3://me/samples/forest/a.tif"]


def _flat():
    return build_raster(times=1).squeeze("time", drop=False)


def test_a_directory_of_geotiff_layers_opens_one_group_per_file(tmp_path: Path) -> None:
    flat = _flat()
    folder = tmp_path / "s1"
    stack({"label": flat[["nir"]], "optical": flat[["red"]]}).gs.to_cog(folder)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        sample = read_stack(folder)

    assert sample.gs.groups == ("label", "optical")
    assert sample.gs.rasters["optical"].gs.variables == ("red",)
    assert sample.gs.geobox == flat.gs.geobox
    assert started == []


def test_a_directory_mixes_layer_formats(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    geotiff.write_cog(_flat()[["nir"]], folder / "label.tif")
    zarr.write(build_raster(times=2)[["red"]], folder / "optical.zarr")

    sample = read_stack(folder)

    assert sample.gs.groups == ("label", "optical")
    assert sample.gs.rasters["optical"].sizes["time"] == 2


def test_a_timed_group_written_as_a_tree_is_one_group(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    stack(
        {"label": _flat()[["nir"]], "optical": build_raster(times=2)[["red"]]}
    ).gs.to_cog(folder)

    sample = read_stack(folder)

    assert sample.gs.groups == ("label", "optical")
    assert sample.gs.rasters["optical"].sizes["time"] == 2


def test_staging_folders_and_other_files_are_not_layers(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    geotiff.write_cog(_flat()[["nir"]], folder / "label.tif")
    (folder / ".optical-draft").mkdir()
    (folder / "manifest.parquet").write_bytes(b"")
    (folder / "notes.txt").write_text("field visit")

    assert read_stack(folder).gs.groups == ("label",)


def test_a_directory_holding_no_raster_is_refused(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("nothing here")

    with pytest.raises(ValueError, match="holds no raster"):
        read_stack(tmp_path)


def test_two_entries_naming_one_layer_are_refused(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    geotiff.write_cog(_flat()[["nir"]], folder / "label.tif")
    zarr.write(_flat()[["nir"]], folder / "label.zarr")

    with pytest.raises(ValueError, match="'label'"):
        read_stack(folder)


def test_a_directory_of_zarr_layers_opens_one_group_per_store(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    zarr.write(_flat()[["nir"]], folder / "label.zarr")
    zarr.write(build_raster(times=2)[["red"]], folder / "optical.zarr")

    assert read_stack(folder).gs.groups == ("label", "optical")


@pytest.mark.parametrize("fail", [False, True])
def test_directory_reader_owns_opened_layers(tmp_path, monkeypatch, fail):
    from geosave_engine.geodata.io import readers

    for name in ("a.tif", "b.tif"):
        (tmp_path / name).touch()
    closed = []

    def read(path, **options):
        if fail and path.stem == "b":
            raise ValueError("unreadable layer")
        raster = _flat().copy()
        raster.set_close(lambda: closed.append(path.stem))
        return raster

    monkeypatch.setattr(readers, "read_raster", read)
    if fail:
        with pytest.raises(ValueError, match="unreadable"):
            read_stack(tmp_path)
        assert closed == ["a"]
    else:
        sample = read_stack(tmp_path)
        assert closed == []
        sample.close()
        assert sorted(closed) == ["a", "b"]


def test_named_sources_open_one_group_each(tmp_path: Path) -> None:
    label = geotiff.write_cog(_flat()[["nir"]], tmp_path / "a.tif")
    optical = zarr.write(build_raster(times=2)[["red"]], tmp_path / "b.zarr")

    sample = read_stack({"label": label, "optical": optical})

    assert sample.gs.groups == ("label", "optical")
    assert sample.gs.rasters["optical"].sizes["time"] == 2


def test_a_group_named_by_several_files_joins_them(tmp_path: Path) -> None:
    cube = build_raster(times=2)[["red"]]
    paths = [
        geotiff.write_cog(cube.isel(time=position), tmp_path / f"{position}.tif")
        for position in range(2)
    ]

    sample = read_stack({"optical": paths})

    assert sample.gs.rasters["optical"].sizes["time"] == 2

from __future__ import annotations

from pathlib import Path

import pytest
from dask.callbacks import Callback

from geosave_engine.geodata import GeoVector, read_stack, stack
from geosave_engine.geodata.io import geotiff, zarr

from tests.geodata.conftest import build_raster


def _flat():
    return build_raster(times=1).squeeze("time", drop=False)


def test_a_directory_of_geotiff_layers_opens_one_group_per_file(tmp_path: Path) -> None:
    flat = _flat()
    folder = stack({"label": flat[["nir"]], "optical": flat[["red"]]}).gs.to_cog(
        tmp_path / "s1"
    )
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
    folder = stack(
        {"label": _flat()[["nir"]], "optical": build_raster(times=2)[["red"]]}
    ).gs.to_cog(tmp_path / "s1")

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


def test_a_read_directory_registers_one_asset_per_layer(tmp_path: Path) -> None:
    flat = _flat()
    folder = stack({"label": flat[["nir"]], "optical": flat[["red"]]}).gs.to_cog(
        tmp_path / "s1"
    )

    row = GeoVector.from_xarray(read_stack(folder), id="s1").iloc[0]

    assert row.id == "s1"
    assert list(row.assets) == ["label", "optical"]
    assert row.assets["label"]["href"] == str(folder / "label.tif")


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
    from geosave_engine.geodata import io

    for name in ("a.tif", "b.tif"):
        (tmp_path / name).touch()
    closed = []

    def read(path, **options):
        if fail and path.stem == "b":
            raise ValueError("unreadable layer")
        raster = _flat().copy()
        raster.set_close(lambda: closed.append(path.stem))
        return raster

    monkeypatch.setattr(io, "read_raster", read)
    if fail:
        with pytest.raises(ValueError, match="unreadable"):
            read_stack(tmp_path)
        assert closed == ["a"]
    else:
        sample = read_stack(tmp_path)
        assert closed == []
        sample.close()
        assert sorted(closed) == ["a", "b"]

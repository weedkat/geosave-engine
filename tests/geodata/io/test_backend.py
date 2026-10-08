"""xarray opens GeoSave rasters and stacks by engine name."""

from __future__ import annotations

from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geosave_engine.geodata import read_raster, read_stack, stack

from tests.geodata.conftest import build_raster


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


@pytest.mark.parametrize("write", ["to_zarr", "to_netcdf"])
def test_the_engine_opens_what_read_raster_opens(tmp_path: Path, write: str) -> None:
    path = getattr(build_raster(times=2, packed=True).gs, write)(
        tmp_path / ("scene.zarr" if write == "to_zarr" else "scene.nc")
    )

    with xr.open_dataset(path, engine="geosave") as opened, read_raster(path) as ours:
        assert opened.identical(ours)
        assert opened.gs.geobox == ours.gs.geobox


def test_the_engine_joins_the_files_of_one_raster(tmp_path: Path) -> None:
    source = build_raster(times=2)
    paths = source.gs.to_cog(tmp_path / "forest")

    with xr.open_dataset(list(paths), engine="geosave") as opened:
        assert dict(opened.sizes) == {"time": 2, "y": 2, "x": 2}
        np.testing.assert_array_equal(opened.red, source.red)


def test_the_engine_stays_lazy_and_forwards_reader_options(tmp_path: Path) -> None:
    path = build_raster(times=2, packed=True).gs.to_zarr(tmp_path / "scene.zarr")

    with dask.config.set(scheduler=_refuse):
        lazy = xr.open_dataset(path, engine="geosave", chunks={})
        decoded = xr.open_dataset(path, engine="geosave", mask_and_scale=True)

    assert isinstance(lazy.red.data, da.Array)
    assert lazy.red.dtype == "uint16"
    assert decoded.red.dtype.kind == "f"
    lazy.close()
    decoded.close()


def test_the_engine_drops_named_variables(tmp_path: Path) -> None:
    path = build_raster().gs.to_zarr(tmp_path / "scene.zarr")

    with xr.open_dataset(path, engine="geosave", drop_variables="nir") as opened:
        assert list(opened.data_vars) == ["red"]


def test_the_engine_opens_a_stack_as_a_tree(tmp_path: Path) -> None:
    sample = stack({"optical": build_raster(times=2), "dem": build_raster()})
    sample.gs.to_zarr(tmp_path / "s0")

    with xr.open_datatree(tmp_path / "s0", engine="geosave") as opened:
        assert opened.identical(read_stack(tmp_path / "s0"))
        assert sorted(opened.children) == ["dem", "optical"]


def test_the_engine_claims_no_file_unasked(tmp_path: Path) -> None:
    path = build_raster().gs.to_netcdf(tmp_path / "scene.nc")

    with xr.open_dataset(path) as opened:
        assert "gs" not in opened.encoding.get("engine", "")
    assert "geosave" in xr.backends.list_engines()
    assert not xr.backends.list_engines()["geosave"].guess_can_open(path)

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.delayed import Delayed

from geosave_engine.geodata.io import netcdf
from tests.geodata.conftest import build_raster


@pytest.mark.parametrize("crs", ["EPSG:32749", "EPSG:4326"])
@pytest.mark.parametrize("times", [0, 2])
def test_netcdf_round_trip_preserves_the_profile(
    tmp_path: Path, crs: str, times: int
) -> None:
    written = build_raster(crs=crs, times=times, packed=True)

    destination = netcdf.write(written, tmp_path / "scene.nc")
    restored = netcdf.read(destination)

    assert restored.gs.geobox == written.gs.geobox
    assert restored.red.dtype == np.dtype("uint16")
    assert int(restored.red.values.ravel()[0]) == 1000
    rows, _ = written.gs.grid_dims
    assert restored[rows].attrs["standard_name"] == written[rows].attrs["standard_name"]


def test_netcdf_round_trip_preserves_a_stack(
    tmp_path: Path, stack: xr.DataTree
) -> None:
    destination = netcdf.write(stack, tmp_path / "stack.nc")
    restored = netcdf.read_stack(destination)

    assert restored.gs.groups == stack.gs.groups
    assert restored.gs.geobox == stack.gs.geobox


def test_netcdf_refuses_an_existing_destination(tmp_path: Path) -> None:
    destination = netcdf.write(build_raster(), tmp_path / "scene.nc")

    with pytest.raises(FileExistsError):
        netcdf.write(build_raster(), destination)

    assert netcdf.write(build_raster(), destination, overwrite=True) == destination


def test_netcdf_refuses_a_foreign_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in"):
        netcdf.write(build_raster(), tmp_path / "scene.tif")


def test_netcdf_deferred_write_returns_a_delayed_result(tmp_path: Path) -> None:
    destination = tmp_path / "deferred.nc"

    deferred = netcdf.write(build_raster(), destination, compute=False)

    assert isinstance(deferred, Delayed)
    assert deferred.compute() == destination
    assert netcdf.read(destination).gs.geobox == build_raster().gs.geobox


def test_netcdf_groups_reopen_inherited_coordinates(tmp_path, monkeypatch):
    raster = build_raster(times=2).isel(time=0)
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(coords=raster.coords), "/image": raster}
    )
    monkeypatch.chdir(tmp_path)
    netcdf.write(tree, "scene.nc")
    opened = netcdf.read_stack("scene.nc")
    monkeypatch.chdir(tmp_path.parent)
    try:
        group = opened.gs.rasters["image"]
        assert "time" in group.coords
        assert "spatial_ref" in group.coords

    finally:
        opened.close()


def test_netcdf_uploads_but_does_not_read_remotely(bucket: str, tmp_path) -> None:
    import fsspec

    from geosave_engine.geodata import read_raster

    written = build_raster(times=2)

    saved = written.gs.to_netcdf(f"{bucket}/scene.nc", storage_options={})

    assert saved == f"{bucket}/scene.nc"
    fsspec.filesystem("memory").get(
        saved.removeprefix("memory:/"), str(tmp_path / "copy.nc")
    )
    with read_raster(tmp_path / "copy.nc") as restored:
        np.testing.assert_array_equal(restored.red, written.red)
    with pytest.raises(ValueError, match="local"):
        read_raster(saved)
    with pytest.raises(ValueError, match="compute"):
        written.gs.to_netcdf(f"{bucket}/later.nc", compute=False)

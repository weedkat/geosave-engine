from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.delayed import Delayed

from geosave_engine.geodata.utils.io import netcdf
from geosave_engine.geodata.utils.io import zarr

from .conftest import build_raster


@pytest.mark.parametrize("crs", ["EPSG:32749", "EPSG:4326"])
@pytest.mark.parametrize("times", [0, 2])
def test_zarr_round_trip_preserves_the_profile(
    tmp_path: Path, crs: str, times: int
) -> None:
    written = build_raster(crs=crs, times=times, packed=True)

    destination = zarr.write(written, tmp_path / "scene.zarr")
    restored = zarr.read(destination)

    assert restored.gs.geobox == written.gs.geobox
    assert restored.red.dtype == np.dtype("uint16")
    assert int(restored.red.values.ravel()[0]) == 1000
    rows, columns = written.gs.grid_dims
    assert restored.gs.grid_dims == (rows, columns)
    assert restored[rows].attrs["standard_name"] == written[rows].attrs["standard_name"]
    assert restored[columns].attrs["axis"] == "X"
    if times:
        assert restored.sizes["time"] == times


def test_digital_numbers_stay_stored_and_decode_on_request(tmp_path: Path) -> None:
    destination = zarr.write(build_raster(packed=True), tmp_path / "packed.zarr")

    stored = zarr.read(destination)
    decoded = zarr.read(destination, mask_and_scale=True)

    assert stored.red.dtype == np.dtype("uint16")
    assert stored.red.attrs["scale_factor"] == pytest.approx(1e-4)
    assert stored.red.attrs["_FillValue"] == 0
    assert "scale_factor" not in decoded.red.attrs
    assert "_FillValue" not in decoded.red.attrs
    assert decoded.red.encoding["scale_factor"] == pytest.approx(1e-4)
    assert float(decoded.red.values.ravel()[0]) == pytest.approx(0.1)
    assert bool(np.isnan(decoded.red.values.ravel()[-1]))


def test_grid_mapping_keeps_spatial_ref_a_coordinate(tmp_path: Path) -> None:
    written = build_raster()

    restored = zarr.read(zarr.write(written, tmp_path / "gm.zarr"))

    assert "spatial_ref" in restored.coords
    assert "spatial_ref" not in restored.data_vars


def test_variable_order_is_not_part_of_the_format(tmp_path: Path) -> None:
    written = build_raster()[["nir", "red"]]

    restored = zarr.read(zarr.write(written, tmp_path / "order.zarr"))

    # A store returns its variables in an unspecified order.
    assert tuple(written.data_vars) == ("nir", "red")
    assert set(restored.data_vars) == {"nir", "red"}


def test_destination_must_name_a_zarr_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.zarr"):
        zarr.write(build_raster(), tmp_path / "scene.nc")


def test_overwrite_guards_an_existing_store(tmp_path: Path) -> None:
    raster = build_raster()
    destination = tmp_path / "scene.zarr"
    zarr.write(raster, destination)

    with pytest.raises(FileExistsError):
        zarr.write(raster, destination)

    assert zarr.write(raster, destination, overwrite=True) == destination


def test_zarr_deferred_write_returns_a_delayed_result(tmp_path: Path) -> None:
    destination = tmp_path / "deferred.zarr"

    deferred = zarr.write(build_raster(), destination, compute=False)

    assert isinstance(deferred, Delayed)
    deferred.compute()
    assert zarr.read(destination).gs.geobox == build_raster().gs.geobox


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
    deferred.compute()
    assert netcdf.read(destination).gs.geobox == build_raster().gs.geobox

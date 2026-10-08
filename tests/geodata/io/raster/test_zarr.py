from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest
import xarray as xr
from dask.delayed import Delayed

from geosave_engine.geodata.attrs import Nodata, rebase
from geosave_engine.geodata import read_raster
from geosave_engine.geodata.io import zarr

from tests.geodata.conftest import build_raster


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
    assert restored.red.dims[-2:] == ("y", "x")
    assert restored["y"].attrs["standard_name"] == written["y"].attrs["standard_name"]
    assert restored["x"].attrs["axis"] == "X"
    if times:
        assert restored.sizes["time"] == times


def build_shuffled(names: list[str]) -> xr.Dataset:
    """Build a raster whose variable order is neither sorted nor reversed.

    Args:
        names: Variable names, in the order the raster should carry them.

    Returns:
        Raster Dataset holding one `uint16` variable per name.
    """
    base = build_raster()
    return xr.Dataset({name: base.red for name in names}, coords=base.coords)


def test_zarr_reads_its_variables_in_the_order_they_were_written(
    tmp_path: Path,
) -> None:
    # Six names scramble on every read; two happen to come back written order.
    names = ["zulu", "alpha", "mike", "bravo", "yankee", "charlie"]

    destination = zarr.write(build_shuffled(names), tmp_path / "ordered.zarr")

    # A Zarr group lists its members in no order, so every read must agree.
    for _ in range(3):
        assert list(zarr.read(destination).data_vars) == names


def test_order_metadata_is_owned_by_storage_without_changing_source_attrs(
    tmp_path: Path,
) -> None:
    names = ["nir", "red", "blue"]
    source = build_shuffled(names)
    source.attrs = {"title": "scene"}

    destination = zarr.write(source, tmp_path / "ordered.zarr")

    assert source.attrs == {"title": "scene"}
    with zarr.read(destination) as restored:
        assert list(restored.data_vars) == names
        assert restored.attrs["title"] == "scene"
        assert restored.gs.attrs.root.foreign["zarr_variable_order"] == names


def test_a_zarr_variable_written_elsewhere_trails_the_written_order(
    tmp_path: Path,
) -> None:
    names = ["zulu", "alpha", "mike", "bravo", "yankee", "charlie"]
    destination = zarr.write(build_shuffled(names), tmp_path / "ordered.zarr")
    grown = zarr.read(destination).assign(swir=lambda ds: ds.zulu)
    grown.to_zarr(tmp_path / "grown.zarr", zarr_format=3, consolidated=False)

    restored = zarr.read(tmp_path / "grown.zarr")

    assert list(restored.data_vars) == [*names, "swir"]


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


def test_external_store_without_order_metadata_reads_every_variable(
    tmp_path: Path,
) -> None:
    written = build_raster()[["nir", "red"]]
    destination = tmp_path / "external.zarr"
    written.to_zarr(destination, zarr_format=3, consolidated=False)

    with zarr.read(destination) as restored:
        assert "zarr_variable_order" not in restored.attrs
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
    assert deferred.compute() == destination
    assert zarr.read(destination).gs.geobox == build_raster().gs.geobox


def test_a_zarr_array_holds_the_fill_value_a_gdal_reader_masks_on(
    tmp_path: Path,
) -> None:
    written = rebase(build_raster(), Nodata(fill_value=255), target=["red", "nir"])

    destination = zarr.write(written, tmp_path / "scene.zarr")

    # A CF attr is not the array's own fill value, and GDAL masks on the latter.
    stored = orjson.loads((destination / "red" / "zarr.json").read_bytes())
    assert stored["fill_value"] == 255
    assert zarr.read(destination).red.attrs["_FillValue"] == 255


def test_a_variable_declaring_no_fill_leaves_zarr_its_own_default(
    tmp_path: Path,
) -> None:
    destination = zarr.write(build_raster(), tmp_path / "scene.zarr")

    stored = orjson.loads((destination / "red" / "zarr.json").read_bytes())
    assert stored["fill_value"] == 0


def test_a_raster_round_trips_through_a_remote_store(
    bucket: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    written = build_raster(times=2)
    destination = f"{bucket}/scene.zarr"

    # zarr maps `memory://` to its own store and refuses unused storage_options.
    saved = written.gs.to_zarr(destination)

    assert saved == destination
    assert list(tmp_path.iterdir()) == []
    local = written.gs.to_zarr(tmp_path / "local.zarr")
    with read_raster(saved) as restored, read_raster(local) as expected:
        xr.testing.assert_identical(restored, expected)


def test_a_remote_store_refuses_to_be_replaced_silently(bucket: str) -> None:
    written = build_raster()
    destination = f"{bucket}/scene.zarr"
    written.gs.to_zarr(destination)

    with pytest.raises(FileExistsError):
        written.gs.to_zarr(destination)
    assert written.gs.to_zarr(destination, overwrite=True) == destination


def test_a_remote_store_needs_the_zarr_suffix(bucket: str) -> None:
    with pytest.raises(ValueError, match=".zarr"):
        build_raster().gs.to_zarr(f"{bucket}/scene.store")


def test_zarr_reads_a_foreign_grid_onto_y_x(tmp_path: Path) -> None:
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    foreign = xr.Dataset(
        {"red": (("latitude", "longitude"), np.ones(grid.shape, "uint16"))},
        coords=xr_coords(grid),
    )
    foreign.red.encoding["grid_mapping"] = "spatial_ref"
    foreign.to_zarr(tmp_path / "foreign.zarr")

    restored = zarr.read(tmp_path / "foreign.zarr", chunks={})

    assert restored.red.dims == ("y", "x")
    assert restored.gs.geobox == grid
    assert restored.red.chunks is not None
    restored.close()

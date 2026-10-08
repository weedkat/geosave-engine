"""Zarr fill encoding preserves CF coordinate links in stack children."""

import numpy as np
import dask.array as da

from geosave_engine.geodata import raster, read_raster, read_stack, stack
from geosave_engine.geodata.io import zarr

from tests.geodata.conftest import build_raster


def test_relative_stores_open_after_changing_directory(tmp_path, monkeypatch):
    source = build_raster(times=2).isel(time=0)
    monkeypatch.chdir(tmp_path)
    stack({"image": source}).gs.to_zarr("scene")
    with read_stack("scene") as opened:
        monkeypatch.chdir(tmp_path.parent)
        child = opened.gs.rasters["image"]
        assert "time" in child.coords
        assert "spatial_ref" in child.coords


def test_stacked_band_metadata_survives_storage(tmp_path):
    grid = build_raster().gs.geobox
    source = raster(
        {
            "red": (
                ("y", "x"),
                da.ones((2, 2), chunks=(1, 1), dtype="uint16"),
            )
        },
        grid,
    )
    source.attrs = {"units": "root-unit", "title": "scene"}
    source.red.attrs = {"units": "band-unit"}
    stacked = source.gs.to_array()
    assert isinstance(stacked.data, da.Array)
    path = zarr.write(stacked.to_dataset(name="pixels"), tmp_path / "bands.zarr")

    with read_raster(path, chunks={}) as reopened:
        restored = reopened.pixels.gs.to_raster()
        assert restored.attrs == source.attrs
        assert restored.red.attrs == source.red.attrs
        assert restored.gs.geobox == grid
        assert isinstance(restored.red.data, da.Array)
        np.testing.assert_array_equal(restored.red, source.red)


def test_selected_stored_band_keeps_its_own_metadata_and_lazy_pixels(tmp_path):
    grid = build_raster().gs.geobox
    source = raster(
        {
            "red": (
                ("y", "x"),
                da.ones((2, 2), chunks=(1, 1), dtype="uint16"),
            ),
            "nir": (
                ("y", "x"),
                da.full((2, 2), 2, chunks=(1, 1), dtype="uint16"),
            ),
        },
        grid,
    )
    source.attrs = {"title": "scene", "nodata": -9999}
    source.red.attrs = {"nodata": 0, "long_name": "red", "units": "1"}
    source.nir.attrs = {"nodata": 1, "long_name": "nir", "units": "1"}
    path = zarr.write(
        source.gs.to_array().to_dataset(name="pixels"), tmp_path / "bands.zarr"
    )

    with read_raster(path, chunks={}) as reopened:
        restored = reopened.pixels.sel(band=["nir"]).gs.to_raster()
        assert list(restored.data_vars) == ["nir"]
        assert restored.attrs == source.attrs
        assert restored.nir.attrs == {
            "nodata": 1,
            "_FillValue": 1,
            "long_name": "nir",
            "units": "1",
        }
        assert restored.gs.geobox == grid
        assert isinstance(restored.nir.data, da.Array)
        np.testing.assert_array_equal(restored.nir, source.nir)


def test_stored_nodata_preserves_child_crs_and_restacking(tmp_path):
    grid = build_raster().gs.geobox
    source = raster(
        {"red": (("y", "x"), np.ones((2, 2), dtype="uint16"))},
        grid,
        nodata=0,
    )
    source = source.assign_coords(time=np.datetime64("2025-06-01", "ns"))
    paths = stack({"optical": source}).gs.to_zarr(tmp_path / "raw")

    with read_stack(paths) as restored:
        child = restored.gs.rasters["optical"]
        assert "spatial_ref" in child.coords
        assert "spatial_ref" not in child.data_vars
        assert child.gs.geobox == grid
        assert child.time.values == source.time.values
        rebuilt = stack({"prepared": child.gs.to_nan()})
        assert rebuilt.gs.geobox == grid


def test_rewriting_a_loaded_child_retains_scalar_time_and_grid(tmp_path):
    grid = build_raster().gs.geobox
    source = raster(
        {"red": (("y", "x"), np.ones((2, 2), dtype="uint16"))},
        grid,
        nodata=0,
    )
    source = source.assign_coords(time=np.datetime64("2025-06-01", "ns"))
    paths = stack({"optical": source}).gs.to_zarr(tmp_path / "raw")
    with read_stack(paths) as restored:
        child = restored.gs.rasters["optical"]
        copied = stack({"derived": child}).gs.to_zarr(tmp_path / "copied")
    with read_stack(copied) as restored:
        child = restored.gs.rasters["derived"]
        assert child.gs.geobox == grid
        assert child.time.values == source.time.values
        np.testing.assert_array_equal(child.red, source.red)


def test_nan_band_metadata_survives_stacked_storage(tmp_path):
    from geosave_engine.geodata.attrs import Nodata

    grid = build_raster().gs.geobox
    source = raster(
        {"red": (("y", "x"), da.ones((2, 2), chunks=(1, 1)))},
        grid,
        nodata=np.nan,
    )
    source.attrs = {"nodata": -9999}
    path = zarr.write(
        source.gs.to_array().to_dataset(name="pixels"), tmp_path / "nan-bands.zarr"
    )
    with read_raster(path, chunks={}) as reopened:
        restored = reopened.pixels.gs.to_raster()
        assert restored.gs.attrs.root.get(Nodata) is None
        assert restored.attrs["nodata"] == -9999
        assert np.isnan(restored.red.gs.attrs.root.get(Nodata).fill_value)

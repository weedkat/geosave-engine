from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pandas as pd

from geosave_engine.geodata.core.raster import raster as build_variable_raster
from geosave_engine.geodata.core.stack import stack as build_stack
from geosave_engine.geodata.transform.time import broadcast, resample
from geosave_engine.geodata.utils.io import zarr

from .conftest import build_raster


def test_stack_carries_the_shared_grid_at_its_root() -> None:
    raster = build_raster()

    built = build_stack({"optical": raster[["red"]], "dem": raster[["nir"]]})

    assert built.gs.groups == ("optical", "dem")
    assert built.gs.geobox == raster.gs.geobox
    expected = (*raster.gs.geobox.dimensions, "spatial_ref")
    assert set(expected) <= set(built.dataset.coords)
    assert not built.dataset.data_vars


def test_stack_names_are_not_restricted_to_identifiers() -> None:
    raster = build_raster()

    built = build_stack({"sentinel-2-l2a": raster[["red"]], "ndvi.v2": raster[["nir"]]})

    assert built.gs.groups == ("sentinel-2-l2a", "ndvi.v2")


def test_stack_refuses_an_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_stack({})


def test_stack_refuses_a_different_resolution() -> None:
    fine = build_raster()
    coarse = build_raster().isel(y=slice(0, 1), x=slice(0, 1))

    with pytest.raises(ValueError, match="different grid"):
        build_stack({"fine": fine, "coarse": coarse})


def test_stack_refuses_a_different_crs() -> None:
    projected = build_raster(crs="EPSG:32749")
    geographic = build_raster(crs="EPSG:4326")

    with pytest.raises(ValueError, match="different grid"):
        build_stack({"projected": projected, "geographic": geographic})


def test_stack_refuses_an_unplaced_raster() -> None:
    unplaced = xr.Dataset({"red": (("y", "x"), np.zeros((2, 2), "uint16"))})

    with pytest.raises(ValueError, match="no locatable grid"):
        build_stack({"unplaced": unplaced})


def _monthly_raster(months: int = 6) -> xr.Dataset:
    """Build a raster spanning `months` whole months, resampled to monthly."""
    box = build_raster().gs.geobox
    times = pd.date_range("2023-01-01", periods=months * 28, freq="D")
    values = np.stack([np.full(box.shape, float(i)) for i in range(len(times))])
    ds = build_variable_raster({"red": values}, box, time=times, nodata=-1.0)
    return resample(ds, "MS", "first")


def _yearly_raster(years: int = 1) -> xr.Dataset:
    """Build a raster with one value per year, resampled to yearly."""
    box = build_raster().gs.geobox
    times = pd.date_range("2023-01-01", periods=years, freq="YS")
    values = np.stack([np.full(box.shape, 100.0)] * years)
    ds = build_variable_raster({"dem": values}, box, time=times, nodata=0.0)
    return resample(ds, "YS", "first")


def test_stack_carries_the_shared_time_axis_at_its_root() -> None:
    monthly = _monthly_raster()

    built = build_stack({"optical": monthly, "other": monthly})

    assert built.dataset.sizes["time"] == monthly.sizes["time"]
    assert "time_bnds" in built.dataset.coords


def test_stack_refuses_a_mismatched_time_axis() -> None:
    monthly = _monthly_raster()
    shorter = monthly.isel(time=slice(0, 1))

    with pytest.raises(ValueError, match="different time axis"):
        build_stack({"optical": monthly, "other": shorter})


def test_stack_exempts_a_group_with_no_time_coordinate() -> None:
    monthly = _monthly_raster()
    static = build_raster()[["nir"]]

    built = build_stack({"optical": monthly, "dem": static})

    assert built.gs.groups == ("optical", "dem")
    assert built.dataset.sizes["time"] == monthly.sizes["time"]


def test_stack_accepts_a_broadcast_group_onto_the_shared_axis() -> None:
    monthly = _monthly_raster()
    yearly = _yearly_raster()
    held = broadcast(yearly, pd.DatetimeIndex(monthly.time.values))

    built = build_stack({"optical": monthly, "dem": held})

    assert built.gs.groups == ("optical", "dem")
    assert built.dataset.sizes["time"] == monthly.sizes["time"]


def test_xarray_refuses_a_misaligned_group_after_construction(
    stack: xr.DataTree,
) -> None:
    shifted = build_raster()
    shifted = shifted.assign_coords(y=shifted.y.values + 12_345.0)

    with pytest.raises(ValueError, match="not aligned"):
        stack["shifted"] = xr.DataTree(shifted)


def test_insert_adds_a_group_on_the_same_grid(stack: xr.DataTree) -> None:
    ndvi = build_raster()[["red"]].rename({"red": "ndvi"})

    grown = stack.gs.insert("ndvi", ndvi)

    assert grown.gs.groups == ("optical", "infrared", "ndvi")
    assert stack.gs.groups == ("optical", "infrared")


def test_insert_refuses_a_name_already_in_the_stack(stack: xr.DataTree) -> None:
    with pytest.raises(ValueError, match="already in this stack"):
        stack.gs.insert("optical", build_raster()[["red"]])


def test_insert_refuses_a_different_grid(stack: xr.DataTree) -> None:
    with pytest.raises(ValueError, match="stack is on"):
        stack.gs.insert("geographic", build_raster(crs="EPSG:4326")[["red"]])


def test_merge_combines_two_stacks(stack: xr.DataTree) -> None:
    other = build_stack({"labels": build_raster()[["red"]].rename({"red": "labels"})})

    merged = stack.gs.merge(other)

    assert merged.gs.groups == ("optical", "infrared", "labels")


def test_merge_refuses_colliding_group_names(stack: xr.DataTree) -> None:
    other = build_stack({"optical": build_raster()[["red"]]})

    with pytest.raises(ValueError, match="both stacks carry"):
        stack.gs.merge(other)


def test_merge_refuses_a_different_grid(stack: xr.DataTree) -> None:
    other = build_stack({"geographic": build_raster(crs="EPSG:4326")[["red"]]})

    with pytest.raises(ValueError, match="other stack is on"):
        stack.gs.merge(other)


def test_stack_round_trip_preserves_groups_and_grid(
    tmp_path: Path, stack: xr.DataTree
) -> None:
    destination = zarr.write(stack, tmp_path / "stack.zarr")

    restored = zarr.read_stack(destination)

    assert set(restored.gs.groups) == {"optical", "infrared"}
    assert restored.gs.geobox == stack.gs.geobox
    assert restored["optical"].dataset.red.dtype == np.dtype("uint16")


def test_every_group_opens_on_its_own(tmp_path: Path, stack: xr.DataTree) -> None:
    destination = zarr.write(stack, tmp_path / "solo.zarr")

    solo = zarr.read(destination, group="optical")

    assert solo.gs.geobox == stack.gs.geobox
    assert "spatial_ref" in solo.coords

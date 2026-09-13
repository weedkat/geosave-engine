from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


from geosave_engine.geodata.core.stack import stack as build_stack
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


def test_xarray_refuses_a_misaligned_group_after_construction(
    stack: xr.DataTree,
) -> None:
    shifted = build_raster()
    shifted = shifted.assign_coords(y=shifted.y.values + 12_345.0)

    with pytest.raises(ValueError, match="not aligned"):
        stack["shifted"] = xr.DataTree(shifted)


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


def test_a_stack_spans_from_its_earliest_group_to_its_latest() -> None:
    dated = build_raster(times=2)
    timeless = build_raster()

    assert build_stack({"dem": timeless}).gs.timespan is None
    assert (
        build_stack({"optical": dated, "dem": timeless}).gs.timespan
        == dated.gs.timespan
    )


def test_a_stack_anchors_on_its_shared_grid_and_joint_span() -> None:
    dated = build_raster(times=2)
    scene = build_stack({"optical": dated, "dem": build_raster()})

    assert scene.gs.anchor.geobox == scene.gs.geobox
    assert scene.gs.anchor.timespan == dated.gs.timespan
    assert build_stack({"dem": build_raster()}).gs.anchor.timespan is None

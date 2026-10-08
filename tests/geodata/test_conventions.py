"""Tests for the spatial dimension convention."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.conventions import require_yx, to_yx

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def loaded(dims: tuple[str, str], name: str = "red") -> xr.Dataset:
    """Build a lazy raster the way a foreign loader names its grid."""
    built = xr.Dataset(
        {name: (dims, da.ones(tuple(GRID.shape), chunks=(1, 3)))},
        coords=xr_coords(GRID, dims=dims),
    )
    built[name].encoding["grid_mapping"] = "spatial_ref"
    return built


@pytest.mark.parametrize("dims", [("latitude", "longitude"), ("lat", "lon")])
def test_to_yx_renames_a_foreign_grid_without_computing(dims) -> None:
    result = to_yx(loaded(dims))

    assert result.red.dims == ("y", "x")
    assert result.odc.geobox == GRID
    assert isinstance(result.red.data, da.Array)
    assert result.red.encoding["grid_mapping"] == "spatial_ref"


def test_to_yx_finds_a_pair_only_its_cf_attrs_name() -> None:
    labelled = loaded(("north", "east")).expand_dims(depth=[0.5, 1.5], axis=-1)
    labelled.north.attrs["axis"] = "Y"
    labelled.east.attrs["standard_name"] = "longitude"

    result = to_yx(labelled)

    assert result.red.dims == ("y", "x", "depth")
    assert result.odc.geobox == GRID


def test_to_yx_leaves_an_unlabelled_pair_alone() -> None:
    unlabelled = loaded(("north", "east"))

    assert to_yx(unlabelled) is unlabelled


def test_to_yx_renames_one_band() -> None:
    result = to_yx(loaded(("latitude", "longitude")).red)

    assert result.dims == ("y", "x")
    assert result.odc.geobox == GRID


def test_to_yx_returns_the_same_object_when_nothing_is_foreign() -> None:
    native = loaded(("y", "x"))
    gridless = xr.Dataset({"count": (("time",), np.arange(2))})

    assert to_yx(native) is native
    assert to_yx(gridless) is gridless


def test_to_yx_renames_each_group_of_a_tree() -> None:
    foreign = loaded(("latitude", "longitude"))
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords=xr_coords(GRID)),
            "/optical": foreign,
            "/dem": loaded(("y", "x"), name="elevation"),
        }
    )

    result = to_yx(tree)

    assert result["optical"].dataset.red.dims == ("y", "x")
    assert result["dem"].dataset.elevation.dims == ("y", "x")
    assert result.dataset.odc.geobox == GRID
    assert isinstance(result["optical"].dataset.red.data, da.Array)


def test_require_yx_names_the_rename_that_fixes_a_foreign_grid() -> None:
    foreign = loaded(("latitude", "longitude"))

    with pytest.raises(ValueError, match=r"rename\(\{'latitude': 'y'"):
        require_yx(foreign)
    with pytest.raises(ValueError, match=r"rename\(\{'latitude': 'y'"):
        require_yx(xr.DataTree.from_dict({"/optical": foreign}))


def test_require_yx_accepts_native_and_gridless_objects() -> None:
    require_yx(loaded(("y", "x")))
    require_yx(xr.DataArray(np.ones(2), dims=("time",)))

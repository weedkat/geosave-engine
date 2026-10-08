"""Tests for projecting cloud onto the ground its shadow falls on."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster
from geosave_engine.geodata.features import shadow_mask

from tests.geodata.features.conftest import sentinel

CHUNKS = {"time": 1, "y": 16, "x": 16}


def cloud(chunks: dict[str, int] | None = None) -> xr.Dataset:
    """Build a scene with a bool cloud band and sun azimuth."""
    scene = sentinel(chunks)
    return scene.assign(cloud=scene.B10 > 0.5)


@pytest.mark.parametrize("chunks", [None, CHUNKS])
def test_shadow_keeps_coordinates_grid_and_laziness(chunks) -> None:
    mask = cloud(chunks)

    result = shadow_mask(
        mask, cloud="cloud", sun_azimuth="sun_azimuth", shadow_distance_m=30
    )

    assert isinstance(result.data, da.Array) == (chunks is not None)
    assert result.dtype == bool
    assert result.name is None
    assert result.attrs == {}
    assert result.gs.geobox == mask.gs.geobox
    xr.testing.assert_identical(
        result.coords.to_dataset(), mask.cloud.coords.to_dataset()
    )


def test_shadow_is_the_same_across_chunk_edges() -> None:
    eager_mask, lazy_mask = cloud(), cloud(CHUNKS)

    eager = shadow_mask(
        eager_mask, cloud="cloud", sun_azimuth="sun_azimuth", shadow_distance_m=30
    )
    lazy = shadow_mask(
        lazy_mask, cloud="cloud", sun_azimuth="sun_azimuth", shadow_distance_m=30
    )

    xr.testing.assert_identical(lazy.compute(), eager)


def test_shadow_falls_opposite_the_sun_at_the_grid_resolution() -> None:
    grid = GeoBox.from_bbox((0, 0, 100, 100), crs="EPSG:32633", resolution=20)
    pixels = np.zeros((5, 5), dtype=bool)
    pixels[2, 2] = True
    mask = raster({"cloud": (("y", "x"), pixels)}, grid)

    # Sun due south: shadow falls north, two 20 m pixels within 40 m.
    result = shadow_mask(mask, cloud="cloud", sun_azimuth=180.0, shadow_distance_m=40)

    assert result.values[:, 2].tolist() == [True, True, False, False, False]
    assert result.values.sum() == 2


def test_each_date_uses_its_own_azimuth() -> None:
    mask = cloud()

    both = shadow_mask(
        mask, cloud="cloud", sun_azimuth="sun_azimuth", shadow_distance_m=30
    )
    first = shadow_mask(
        mask.isel(time=0), cloud="cloud", sun_azimuth=45.0, shadow_distance_m=30
    )
    second = shadow_mask(
        mask.isel(time=1), cloud="cloud", sun_azimuth=135.0, shadow_distance_m=30
    )

    np.testing.assert_array_equal(both.isel(time=0), first)
    np.testing.assert_array_equal(both.isel(time=1), second)
    assert not np.array_equal(first, second)


def test_shadow_refuses_a_mask_that_is_not_boolean() -> None:
    scene = sentinel()

    with pytest.raises(ValueError, match="boolean"):
        shadow_mask(scene, cloud="B10", sun_azimuth="sun_azimuth")


def test_shadow_refuses_a_grid_that_does_not_measure_metres() -> None:
    geographic = GeoBox((4, 4), Affine(0.01, 0, 10, 0, -0.01, 20), "EPSG:4326")
    mask = raster({"cloud": (("y", "x"), np.ones((4, 4), dtype=bool))}, geographic)
    unreferenced = xr.Dataset({"cloud": (("y", "x"), np.ones((4, 4), dtype=bool))})

    for held in (mask, unreferenced):
        with pytest.raises(ValueError, match="metres"):
            shadow_mask(held, cloud="cloud", sun_azimuth=180.0)


def test_shadow_refuses_an_azimuth_that_does_not_match_the_mask() -> None:
    scene = cloud().assign_coords(azimuth=("stations", [1.0, 2.0]))

    with pytest.raises(ValueError, match="scalar or along 'time'"):
        shadow_mask(scene, cloud="cloud", sun_azimuth="azimuth")

    timeless = scene.isel(time=0, drop=True).assign_coords(
        sun_azimuth=("time", [45.0, 135.0])
    )
    with pytest.raises(ValueError, match="varies with time"):
        shadow_mask(timeless, cloud="cloud", sun_azimuth="sun_azimuth")

"""Tests for Sentinel-2 cloud and validity masks."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import features, raster

from tests.geodata.features.conftest import sentinel

CHUNKS = {"time": 1, "y": 16, "x": 16}
MASKS = {
    "cdi_cloud_mask": {"b07": "B07", "b08": "B08", "b8a": "B8A"},
    "cirrus_cloud_mask": {"b10": "B10"},
    "scl_valid_mask": {"scl": "SCL"},
}


@pytest.mark.parametrize("mask", sorted(MASKS))
def test_a_mask_is_a_lazy_unnamed_bool_band_on_its_inputs_grid(mask: str) -> None:
    scene = sentinel(CHUNKS)
    before = scene.copy(deep=True)

    result = getattr(features, mask)(scene, **MASKS[mask])

    assert isinstance(result.data, da.Array)
    assert result.dtype == bool
    assert result.name is None
    assert result.attrs == {}
    assert result.gs.geobox == scene.gs.geobox
    xr.testing.assert_identical(
        result.coords.to_dataset(), scene.B04.coords.to_dataset()
    )
    xr.testing.assert_identical(scene, before)


def test_scl_valid_mask_keeps_only_the_classes_asked_for() -> None:
    scene = sentinel()

    assert features.scl_valid_mask(scene, scl="SCL").all()
    assert not features.scl_valid_mask(scene, scl="SCL", valid_classes=[5, 6]).any()


def test_s2cloudless_mask_schedules_without_running_the_model() -> None:
    scene = sentinel(CHUNKS)

    result = features.s2cloudless_mask(
        scene,
        b01="B01",
        b02="B02",
        b04="B04",
        b05="B05",
        b08="B08",
        b8a="B8A",
        b09="B09",
        b10="B10",
        b11="B11",
        b12="B12",
    )

    assert isinstance(result.data, da.Array)
    assert result.dtype == bool
    assert result.name is None
    assert result.gs.geobox == scene.gs.geobox


def test_s2cloudless_mask_uses_supplied_values_without_interpreting_storage_attrs() -> (
    None
):
    scene = sentinel(CHUNKS)
    before = scene.copy(deep=True)
    scene.B09.attrs["scale_factor"] = 0.0001
    tagged = scene.copy(deep=True)
    result = features.s2cloudless_mask(
        scene,
        b01="B01",
        b02="B02",
        b04="B04",
        b05="B05",
        b08="B08",
        b8a="B8A",
        b09="B09",
        b10="B10",
        b11="B11",
        b12="B12",
    )
    expected = features.s2cloudless_mask(
        before,
        b01="B01",
        b02="B02",
        b04="B04",
        b05="B05",
        b08="B08",
        b8a="B8A",
        b09="B09",
        b10="B10",
        b11="B11",
        b12="B12",
    )

    assert isinstance(result.data, da.Array)
    xr.testing.assert_identical(result.compute(), expected.compute())
    xr.testing.assert_identical(scene, tagged)


def test_s2cloudless_preserves_repeated_selectors_for_eager_and_lazy_inputs() -> None:
    scene = sentinel()
    selectors = dict.fromkeys(
        ("b01", "b02", "b04", "b05", "b08", "b8a", "b09", "b10", "b11", "b12"),
        "B08",
    )

    eager = features.s2cloudless_mask(scene, **selectors)
    lazy = features.s2cloudless_mask(scene.chunk(CHUNKS), **selectors)

    assert eager.dtype == bool
    assert lazy.dtype == bool
    assert eager.dims == scene.B08.dims
    assert isinstance(lazy.data, da.Array)
    xr.testing.assert_identical(lazy.compute(), eager)


def test_cdi_casts_unsigned_inputs_before_allocating_nan_buffers() -> None:
    scene = sentinel()[["B07", "B08", "B8A"]].astype("uint16")
    scene["B07"] = xr.ones_like(scene.B07)
    scene["B8A"] = xr.ones_like(scene.B8A)
    scene.B08.values[:] = np.random.default_rng(6).integers(0, 2, scene.B08.shape)
    expected = features.cdi_cloud_mask(
        scene.astype("float32"), b07="B07", b08="B08", b8a="B8A"
    )

    eager = features.cdi_cloud_mask(scene, b07="B07", b08="B08", b8a="B8A")
    lazy = features.cdi_cloud_mask(scene.chunk(CHUNKS), b07="B07", b08="B08", b8a="B8A")

    assert expected.any()
    xr.testing.assert_identical(eager, expected)
    xr.testing.assert_identical(lazy.compute(), expected)


def test_cirrus_uses_supplied_values_without_interpreting_storage_attrs() -> None:
    scene = sentinel()
    scene["B10"] = xr.full_like(scene.B10, 0.02).assign_attrs(_FillValue=0.02)
    before = scene.copy(deep=True)

    result = features.cirrus_cloud_mask(scene, b10="B10", reflectance_threshold=0.01)

    assert result.all()
    assert result.name is None
    assert result.attrs == {}
    xr.testing.assert_identical(
        result.coords.to_dataset(), scene.B10.coords.to_dataset()
    )
    xr.testing.assert_identical(scene, before)


@pytest.mark.parametrize("chunked", [False, True])
def test_cirrus_compares_after_float32_conversion(chunked: bool) -> None:
    scene = sentinel()
    value = np.nextafter(np.float64(0.01), np.inf)
    scene["B10"] = xr.full_like(scene.B10, value, dtype="float64")
    if chunked:
        scene = scene.chunk(CHUNKS)

    result = features.cirrus_cloud_mask(scene, b10="B10", reflectance_threshold=0.01)

    assert not result.compute().any()


def test_cdi_cloud_mask_is_the_same_across_chunk_edges() -> None:
    scene = sentinel()
    scene = scene.assign(B08=scene.B08 * 0.8, B8A=scene.B8A * 1.1)
    eager = features.cdi_cloud_mask(scene, b07="B07", b08="B08", b8a="B8A")

    chunked = scene.chunk(CHUNKS)
    lazy = features.cdi_cloud_mask(chunked, b07="B07", b08="B08", b8a="B8A")

    assert isinstance(lazy.data, da.Array)
    xr.testing.assert_identical(lazy.compute(), eager)


@pytest.mark.parametrize("missing", ["one", "all"])
def test_cdi_missing_pixels_have_local_chunk_independent_effects(missing) -> None:
    grid = GeoBox.from_bbox((0, 0, 640, 640), crs="EPSG:32633", resolution=10)
    noise = np.random.default_rng(6).uniform(0.1, 0.8, (64, 64)).astype("float32")
    source = raster(
        {
            "b07": (("y", "x"), np.full((64, 64), 0.3, dtype="float32")),
            "b08": (("y", "x"), noise),
            "b8a": (("y", "x"), np.full((64, 64), 0.4, dtype="float32")),
        },
        grid,
    )
    if missing == "one":
        source.b07.values[10, 10] = np.nan
    else:
        source = source * np.nan

    eager = features.cdi_cloud_mask(source, b07="b07", b08="b08", b8a="b8a")
    chunked = source.chunk({"y": 32, "x": 32})
    lazy = features.cdi_cloud_mask(chunked, b07="b07", b08="b08", b8a="b8a")

    xr.testing.assert_identical(lazy.compute(), eager)
    assert not eager.values[10, 10]
    if missing == "one":
        assert eager.values[50, 50]
    else:
        assert not eager.any()


def test_cirrus_threshold_uses_the_selected_variable_without_source_metadata() -> None:
    scene = sentinel().rename_vars({"B10": "cirrus"})
    scene["cirrus"] = xr.full_like(scene.cirrus, 0.02).assign_attrs(units="1")
    before = scene.copy(deep=True)

    result = features.cirrus_cloud_mask(scene, b10="cirrus", reflectance_threshold=0.03)

    assert not result.any()
    assert result.attrs == {}
    assert result.name is None
    xr.testing.assert_identical(scene, before)

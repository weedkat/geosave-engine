"""Tests for Dataset-based spectral indices."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geosave_engine.geodata import features

from tests.geodata.features.conftest import sentinel

CHUNKS = {"time": 1, "y": 16, "x": 16}
INDICES = {
    "ndvi": {"nir": "B08", "red": "B04"},
    "evi": {"nir": "B08", "red": "B04", "blue": "B02"},
    "evi2": {"nir": "B08", "red": "B04"},
    "savi": {"nir": "B08", "red": "B04"},
    "msavi2": {"nir": "B08", "red": "B04"},
    "ndre": {"nir": "B08", "red_edge": "B05"},
    "bsi": {"swir1": "B11", "red": "B04", "nir": "B08", "blue": "B02"},
    "ndbi": {"swir1": "B11", "nir": "B08"},
    "ndwi": {"green": "B04", "nir": "B08"},
    "mndwi": {"green": "B04", "swir1": "B11"},
    "ndci": {"red_edge": "B05", "red": "B04"},
    "ndmi": {"nir": "B08", "swir1": "B11"},
    "nbr": {"nir": "B08", "swir2": "B12"},
    "ndsi": {"green": "B04", "swir1": "B11"},
}


def test_ndvi_is_the_normalized_difference_of_its_two_bands() -> None:
    red = xr.DataArray([[1.0]], dims=("y", "x"))

    scene = xr.Dataset({"near": 3 * red, "visible": red})
    result = features.ndvi(scene, nir="near", red="visible")

    np.testing.assert_allclose(result.values, [[0.5]])


@pytest.mark.parametrize("chunked", [False, True])
def test_decoded_reflectance_is_reusable_across_indices(chunked: bool) -> None:
    scene = xr.Dataset(
        {
            "nir": (("y", "x"), np.array([[6000, 0]], dtype="uint16")),
            "red": (("y", "x"), np.array([[2000, 0]], dtype="uint16")),
        }
    )
    for name in ("nir", "red"):
        scene[name].attrs.update(scale_factor=0.0001, add_offset=0.1, _FillValue=0)
    if chunked:
        scene = scene.chunk({"y": 1, "x": 2})
    before = scene.copy(deep=True)
    reflectance = scene.gs.mask_and_scale()
    decoded = reflectance.copy(deep=True)

    result = reflectance.assign(
        ndvi=features.ndvi(reflectance, nir="nir", red="red"),
        evi2=features.evi2(reflectance, nir="nir", red="red"),
    )

    for name, expected in (("ndvi", 0.4), ("evi2", 0.4132231405)):
        band = result[name]
        assert band.dtype == np.float32
        assert band.attrs == {}
        if chunked:
            assert isinstance(band.data, da.Array)
        np.testing.assert_allclose(band.compute(), [[expected, np.nan]], atol=1e-7)
    xr.testing.assert_identical(scene, before)
    xr.testing.assert_identical(reflectance, decoded)


@pytest.mark.parametrize("index", sorted(INDICES))
def test_an_index_is_a_lazy_unnamed_band_on_its_inputs_grid(index: str) -> None:
    scene = sentinel(CHUNKS)
    before = scene.copy(deep=True)

    result = getattr(features, index)(scene, **INDICES[index])

    assert isinstance(result.data, da.Array)
    assert result.name is None
    assert result.attrs == {}
    assert result.dtype == np.float32
    assert result.gs.geobox == scene.gs.geobox
    xr.testing.assert_identical(
        result.coords.to_dataset(), scene.B04.coords.to_dataset()
    )
    assert np.isfinite(result.compute()).all()
    xr.testing.assert_identical(scene, before)


@pytest.mark.parametrize("dtype", ["uint16", "float32"])
@pytest.mark.parametrize("chunked", [False, True])
def test_ndvi_decodes_packing_and_masks_stored_nodata(
    dtype: str, chunked: bool
) -> None:
    scene = xr.Dataset(
        {
            "nir": (("y", "x"), np.array([[6000, 0]], dtype=dtype)),
            "red": (("y", "x"), np.array([[2000, 0]], dtype=dtype)),
        }
    )
    for name in ("nir", "red"):
        scene[name].attrs.update(scale_factor=0.0001, add_offset=0.1, _FillValue=0)
    if chunked:
        scene = scene.chunk({"y": 1, "x": 2})
    before = scene.copy(deep=True)

    result = features.ndvi(scene, nir="nir", red="red")

    np.testing.assert_allclose(result.compute(), [[0.4, np.nan]], atol=1e-7)
    assert result.name is None
    assert result.attrs == {}
    assert result.dtype == np.float32
    if chunked:
        assert isinstance(result.data, da.Array)
    xr.testing.assert_identical(scene, before)


def test_ndvi_casts_before_unsigned_subtraction() -> None:
    scene = xr.Dataset(
        {
            "nir": (("y", "x"), np.array([[0]], dtype="uint16")),
            "red": (("y", "x"), np.array([[1]], dtype="uint16")),
        }
    )

    result = features.ndvi(scene, nir="nir", red="red")

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[-1]])


def test_an_index_reports_a_missing_selected_band() -> None:
    with pytest.raises(KeyError, match="absent"):
        features.ndvi(sentinel(), nir="absent", red="B04")


def test_a_dead_pixel_is_nan_rather_than_infinite() -> None:
    dark = xr.DataArray([[0.0]], dims=("y", "x"))

    scene = xr.Dataset({"nir": dark, "red": dark})
    assert np.isnan(features.ndvi(scene, nir="nir", red="red").values).all()


@pytest.mark.parametrize("dtype", [None, "uint16", "float32"])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    ("index", "want"),
    [
        ("ndvi", 0.5),
        ("evi", 0.4878048780),
        ("evi2", 0.4807692308),
        ("savi", 0.4615384615),
        ("msavi2", 0.4596875763),
        ("ndre", 0.2),
        ("bsi", 0.0),
        ("ndbi", -0.0909090909),
        ("ndwi", -0.3333333333),
        ("mndwi", -0.25),
        ("ndci", 0.3333333333),
        ("ndmi", 0.0909090909),
        ("nbr", -0.0769230769),
        ("ndsi", -0.25),
    ],
)
def test_an_index_uses_the_selected_bands(
    index: str, want: float, dtype: str | None, chunked: bool
) -> None:
    values = {"B08": 0.6, "B04": 0.2, "B02": 0.1, "B05": 0.4, "B11": 0.5, "B12": 0.7}
    if index in {"ndwi", "mndwi", "ndsi"}:
        values["B04"] = 0.3
    scene = xr.Dataset(
        {name: (("y", "x"), [[value, np.nan]]) for name, value in values.items()}
    )
    if dtype is not None:
        scene = xr.Dataset(
            {
                name: (
                    ("y", "x"),
                    np.array([[round((value - 0.1) / 0.0001), 65535]], dtype=dtype),
                )
                for name, value in values.items()
            }
        )
        for name in values:
            scene[name].attrs.update(
                scale_factor=0.0001, add_offset=0.1, _FillValue=65535
            )
    if chunked:
        scene = scene.chunk({"y": 1, "x": 1})
    scene = scene.rename_vars({name: f"selected_{name}" for name in values})
    before = scene.copy(deep=True)
    selectors = {key: f"selected_{name}" for key, name in INDICES[index].items()}

    result = getattr(features, index)(scene, **selectors)

    np.testing.assert_allclose(result.compute(), [[want, np.nan]], atol=1e-7)
    assert result.dtype == np.float32
    assert result.name is None
    assert result.attrs == {}
    if chunked:
        assert isinstance(result.data, da.Array)
    xr.testing.assert_identical(scene, before)

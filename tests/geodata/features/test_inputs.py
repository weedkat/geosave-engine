import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster
from geosave_engine.geodata.features import (
    cdi_cloud_mask,
    ndvi,
    s2cloudless_mask,
    shadow_mask,
)
from geosave_engine.geodata.features.cloud_mask import S2C_BAND_ORDER


def reflectance() -> xr.Dataset:
    grid = GeoBox.from_bbox((0, 0, 320, 320), crs="EPSG:32633", resolution=10)
    pixels = np.random.default_rng(5).uniform(0.1, 0.8, (2, 32, 32)).astype("float32")
    source = raster(
        {"red": pixels},
        grid,
        time=np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
    )
    source = source.assign_coords(sun_azimuth=("time", [40.0, 45.0]), site="plot-1")
    source.sun_azimuth.attrs["units"] = "degree"
    return source


@pytest.mark.parametrize("lazy", [False, True])
def test_shadow_preserves_auxiliary_coordinates_and_lazy_pixels(lazy):
    source = reflectance().assign(cloud=lambda data: data.red > 0.5)
    if lazy:
        source = source.chunk({"time": 1, "y": 16, "x": 16})
    result = shadow_mask(
        source,
        name="shadow",
        cloud_mask="cloud",
        sun_azimuth="sun_azimuth",
        shadow_distance_m=30,
    )
    assert isinstance(result.shadow.data, da.Array) == lazy
    xr.testing.assert_identical(result.coords.to_dataset(), source.coords.to_dataset())
    assert result.gs.geobox == source.gs.geobox
    assert "scale_factor" not in result.shadow.attrs


def test_chunked_spatial_kernels_match_eager_across_chunk_edges():
    source = reflectance().assign(
        b07=lambda data: data.red,
        b08=lambda data: data.red * 0.8,
        b8a=lambda data: data.red * 1.1,
        cloud=lambda data: data.red > 0.5,
    )
    eager = cdi_cloud_mask(source, name="clouds", b07="b07", b08="b08", b8a="b8a")
    chunked = source.chunk({"time": 1, "y": 16, "x": 16})
    lazy = cdi_cloud_mask(chunked, name="clouds", b07="b07", b08="b08", b8a="b8a")
    assert isinstance(lazy.clouds.data, da.Array)
    xr.testing.assert_identical(lazy.compute(), eager)

    eager_shadow = shadow_mask(
        source,
        name="shadow",
        cloud_mask="cloud",
        sun_azimuth="sun_azimuth",
        shadow_distance_m=30,
    )
    lazy_shadow = shadow_mask(
        chunked,
        name="shadow",
        cloud_mask="cloud",
        sun_azimuth="sun_azimuth",
        shadow_distance_m=30,
    )
    xr.testing.assert_identical(lazy_shadow.compute(), eager_shadow)


@pytest.mark.parametrize("metadata", [{"scale_factor": 0.0001}, {"nodata": 0}])
def test_indices_require_explicit_preparation(metadata):
    source = reflectance().assign(nir=lambda data: data.red)
    source.red.attrs.update(metadata)
    with pytest.raises(ValueError, match="to_nan.*unpack"):
        ndvi(source, name="ndvi", red="red", nir="nir")


def test_s2cloudless_requires_explicit_preparation_before_scheduling():
    source = reflectance()
    packed = source.red.assign_attrs(scale_factor=0.0001)
    source = source.assign({name: packed for name in S2C_BAND_ORDER}).chunk(
        {"y": 16, "x": 16}
    )
    with pytest.raises(ValueError, match="to_nan.*unpack"):
        s2cloudless_mask(
            source,
            name="cloud",
            **{name: name for name in S2C_BAND_ORDER},
        )


def test_prepared_index_preserves_lazy_grid_and_values():
    source = (
        reflectance().assign(nir=lambda data: 3 * data.red).chunk({"y": 16, "x": 16})
    )
    result = ndvi(source, name="ndvi", red="red", nir="nir")
    assert isinstance(result.ndvi.data, da.Array)
    assert result.gs.geobox == source.gs.geobox
    xr.testing.assert_identical(result.coords.to_dataset(), source.coords.to_dataset())
    np.testing.assert_allclose(result.ndvi, 0.5, atol=1e-7)


@pytest.mark.parametrize("missing", ["one", "all"])
def test_cdi_missing_pixels_have_local_chunk_independent_effects(missing):
    grid = GeoBox.from_bbox((0, 0, 640, 640), crs="EPSG:32633", resolution=10)
    source = raster(
        {
            "b07": np.full((64, 64), 0.3, dtype="float32"),
            "b08": np.random.default_rng(6)
            .uniform(0.1, 0.8, (64, 64))
            .astype("float32"),
            "b8a": np.full((64, 64), 0.4, dtype="float32"),
        },
        grid,
    )
    if missing == "one":
        source.b07.values[10, 10] = np.nan
    else:
        source = source * np.nan
    eager = cdi_cloud_mask(source, name="cloud", b07="b07", b08="b08", b8a="b8a")
    lazy = cdi_cloud_mask(
        source.chunk({"y": 32, "x": 32}),
        name="cloud",
        b07="b07",
        b08="b08",
        b8a="b8a",
    )
    xr.testing.assert_identical(lazy.compute(), eager)
    assert not eager.cloud.values[10, 10]
    if missing == "one":
        assert eager.cloud.values[50, 50]
    else:
        assert not eager.cloud.any()


def test_shadow_retains_cf_mapping_when_multiple_crs_coordinates_exist():
    source = reflectance().isel(time=0)
    cloud = source.red > 0.5
    other = cloud.odc.assign_crs("EPSG:32634").spatial_ref
    native = xr.DataArray(
        cloud.values,
        dims=cloud.dims,
        coords={
            "other_ref": other,
            "x": cloud.x,
            "y": cloud.y,
            "spatial_ref": cloud.spatial_ref,
        },
        attrs={"grid_mapping": "spatial_ref"},
    )
    raster_with_context = native.to_dataset(name="cloud").assign_coords(
        sun_azimuth=90.0
    )
    result = shadow_mask(
        raster_with_context,
        name="shadow",
        cloud_mask="cloud",
        sun_azimuth="sun_azimuth",
        shadow_distance_m=10,
    )
    assert result.shadow.odc.crs == native.odc.crs
    assert result.shadow.odc.crs.epsg == 32633

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster
from geosave_engine.geodata import features


def test_index_returns_named_lazy_raster_with_source_context() -> None:
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    source = raster(
        {
            "B04": da.ones((4, 4), chunks=(2, 2)),
            "B08": da.full((4, 4), 3.0, chunks=(2, 2)),
        },
        grid,
    ).assign_attrs(collection="sentinel-2-l2a")

    result = features.ndvi(source, name="vegetation", nir="B08", red="B04")

    assert isinstance(result, xr.Dataset)
    assert list(result.data_vars) == ["vegetation"]
    assert isinstance(result.vegetation.data, da.Array)
    assert result.gs.geobox == source.gs.geobox
    assert result.attrs == source.attrs
    np.testing.assert_allclose(result.vegetation.compute(), 0.5)


@pytest.mark.parametrize(
    ("feature", "selectors"),
    [
        ("evi", {"nir": "nir", "red": "red", "blue": "blue"}),
        ("evi2", {"nir": "nir", "red": "red"}),
        ("savi", {"nir": "nir", "red": "red"}),
        ("msavi2", {"nir": "nir", "red": "red"}),
        ("ndre", {"nir": "nir", "red_edge": "red_edge"}),
        (
            "bsi",
            {"swir1": "swir1", "red": "red", "nir": "nir", "blue": "blue"},
        ),
        ("ndbi", {"swir1": "swir1", "nir": "nir"}),
        ("ndwi", {"green": "green", "nir": "nir"}),
        ("mndwi", {"green": "green", "swir1": "swir1"}),
        ("ndci", {"red_edge": "red_edge", "red": "red"}),
        ("ndmi", {"nir": "nir", "swir1": "swir1"}),
        ("nbr", {"nir": "nir", "swir2": "swir2"}),
        ("ndsi", {"green": "green", "swir1": "swir1"}),
    ],
)
def test_indices_share_the_named_raster_contract(feature, selectors) -> None:
    values = {
        "nir": 0.6,
        "red": 0.2,
        "blue": 0.1,
        "green": 0.3,
        "red_edge": 0.45,
        "swir1": 0.4,
        "swir2": 0.5,
    }
    source = xr.Dataset(
        {
            variable: (("y", "x"), da.full((2, 2), value, chunks=(1, 1)))
            for variable, value in values.items()
        },
        attrs={"collection": "prepared"},
    )

    result = getattr(features, feature)(source, name=f"derived_{feature}", **selectors)

    assert list(result.data_vars) == [f"derived_{feature}"]
    assert isinstance(result[f"derived_{feature}"].data, da.Array)
    assert result.attrs == source.attrs
    assert np.isfinite(result[f"derived_{feature}"].compute()).all()


def _sentinel_mask_source() -> xr.Dataset:
    grid = GeoBox.from_bbox((0, 0, 320, 320), crs="EPSG:32633", resolution=10)
    values = np.random.default_rng(4).uniform(0.1, 0.8, (2, 32, 32))
    source = raster(
        {
            band: values + offset
            for offset, band in enumerate(
                [
                    "B01",
                    "B02",
                    "B04",
                    "B05",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B10",
                    "B11",
                    "B12",
                ]
            )
        }
        | {"SCL": np.full_like(values, 4, dtype="uint8")},
        grid,
        time=np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
    )
    return source.assign_coords(sun_azimuth=("time", [45.0, 135.0])).chunk(
        {"time": 1, "y": 16, "x": 16}
    )


@pytest.mark.parametrize(
    ("feature", "selectors"),
    [
        ("cdi_cloud_mask", {"b07": "B07", "b08": "B08", "b8a": "B8A"}),
        ("cirrus_cloud_mask", {"b10": "B10"}),
        ("scl_valid_mask", {"scl": "SCL"}),
    ],
)
def test_masks_share_the_named_raster_contract(feature, selectors) -> None:
    source = _sentinel_mask_source()

    result = getattr(features, feature)(source, name=f"derived_{feature}", **selectors)

    assert list(result.data_vars) == [f"derived_{feature}"]
    assert result[f"derived_{feature}"].dtype == bool
    assert isinstance(result[f"derived_{feature}"].data, da.Array)
    assert result.gs.geobox == source.gs.geobox
    assert result.attrs == source.attrs


def test_s2cloudless_mask_builds_a_lazy_named_raster_without_running_model() -> None:
    source = _sentinel_mask_source()

    result = features.s2cloudless_mask(
        source,
        name="cloud",
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

    assert list(result.data_vars) == ["cloud"]
    assert result.cloud.dtype == bool
    assert isinstance(result.cloud.data, da.Array)
    assert result.gs.geobox == source.gs.geobox


def test_shadow_mask_reads_time_varying_sun_azimuth_from_raster_coordinate() -> None:
    source = _sentinel_mask_source().assign(
        cloud=lambda data: data.B10 > data.B10.mean(("y", "x"))
    )

    result = features.shadow_mask(
        source,
        name="shadow",
        cloud_mask="cloud",
        sun_azimuth="sun_azimuth",
        shadow_distance_m=30,
    )

    assert list(result.data_vars) == ["shadow"]
    assert result.shadow.dtype == bool
    assert isinstance(result.shadow.data, da.Array)
    assert result.gs.geobox == source.gs.geobox
    assert "sun_azimuth" in result.coords

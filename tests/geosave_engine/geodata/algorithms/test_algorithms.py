"""Unit tests for geodata.algorithms — no network required."""
from datetime import datetime

import numpy as np
import pystac
import xarray as xr

from geosave_engine.geodata.algorithms import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_ndvi,
    compute_s2c_mask,
)
from geosave_engine.geodata.pipeline import Derived, SourceData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 32, 32
RNG = np.random.default_rng(42)


def _band(fill: float = 0.3) -> np.ndarray:
    return np.full((H, W), fill, dtype=np.float32)


def _rand_band() -> np.ndarray:
    return RNG.random((H, W)).astype(np.float32)


def _make_source_data(**bands: np.ndarray) -> SourceData:
    ds = xr.Dataset({k: xr.DataArray(v[np.newaxis], dims=["time", "y", "x"]) for k, v in bands.items()})
    item = pystac.Item("test", None, None, datetime(2019, 2, 23), {"sun_azimuth": 145.0})
    return SourceData(ds=ds, items=[item])


# ---------------------------------------------------------------------------
# compute_ndvi
# ---------------------------------------------------------------------------

class TestComputeNDVI:
    def test_output_shape(self):
        result = compute_ndvi(_band(0.4), _band(0.2))
        assert result.shape == (H, W)

    def test_output_dtype_float32(self):
        result = compute_ndvi(_band(0.4), _band(0.2))
        assert result.dtype == np.float32

    def test_pure_vegetation(self):
        # nir >> red → NDVI close to 1
        result = compute_ndvi(_band(0.8), _band(0.1))
        assert float(result.mean()) > 0.7

    def test_bare_soil(self):
        # nir ≈ red → NDVI near 0
        result = compute_ndvi(_band(0.3), _band(0.3))
        assert float(np.abs(result).mean()) < 0.01

    def test_water(self):
        # red > nir → NDVI negative
        result = compute_ndvi(_band(0.05), _band(0.2))
        assert float(result.mean()) < 0

    def test_no_division_by_zero(self):
        result = compute_ndvi(np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32))
        assert np.isfinite(result).all()

    def test_range_clipped_to_minus1_plus1(self):
        result = compute_ndvi(_rand_band(), _rand_band())
        assert float(result.min()) >= -1.0
        assert float(result.max()) <= 1.0

    def test_pipeline_integration(self):
        sd = _make_source_data(B08=_band(0.8), B04=_band(0.1))

        def fn(cache: dict) -> xr.DataArray:
            ds = cache["s2"].ds.median("time")
            return xr.DataArray(compute_ndvi(ds["B08"].values, ds["B04"].values), dims=["y", "x"])

        d = Derived.from_cache(name="ndvi", compute_fn=fn, sources="s2")
        result = d.compute({"s2": sd})
        assert isinstance(result, xr.DataArray)
        assert result.shape == (H, W)


# ---------------------------------------------------------------------------
# compute_b10_mask
# ---------------------------------------------------------------------------

class TestComputeB10Mask:
    def test_output_shape(self):
        result = compute_b10_mask(_band(0.05))
        assert result.shape == (H, W)

    def test_output_dtype_bool(self):
        result = compute_b10_mask(_band(0.05))
        assert result.dtype == bool

    def test_above_threshold_flagged(self):
        result = compute_b10_mask(_band(0.5), b10_threshold=0.01)
        assert result.all()

    def test_below_threshold_not_flagged(self):
        result = compute_b10_mask(_band(0.0), b10_threshold=0.01)
        assert not result.any()

    def test_mixed_values(self):
        b10 = np.zeros((H, W), dtype=np.float32)
        b10[0, 0] = 0.5
        result = compute_b10_mask(b10, b10_threshold=0.1)
        assert result[0, 0]
        assert not result[0, 1]


# ---------------------------------------------------------------------------
# compute_cdi_mask
# ---------------------------------------------------------------------------

class TestComputeCDIMask:
    def test_output_shape(self):
        result = compute_cdi_mask(_rand_band(), _rand_band(), _rand_band())
        assert result.shape == (H, W)

    def test_output_dtype_bool(self):
        result = compute_cdi_mask(_rand_band(), _rand_band(), _rand_band())
        assert result.dtype == bool

    def test_uniform_bands_no_variance(self):
        # Uniform arrays → local variance ≈ 0 → CDI ≈ 0 → not below default -0.5
        result = compute_cdi_mask(_band(0.3), _band(0.3), _band(0.3))
        assert not result.any()

    def test_threshold_controls_output(self):
        b07, b08, b8a = _rand_band(), _rand_band(), _rand_band()
        strict = compute_cdi_mask(b07, b08, b8a, cdi_threshold=0.5)
        loose = compute_cdi_mask(b07, b08, b8a, cdi_threshold=-0.99)
        assert strict.sum() >= loose.sum()

    def test_no_nan_inf(self):
        result = compute_cdi_mask(
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
        )
        assert np.isfinite(result.astype(np.float32)).all()


# ---------------------------------------------------------------------------
# compute_s2c_mask
# ---------------------------------------------------------------------------

class TestComputeS2CMask:
    def _all_bands(self, fill: float = 0.1) -> dict:
        return {k: _band(fill) for k in ["b01", "b02", "b04", "b05", "b08", "b8a", "b09", "b10", "b11", "b12"]}

    def test_output_shape(self):
        result = compute_s2c_mask(**self._all_bands())
        assert result.shape == (H, W)

    def test_output_dtype_bool(self):
        result = compute_s2c_mask(**self._all_bands())
        assert result.dtype == bool

    def test_high_reflectance_flags_cloud(self):
        # Very high uniform reflectance → cloud detected
        result = compute_s2c_mask(**self._all_bands(fill=0.9))
        assert result.any()

    def test_low_reflectance_no_cloud(self):
        # Near-zero reflectance → no cloud
        result = compute_s2c_mask(**self._all_bands(fill=0.0))
        assert not result.any()

    def test_threshold_controls_sensitivity(self):
        bands = self._all_bands(fill=0.5)
        strict = compute_s2c_mask(**bands, cloud_threshold=0.9)
        loose = compute_s2c_mask(**bands, cloud_threshold=0.1)
        assert loose.sum() >= strict.sum()


# ---------------------------------------------------------------------------
# build_shadow_mask
# ---------------------------------------------------------------------------

class TestBuildShadowMask:
    def test_output_shape(self):
        cloud = np.zeros((H, W), dtype=bool)
        result = build_shadow_mask(cloud, sun_azimuth_deg=180.0)
        assert result.shape == (H, W)

    def test_output_dtype_bool(self):
        cloud = np.zeros((H, W), dtype=bool)
        result = build_shadow_mask(cloud, sun_azimuth_deg=180.0)
        assert result.dtype == bool

    def test_no_cloud_no_shadow(self):
        cloud = np.zeros((H, W), dtype=bool)
        result = build_shadow_mask(cloud, sun_azimuth_deg=180.0)
        assert not result.any()

    def test_shadow_projected_from_cloud(self):
        cloud = np.zeros((H, W), dtype=bool)
        cloud[H // 2, W // 2] = True
        result = build_shadow_mask(cloud, sun_azimuth_deg=180.0, shadow_distance_m=50, resolution=10)
        # Sun from south (180°) → shadow projects northward → rows < H//2 should be hit
        assert result.any()
        assert not result[H // 2, W // 2]  # cloud pixel itself not in shadow

    def test_shadow_does_not_wrap(self):
        # Cloud at top row; shadow should not appear at bottom via wrap-around
        cloud = np.zeros((H, W), dtype=bool)
        cloud[0, W // 2] = True
        result = build_shadow_mask(cloud, sun_azimuth_deg=180.0, shadow_distance_m=200, resolution=10)
        assert not result[H - 1, W // 2]

    def test_shadow_distance_zero_no_shadow(self):
        cloud = np.ones((H, W), dtype=bool)
        result = build_shadow_mask(cloud, sun_azimuth_deg=45.0, shadow_distance_m=0)
        assert not result.any()

    def test_composable_with_cloud_mask(self):
        # Typical recipe: combined = cloud | shadow
        cloud = np.zeros((H, W), dtype=bool)
        cloud[H // 2, W // 2] = True
        shadow = build_shadow_mask(cloud, sun_azimuth_deg=180.0, shadow_distance_m=50)
        combined = cloud | shadow
        assert combined.dtype == bool
        assert combined[H // 2, W // 2]

    def test_pipeline_integration(self):
        sd = _make_source_data(
            B01=_band(0.0), B02=_band(0.0), B04=_band(0.0), B05=_band(0.0),
            B08=_band(0.0), B8A=_band(0.0), B09=_band(0.0), B10=_band(0.0),
            B11=_band(0.0), B12=_band(0.0), B07=_band(0.3),
        )

        def fn(cache: dict) -> xr.DataArray:
            source = cache["s2"]
            ds = source.ds.median("time")
            sun_az = source.items[0].properties["sun_azimuth"]
            cloud = compute_s2c_mask(
                b01=ds["B01"].values, b02=ds["B02"].values, b04=ds["B04"].values,
                b05=ds["B05"].values, b08=ds["B08"].values, b8a=ds["B8A"].values,
                b09=ds["B09"].values, b10=ds["B10"].values, b11=ds["B11"].values,
                b12=ds["B12"].values,
            )
            shadow = build_shadow_mask(cloud, sun_azimuth_deg=sun_az)
            b10 = compute_b10_mask(ds["B10"].values)
            mask = cloud | shadow | b10
            return xr.DataArray(mask, dims=["y", "x"])

        d = Derived.from_cache(name="mask", compute_fn=fn, sources="s2")
        result = d.compute({"s2": sd})
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool

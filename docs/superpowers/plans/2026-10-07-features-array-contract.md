# Features Take Arrays, Return Arrays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every function in `geodata/features` takes the bands it reads as DataArrays and returns one unnamed DataArray, so a YAML `!ref` passes pixels and coordinates directly.

**Architecture:** The string selectors, the `name=` parameter and `feature_raster` served a config that could only name things. `CallSpec` now resolves `!ref raster.B08` and `!ref raster.sun_azimuth` to the objects themselves, and `call: !ref image.assign` puts a result back on a raster under the step's own name. What `feature_raster` still did that matters (stop a source band's name and attrs leaking onto the result) moves to one line on the way in.

**Tech Stack:** xarray 2026.4, dask, odc-geo, pytest.

**Spec:** none; settled in conversation on 2026-10-07 and restated under Design.

## Design

```python
# before
shadow = shadow_mask(scene, name="shadow", cloud_mask="cloud", sun_azimuth="sun_azimuth",
                     resolution=10, shadow_distance_m=500)["shadow"]
# after
shadow = shadow_mask(scene.cloud, scene.sun_azimuth, shadow_distance_m=500)
```

```yaml
rasters:
  sentinel_2_l1c:
    variables: [B10]
    coordinates: [x, y, sun_azimuth]        # refuses a raster ingested without it
    stac:
      load:
        with_properties:
          - {key: "view:sun_azimuth", name: sun_azimuth}   # a colon cannot sit in a !ref path
preprocessing:
  cloud:
    call: geosave_engine.geodata.features.cirrus_cloud_mask
    kwargs: {b10: !ref sentinel_2_l1c.B10}
  shadow:
    call: geosave_engine.geodata.features.shadow_mask
    kwargs: {cloud: !ref cloud, sun_azimuth: !ref sentinel_2_l1c.sun_azimuth}
  image:
    call: !ref sentinel_2_l1c.assign
    kwargs: {cloud: !ref cloud, shadow: !ref shadow}
```

Smoke-tested on 2026-10-07:

| Claim | Result |
| --- | --- |
| `!ref s2.sun_azimuth` through `CallSpec.invoke` | the function receives the time-indexed coordinate as a DataArray |
| `call: !ref image.assign`, `kwargs: {ndvi: !ref ndvi}` | Dataset gains `ndvi`; root attrs and geobox kept |
| Derived band with no `grid_mapping` encoding | geobox resolves; Zarr, NetCDF and COG round trips keep it |
| xarray 2026.4 arithmetic and comparisons | **keep** the left operand's attrs, and its name when unary (`red > 0.1` is still named `B04` with `long_name: red`) |
| `band.drop_attrs()` (deep) | also strips coordinate attrs and loses the geobox; `drop_attrs(deep=False)` keeps both |
| `Ref("s2.view:sun_azimuth")` | `ValueError`: the property must be loaded under a plain `name` |
| `band.gs.crs.units` | `("metre", "metre")` on UTM, `("degrees_north", "degrees_east")` on EPSG:4326 |

## Global Constraints

- Features take `xr.DataArray` inputs and return one `xr.DataArray` with `name is None` and `attrs == {}`.
- Laziness is preserved; each feature's test asserts a dask input gives a dask output without computing.
- No band is aligned implicitly: bands on different grids raise.
- No compatibility aliases for the old signatures (CLAUDE.md).
- `SPATIAL_DIMENSIONS` and `TIME_COORDINATE` come from `geodata/conventions.py`; no literals in `src/`.
- Helpers stay in `geodata/features` (`_raster.py`, `_overlap.py`); only `features` uses them.
- The working tree carries unrelated uncommitted changes. Do not commit, stash, checkout or restore.
- `workspace/modules/data_pipeline_*.py` already import names that no longer exist; it is a generated consumer workspace and out of scope.
- Baseline: `uv run pytest tests/geodata tests/ml tests/model -q` gives 1519 passed.

## Scenario Cut

`test_shadow_retains_cf_mapping_when_multiple_crs_coordinates_exist` is removed. `feature_raster` copied `grid_mapping` so that an array carrying two CRS coordinates kept naming the right one. No GeoSave constructor, reader or transform produces such an array, and keeping it would mean a finisher at every return, which is what this plan removes. If you want it kept, say so before execution: it costs a private `_on_grid(result, reference)` called at all 19 returns.

## Test Files

Tests move to mirror the source tree. `test_inputs.py` and `test_raster_interface.py` (both modified in the working tree) are **deleted**; every case in them is ported below except the one under Scenario Cut.

| Source | Test |
| --- | --- |
| `features/_raster.py`, `spectral_indices.py` | `tests/geodata/features/test_spectral_indices.py` (rewritten) |
| `features/cloud_mask.py` | `tests/geodata/features/test_cloud_mask.py` (new) |
| `features/shadow_mask.py`, `_overlap.py` | `tests/geodata/features/test_shadow_mask.py` (new) |
| shared builder | `tests/geodata/features/conftest.py` (new) |

## Review Focus

1. **Bands from two rasters on different grids** (`!ref s2_10m.B08` with `!ref s2_20m.B11`): xarray arithmetic inner-joins and would silently return a sliver. `prepared_reflectance` aligns `join="exact"` and raises. Tested in Task 1.
2. **Attrs leaking from a source band** (`long_name: red` on an NDVI, `flag_values` on a validity mask): every test asserts `attrs == {}` on inputs that carry attrs.
3. **Shadow on a geographic or unreferenced mask**: a distance in metres spans no known pixel count. Raises; tested in Task 2.
4. **A time-varying azimuth whose length differs from the mask's**: raises instead of an `IndexError` mid-loop. Tested in Task 2.
5. **Non-square pixels** in `shadow_mask`: steps are counted along `x` only, as before. Not tested; stated in the docstring.

---

### Task 1: Reflectance features on arrays

**Files:**
- Modify: `src/geosave_engine/geodata/features/_raster.py`, `spectral_indices.py`, `cloud_mask.py`
- Create: `tests/geodata/features/conftest.py`, `tests/geodata/features/test_cloud_mask.py`
- Rewrite: `tests/geodata/features/test_spectral_indices.py`
- Modify: `tests/geodata/stac/test_dataflow.py:15-27`

**Interfaces:**
- Produces: `bare(band: xr.DataArray) -> xr.DataArray`; `prepared_reflectance(*bands: xr.DataArray) -> tuple[xr.DataArray, ...]`; the 14 indices as `index(<bands>, *, <parameters>) -> xr.DataArray`; `s2cloudless_mask(*, b01, …, b12, probability_threshold=0.4)`, `cdi_cloud_mask(b07, b08, b8a, *, cdi_threshold=-0.5, eps=1e-6)`, `cirrus_cloud_mask(b10, *, reflectance_threshold=0.01)`, `scl_valid_mask(scl, *, valid_classes=SCL_VALID_CLASSES)`.
- `feature_raster` stays in `_raster.py` until Task 2; `shadow_mask` still uses it.

- [ ] **Step 1: Write the shared builder**

`tests/geodata/features/conftest.py`:

```python
"""Shared Sentinel-2 builder for feature tests."""

from __future__ import annotations

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import raster

BANDS = ("B01", "B02", "B04", "B05", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


def sentinel(chunks: dict[str, int] | None = None) -> xr.Dataset:
    """Build prepared reflectance, an SCL band, and per-date sun azimuth.

    Args:
        chunks: Dask chunking to apply, or None to stay eager.

    Returns:
        Two dates of 32 by 32 pixels at 10 m, every band carrying a
        `long_name` so a test can see whether it leaks.
    """
    grid = GeoBox.from_bbox((0, 0, 320, 320), crs="EPSG:32633", resolution=10)
    values = np.random.default_rng(4).uniform(0.1, 0.8, (2, 32, 32)).astype("float32")
    dims = ("time", "y", "x")
    built = raster(
        {band: (dims, values + 0.01 * offset) for offset, band in enumerate(BANDS)}
        | {"SCL": (dims, np.full(values.shape, 4, dtype="uint8"))},
        grid,
        coords={"time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")},
    )
    built = built.assign_coords(sun_azimuth=("time", [45.0, 135.0]), site="plot-1")
    built.sun_azimuth.attrs["units"] = "degree"
    for name in built.data_vars:
        built[name].attrs["long_name"] = f"band {name}"
    return built if chunks is None else built.chunk(chunks)
```

- [ ] **Step 2: Write the failing tests**

Replace `tests/geodata/features/test_spectral_indices.py` with:

```python
"""Tests for spectral indices and the reflectance guard they share."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geosave_engine.geodata import features

from tests.geodata.features.conftest import sentinel

CHUNKS = {"time": 1, "y": 16, "x": 16}
INDICES = {
    "ndvi": ("B08", "B04"),
    "evi": ("B08", "B04", "B02"),
    "evi2": ("B08", "B04"),
    "savi": ("B08", "B04"),
    "msavi2": ("B08", "B04"),
    "ndre": ("B08", "B05"),
    "bsi": ("B11", "B04", "B08", "B02"),
    "ndbi": ("B11", "B08"),
    "ndwi": ("B04", "B08"),
    "mndwi": ("B04", "B11"),
    "ndci": ("B05", "B04"),
    "ndmi": ("B08", "B11"),
    "nbr": ("B08", "B12"),
    "ndsi": ("B04", "B11"),
}


def test_ndvi_is_the_normalized_difference_of_its_two_bands() -> None:
    red = xr.DataArray([[1.0]], dims=("y", "x"))

    result = features.ndvi(3 * red, red)

    np.testing.assert_allclose(result.values, [[0.5]])


@pytest.mark.parametrize("index", sorted(INDICES))
def test_an_index_is_a_lazy_unnamed_band_on_its_inputs_grid(index: str) -> None:
    scene = sentinel(CHUNKS)

    result = getattr(features, index)(*(scene[band] for band in INDICES[index]))

    assert isinstance(result.data, da.Array)
    assert result.name is None
    assert result.attrs == {}
    assert result.dtype == np.float32
    assert result.gs.geobox == scene.gs.geobox
    xr.testing.assert_identical(
        result.coords.to_dataset(), scene.B04.coords.to_dataset()
    )
    assert np.isfinite(result.compute()).all()


@pytest.mark.parametrize("stored", [{"scale_factor": 0.0001}, {"nodata": 0}])
def test_an_index_refuses_a_band_still_holding_stored_values(stored) -> None:
    scene = sentinel()
    scene.B04.attrs.update(stored)

    with pytest.raises(ValueError, match="to_nan.*unpack"):
        features.ndvi(scene.B08, scene.B04)


def test_an_index_refuses_bands_on_different_grids() -> None:
    scene = sentinel()
    shifted = scene.B04.assign_coords(x=scene.x + 10)

    with pytest.raises(ValueError, match="same grid"):
        features.ndvi(scene.B08, shifted)


def test_a_dead_pixel_is_nan_rather_than_infinite() -> None:
    dark = xr.DataArray([[0.0]], dims=("y", "x"))

    assert np.isnan(features.ndvi(dark, dark).values).all()
```

Create `tests/geodata/features/test_cloud_mask.py`:

```python
"""Tests for Sentinel-2 cloud and validity masks."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import features, raster
from geosave_engine.geodata.features.cloud_mask import S2C_BAND_ORDER

from tests.geodata.features.conftest import sentinel

CHUNKS = {"time": 1, "y": 16, "x": 16}
MASKS = {
    "cdi_cloud_mask": ("B07", "B08", "B8A"),
    "cirrus_cloud_mask": ("B10",),
    "scl_valid_mask": ("SCL",),
}


def s2cloudless_bands(scene: xr.Dataset) -> dict[str, xr.DataArray]:
    """Name the scene's bands the way `s2cloudless_mask` takes them."""
    return {name: scene[name.upper()] for name in S2C_BAND_ORDER}


@pytest.mark.parametrize("mask", sorted(MASKS))
def test_a_mask_is_a_lazy_unnamed_bool_band_on_its_inputs_grid(mask: str) -> None:
    scene = sentinel(CHUNKS)

    result = getattr(features, mask)(*(scene[band] for band in MASKS[mask]))

    assert isinstance(result.data, da.Array)
    assert result.dtype == bool
    assert result.name is None
    assert result.attrs == {}
    assert result.gs.geobox == scene.gs.geobox


def test_scl_valid_mask_keeps_only_the_classes_asked_for() -> None:
    scl = sentinel().SCL

    assert features.scl_valid_mask(scl).all()
    assert not features.scl_valid_mask(scl, valid_classes=[5, 6]).any()


def test_s2cloudless_mask_schedules_without_running_the_model() -> None:
    scene = sentinel(CHUNKS)

    result = features.s2cloudless_mask(**s2cloudless_bands(scene))

    assert isinstance(result.data, da.Array)
    assert result.dtype == bool
    assert result.name is None
    assert result.gs.geobox == scene.gs.geobox


def test_s2cloudless_mask_refuses_stored_values_before_scheduling() -> None:
    scene = sentinel(CHUNKS)
    scene.B09.attrs["scale_factor"] = 0.0001

    with pytest.raises(ValueError, match="to_nan.*unpack"):
        features.s2cloudless_mask(**s2cloudless_bands(scene))


def test_cdi_cloud_mask_is_the_same_across_chunk_edges() -> None:
    scene = sentinel()
    eager = features.cdi_cloud_mask(scene.B07, scene.B08 * 0.8, scene.B8A * 1.1)

    chunked = sentinel(CHUNKS)
    lazy = features.cdi_cloud_mask(chunked.B07, chunked.B08 * 0.8, chunked.B8A * 1.1)

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

    eager = features.cdi_cloud_mask(source.b07, source.b08, source.b8a)
    chunked = source.chunk({"y": 32, "x": 32})
    lazy = features.cdi_cloud_mask(chunked.b07, chunked.b08, chunked.b8a)

    xr.testing.assert_identical(lazy.compute(), eager)
    assert not eager.values[10, 10]
    if missing == "one":
        assert eager.values[50, 50]
    else:
        assert not eager.any()
```

In `tests/geodata/stac/test_dataflow.py` replace the derived/composite lines with:

```python
    derived = reflectance.assign(ndvi=ndvi(reflectance.nir, reflectance.red))[["ndvi"]]
    composite = derived.median("time", keep_attrs=True)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/geodata/features/test_spectral_indices.py tests/geodata/features/test_cloud_mask.py tests/geodata/stac/test_dataflow.py -q`
Expected: every test fails with `TypeError` on the old keyword-only signatures (`missing 1 required keyword-only argument: 'name'` or `takes 1 positional argument`).

- [ ] **Step 4: Implement the guard**

`src/geosave_engine/geodata/features/_raster.py`: keep `feature_raster` untouched for now, and replace `prepared_reflectance` with:

```python
def bare(band: xr.DataArray) -> xr.DataArray:
    """Strip a band's name and attrs so neither describes what is derived from it.

    xarray carries both through arithmetic and comparisons. Coordinate attrs
    stay, since the grid is read off them.
    """
    return band.drop_attrs(deep=False).rename(None)


def prepared_reflectance(*bands: xr.DataArray) -> tuple[xr.DataArray, ...]:
    """Require unpacked bands on one grid and return them bare, as float32.

    Raises:
        ValueError: A band still holds packed or fill-valued storage, or the
            bands do not sit on the same grid.
    """
    for band in bands:
        packing = attrs.Packing.from_attrs(band.attrs)
        nodata = attrs.Nodata.from_attrs(band.attrs)
        packed = packing is not None and (
            packing.scale_factor is not None or packing.add_offset is not None
        )
        filled = (
            nodata is not None
            and nodata.fill_value is not None
            and not np.isnan(nodata.fill_value)
        )
        if packed or filled:
            raise ValueError(
                f"{band.name!r} still holds stored values; call "
                ".gs.to_nan().gs.unpack() before computing reflectance features"
            )
    try:
        # Arithmetic would inner-join bands on different grids into a sliver.
        xr.align(*bands, join="exact", copy=False)
    except ValueError as error:
        names = [band.name for band in bands]
        raise ValueError(
            f"{names} do not sit on the same grid; reproject them onto one first"
        ) from error
    return tuple(bare(band).astype(np.float32) for band in bands)
```

Change the module docstring to `"""Input preparation the reflectance features share."""`.

- [ ] **Step 5: Implement the indices**

Replace `src/geosave_engine/geodata/features/spectral_indices.py` with:

```python
"""Spectral indices derived from prepared reflectance bands.

Every index takes its bands as DataArrays on one grid and returns an unnamed
float32 band on it, lazy where the inputs are. Name the result where it is
put: `scene.assign(ndvi=ndvi(scene.B08, scene.B04))`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._raster import prepared_reflectance


def _ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    """Divide, answering NaN where the denominator vanishes."""
    return numerator / denominator.where(denominator != 0)


def _normalized_difference(
    a: xr.DataArray, b: xr.DataArray, eps: float
) -> xr.DataArray:
    """Return `(a - b) / (a + b + eps)` over prepared bands."""
    a, b = prepared_reflectance(a, b)
    return _ratio(a - b, a + b + eps)


def ndvi(nir: xr.DataArray, red: xr.DataArray, *, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference vegetation index.

    Args:
        nir: Near-infrared reflectance.
        red: Red reflectance.
        eps: Added to the denominator; zero leaves a dead pixel as NaN.

    Returns:
        Unnamed float32 band on the inputs' grid.

    Raises:
        ValueError: A band still holds stored values, or the bands sit on
            different grids.

    Examples:
        >>> scene.assign(ndvi=ndvi(scene.B08, scene.B04)).ndvi.dims
        ('time', 'y', 'x')
    """
    return _normalized_difference(nir, red, eps)


def evi(
    nir: xr.DataArray,
    red: xr.DataArray,
    blue: xr.DataArray,
    *,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    L: float = 1.0,
    eps: float = 0.0,
) -> xr.DataArray:
    """Derive the enhanced vegetation index."""
    nir, red, blue = prepared_reflectance(nir, red, blue)
    return g * _ratio(nir - red, nir + c1 * red - c2 * blue + L + eps)


def evi2(
    nir: xr.DataArray,
    red: xr.DataArray,
    *,
    g: float = 2.5,
    c: float = 2.4,
    L: float = 1.0,
    eps: float = 0.0,
) -> xr.DataArray:
    """Derive the two-band enhanced vegetation index."""
    nir, red = prepared_reflectance(nir, red)
    return g * _ratio(nir - red, nir + c * red + L + eps)


def savi(
    nir: xr.DataArray, red: xr.DataArray, *, L: float = 0.5, eps: float = 0.0
) -> xr.DataArray:
    """Derive the soil-adjusted vegetation index."""
    nir, red = prepared_reflectance(nir, red)
    return _ratio(nir - red, nir + red + L + eps) * (1 + L)


def msavi2(nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
    """Derive the modified soil-adjusted vegetation index."""
    nir, red = prepared_reflectance(nir, red)
    term = np.clip((2 * nir + 1) ** 2 - 8 * (nir - red), 0.0, None)
    return (2 * nir + 1 - np.sqrt(term)) / 2


def ndre(
    nir: xr.DataArray, red_edge: xr.DataArray, *, eps: float = 0.0
) -> xr.DataArray:
    """Derive the normalized difference red-edge index."""
    return _normalized_difference(nir, red_edge, eps)


def bsi(
    swir1: xr.DataArray,
    red: xr.DataArray,
    nir: xr.DataArray,
    blue: xr.DataArray,
    *,
    eps: float = 0.0,
) -> xr.DataArray:
    """Derive the bare soil index."""
    swir1, red, nir, blue = prepared_reflectance(swir1, red, nir, blue)
    return _ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue) + eps)


def ndbi(swir1: xr.DataArray, nir: xr.DataArray, *, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference built-up index."""
    return _normalized_difference(swir1, nir, eps)


def ndwi(green: xr.DataArray, nir: xr.DataArray, *, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference water index."""
    return _normalized_difference(green, nir, eps)


def mndwi(
    green: xr.DataArray, swir1: xr.DataArray, *, eps: float = 0.0
) -> xr.DataArray:
    """Derive the modified normalized difference water index."""
    return _normalized_difference(green, swir1, eps)


def ndci(
    red_edge: xr.DataArray, red: xr.DataArray, *, eps: float = 0.0
) -> xr.DataArray:
    """Derive the normalized difference chlorophyll index."""
    return _normalized_difference(red_edge, red, eps)


def ndmi(nir: xr.DataArray, swir1: xr.DataArray, *, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized difference moisture index."""
    return _normalized_difference(nir, swir1, eps)


def nbr(nir: xr.DataArray, swir2: xr.DataArray, *, eps: float = 0.0) -> xr.DataArray:
    """Derive the normalized burn ratio."""
    return _normalized_difference(nir, swir2, eps)


def ndsi(
    green: xr.DataArray, swir1: xr.DataArray, *, eps: float = 0.0
) -> xr.DataArray:
    """Derive the normalized difference snow index."""
    return _normalized_difference(green, swir1, eps)
```

- [ ] **Step 6: Implement the cloud masks**

In `src/geosave_engine/geodata/features/cloud_mask.py`, change the import to `from ._raster import bare, prepared_reflectance` and replace the four public functions (constants and the private `_…_block`, `_local_variance`, `_s2cloudless_detector` helpers stay as they are):

```python
def s2cloudless_mask(
    *,
    b01: xr.DataArray,
    b02: xr.DataArray,
    b04: xr.DataArray,
    b05: xr.DataArray,
    b08: xr.DataArray,
    b8a: xr.DataArray,
    b09: xr.DataArray,
    b10: xr.DataArray,
    b11: xr.DataArray,
    b12: xr.DataArray,
    probability_threshold: float = 0.4,
) -> xr.DataArray:
    """Flag cloud with s2cloudless, one chunk at a time.

    Needs Sentinel-2 L1C TOA reflectance in [0, 1]; prepare stored bands with
    `.gs.to_nan().gs.unpack()` first. Bands are keyword-only because ten
    positional arrays are easy to misorder.

    Args:
        b01: Coastal aerosol reflectance.
        b02: Blue reflectance.
        b04: Red reflectance.
        b05: Red-edge 1 reflectance.
        b08: Near-infrared reflectance.
        b8a: Narrow near-infrared reflectance.
        b09: Water vapour reflectance.
        b10: Cirrus reflectance.
        b11: Shortwave-infrared 1 reflectance.
        b12: Shortwave-infrared 2 reflectance.
        probability_threshold: Cloud probability above which a pixel is flagged.

    Returns:
        Unnamed bool band, True where cloud, lazy when the inputs are.

    Raises:
        ValueError: A band still holds stored values, or the bands sit on
            different grids.
    """
    bands = prepared_reflectance(b01, b02, b04, b05, b08, b8a, b09, b10, b11, b12)
    return map_spatial_overlap(
        _s2cloudless_block,
        *bands,
        depth=3,
        dtype="bool",
        probability_threshold=probability_threshold,
    )


def cdi_cloud_mask(
    b07: xr.DataArray,
    b08: xr.DataArray,
    b8a: xr.DataArray,
    *,
    cdi_threshold: float = -0.5,
    eps: float = 1e-6,
) -> xr.DataArray:
    """Flag cloud with the Cloud Displacement Index.

    CDI = (V(B07/B8A) - V(B08/B8A)) / (V(B07/B8A) + V(B08/B8A)), where V is
    local variance. B08 is pre-smoothed to match B07/B8A's coarser native
    resolution. Missing observations are excluded from neighborhood estimates;
    an unavailable center pixel is not flagged as cloud.

    Args:
        b07: Band 7 reflectance.
        b08: Band 8 reflectance.
        b8a: Band 8A reflectance.
        cdi_threshold: CDI below this is flagged as cloud.
        eps: Guards division by zero.

    Returns:
        Unnamed bool band, True where cloud, lazy when the inputs are.

    Raises:
        ValueError: A band still holds stored values, or the bands sit on
            different grids.
    """
    return map_spatial_overlap(
        _cdi_block,
        *prepared_reflectance(b07, b08, b8a),
        depth=8,
        dtype="bool",
        cdi_threshold=cdi_threshold,
        eps=eps,
    )


def cirrus_cloud_mask(
    b10: xr.DataArray, *, reflectance_threshold: float = 0.01
) -> xr.DataArray:
    """Flag cirrus where Sentinel-2 Band 10 reflectance exceeds a threshold.

    Args:
        b10: Band 10 reflectance.
        reflectance_threshold: Reflectance above which cirrus is flagged.

    Returns:
        Unnamed bool band, True where cirrus, lazy when the input is.

    Raises:
        ValueError: The band still holds stored values.
    """
    (band,) = prepared_reflectance(b10)
    return band > reflectance_threshold


def scl_valid_mask(
    scl: xr.DataArray, *, valid_classes: Sequence[int] = SCL_VALID_CLASSES
) -> xr.DataArray:
    """Flag pixels valid under Sentinel-2 L2A's Scene Classification Layer.

    Sen2Cor's own per-pixel classification: 0=no data, 1=saturated, 2=dark,
    3=cloud shadow, 4=vegetation, 5=bare soil, 6=water, 7=unclassified,
    8/9=cloud med/high prob, 10=cirrus, 11=snow/ice.

    Args:
        scl: Scene Classification Layer codes.
        valid_classes: SCL values retained as valid. The default includes dark
            areas and snow; pass a narrower set such as 4, 5, 6, and 7 when
            those should be excluded.

    Returns:
        Unnamed bool band, True where pixels are valid, lazy when the input is.
    """
    return bare(scl).isin(valid_classes)
```

Update the module docstring to `"""Cloud and validity masks derived from Sentinel-2 bands."""`.

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/geodata/features/test_spectral_indices.py tests/geodata/features/test_cloud_mask.py tests/geodata/stac/test_dataflow.py -q`
Expected: all pass. `test_inputs.py` and `test_raster_interface.py` now fail on the old signatures; Task 2 removes them.

---

### Task 2: `shadow_mask` on arrays; drop `feature_raster`

**Files:**
- Modify: `src/geosave_engine/geodata/features/shadow_mask.py`, `_overlap.py`, `_raster.py`
- Create: `tests/geodata/features/test_shadow_mask.py`
- Delete: `tests/geodata/features/test_inputs.py`, `tests/geodata/features/test_raster_interface.py`

**Interfaces:**
- Consumes: `sentinel` from Task 1's `conftest.py`.
- Produces: `shadow_mask(cloud: xr.DataArray, sun_azimuth: xr.DataArray | float, *, shadow_distance_m: float = 500) -> xr.DataArray`. `feature_raster` no longer exists.

- [ ] **Step 1: Write the failing tests**

`tests/geodata/features/test_shadow_mask.py`:

```python
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


def cloud(chunks: dict[str, int] | None = None) -> xr.DataArray:
    """Build a bool cloud band carrying the scene's `sun_azimuth` coordinate."""
    scene = sentinel(chunks)
    return scene.B10 > 0.5


@pytest.mark.parametrize("chunks", [None, CHUNKS])
def test_shadow_keeps_coordinates_grid_and_laziness(chunks) -> None:
    mask = cloud(chunks)

    result = shadow_mask(mask, mask.sun_azimuth, shadow_distance_m=30)

    assert isinstance(result.data, da.Array) == (chunks is not None)
    assert result.dtype == bool
    assert result.name is None
    assert result.attrs == {}
    assert result.gs.geobox == mask.gs.geobox
    xr.testing.assert_identical(result.coords.to_dataset(), mask.coords.to_dataset())


def test_shadow_is_the_same_across_chunk_edges() -> None:
    eager_mask, lazy_mask = cloud(), cloud(CHUNKS)

    eager = shadow_mask(eager_mask, eager_mask.sun_azimuth, shadow_distance_m=30)
    lazy = shadow_mask(lazy_mask, lazy_mask.sun_azimuth, shadow_distance_m=30)

    xr.testing.assert_identical(lazy.compute(), eager)


def test_shadow_falls_opposite_the_sun_at_the_grid_resolution() -> None:
    grid = GeoBox.from_bbox((0, 0, 100, 100), crs="EPSG:32633", resolution=20)
    pixels = np.zeros((5, 5), dtype=bool)
    pixels[2, 2] = True
    mask = raster({"cloud": (("y", "x"), pixels)}, grid).cloud

    # Sun due south: shadow falls north, two 20 m pixels within 40 m.
    result = shadow_mask(mask, 180.0, shadow_distance_m=40)

    assert result.values[:, 2].tolist() == [True, True, False, False, False]
    assert result.values.sum() == 2


def test_each_date_uses_its_own_azimuth() -> None:
    mask = cloud()

    both = shadow_mask(mask, mask.sun_azimuth, shadow_distance_m=30)
    first = shadow_mask(mask.isel(time=0), 45.0, shadow_distance_m=30)
    second = shadow_mask(mask.isel(time=1), 135.0, shadow_distance_m=30)

    np.testing.assert_array_equal(both.isel(time=0), first)
    np.testing.assert_array_equal(both.isel(time=1), second)
    assert not np.array_equal(first, second)


def test_shadow_refuses_a_mask_that_is_not_boolean() -> None:
    scene = sentinel()

    with pytest.raises(ValueError, match="boolean"):
        shadow_mask(scene.B10, scene.sun_azimuth)


def test_shadow_refuses_a_grid_that_does_not_measure_metres() -> None:
    geographic = GeoBox((4, 4), Affine(0.01, 0, 10, 0, -0.01, 20), "EPSG:4326")
    mask = raster({"cloud": (("y", "x"), np.ones((4, 4), dtype=bool))}, geographic)
    unreferenced = xr.DataArray(np.ones((4, 4), dtype=bool), dims=("y", "x"))

    for held in (mask.cloud, unreferenced):
        with pytest.raises(ValueError, match="metres"):
            shadow_mask(held, 180.0)


def test_shadow_refuses_an_azimuth_that_does_not_match_the_mask() -> None:
    mask = cloud()

    with pytest.raises(ValueError, match="scalar or along 'time'"):
        shadow_mask(mask, xr.DataArray([1.0, 2.0], dims="site"))
    with pytest.raises(ValueError, match="varies with time"):
        shadow_mask(mask.isel(time=0, drop=True), mask.sun_azimuth)
    with pytest.raises(ValueError, match="2 dates"):
        shadow_mask(mask, xr.DataArray([1.0, 2.0, 3.0], dims="time"))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/geodata/features/test_shadow_mask.py -q`
Expected: every test fails with `TypeError` (`shadow_mask() takes 1 positional argument but 2 were given`).

- [ ] **Step 3: Implement `shadow_mask`**

In `src/geosave_engine/geodata/features/shadow_mask.py`, replace the imports and the public function (`_project` and `_shadow_block` stay):

```python
from __future__ import annotations

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.conventions import TIME_COORDINATE

from ._overlap import map_spatial_overlap


def shadow_mask(
    cloud: xr.DataArray,
    sun_azimuth: xr.DataArray | float,
    *,
    shadow_distance_m: float = 500,
) -> xr.DataArray:
    """Flag the ground cloud shadow is estimated to fall on.

    Shifts cloud pixels opposite the sun azimuth for up to
    `shadow_distance_m`, one pixel per step. Steps are counted along `x`, so
    pixels are taken as square.

    Args:
        cloud: Boolean cloud mask on a grid measured in metres.
        sun_azimuth: Degrees clockwise from north, one value or one per date
            along `time`, as `with_properties` loads it onto a raster.
        shadow_distance_m: Furthest a shadow is projected, in metres.

    Returns:
        Unnamed bool band, True where shadow is estimated, on the mask's
        coordinates and lazy when it is.

    Raises:
        ValueError: The mask is not boolean or sits on no metric grid, or the
            azimuth varies along anything but the mask's own dates.

    Examples:
        >>> shadow = shadow_mask(scene.cloud, scene.sun_azimuth)
        >>> scene.assign(shadow=shadow).shadow.dtype
        dtype('bool')
    """
    if cloud.dtype != bool:
        raise ValueError(f"cloud mask is {cloud.dtype}, not boolean")
    geobox = cloud.gs.geobox
    if (
        not isinstance(geobox, GeoBox)
        or geobox.crs is None
        or geobox.crs.units != ("metre", "metre")
    ):
        raise ValueError(
            "cloud mask sits on no grid measured in metres, so a shadow distance "
            "in metres spans no known pixels; reproject it to a projected CRS first"
        )

    azimuth = xr.DataArray(sun_azimuth)
    if set(azimuth.dims) - {TIME_COORDINATE}:
        raise ValueError(
            f"sun azimuth varies along {list(azimuth.dims)}; expected a scalar "
            f"or along {TIME_COORDINATE!r}"
        )
    steps = round(shadow_distance_m / abs(geobox.resolution.x))
    if not azimuth.dims:
        return _project(cloud, float(azimuth), steps)

    if TIME_COORDINATE not in cloud.dims:
        raise ValueError("sun azimuth varies with time but the cloud mask does not")
    dates = cloud.sizes[TIME_COORDINATE]
    if azimuth.sizes[TIME_COORDINATE] != dates:
        raise ValueError(
            f"sun azimuth holds {azimuth.sizes[TIME_COORDINATE]} values but the "
            f"cloud mask spans {dates} dates"
        )
    fields = [
        _project(cloud.isel(time=index), float(azimuth.isel(time=index)), steps)
        for index in range(dates)
    ]
    # Each date's slice carries scalar copies of the time-indexed coordinates.
    return xr.concat(
        fields, dim=cloud.coords[TIME_COORDINATE], coords="minimal", compat="override"
    ).assign_coords(cloud.coords)
```

Update the module docstring's first line only if it still mentions a raster; it reads `"""Project cloud pixels onto the ground their shadow is estimated to fall on."""` and stays.

- [ ] **Step 4: Tidy the two helpers**

`_raster.py`: delete `feature_raster`.

`_overlap.py`: import `from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS, TIME_COORDINATE` and replace the dimension check and the later `spatial_dims = dims[-2:]` with the convention:

```python
    reference = fields[0]
    dims = tuple(str(dim) for dim in reference.dims)
    if dims not in (SPATIAL_DIMENSIONS, (TIME_COORDINATE, *SPATIAL_DIMENSIONS)):
        raise ValueError(
            f"Field has dimensions {dims}; expected {SPATIAL_DIMENSIONS}, "
            f"optionally preceded by {TIME_COORDINATE!r}"
        )
```

and, in the dask branch, drop the `spatial_dims = dims[-2:]` line and read `SPATIAL_DIMENSIONS` in the two places it was used. In the docstring, change "the two trailing spatial dimensions" to "`y` and `x`".

- [ ] **Step 5: Remove the superseded tests**

Delete `tests/geodata/features/test_inputs.py` and `tests/geodata/features/test_raster_interface.py`.

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/geodata/features -q && grep -rn "feature_raster\|name=\"" src/geosave_engine/geodata/features`
Expected: all feature tests pass; the grep prints nothing.

---

### Task 3: Prove the YAML path and show it in the template

**Files:**
- Modify: `tests/model/spec/test_execution.py`
- Modify: `src/geosave_engine/templates/workspaces/segmentation/configs/model_spec.yaml`

**Interfaces:**
- Consumes: `cirrus_cloud_mask` and `shadow_mask` from Tasks 1-2.

- [ ] **Step 1: Write the test**

Append to `tests/model/spec/test_execution.py`:

```python
def test_preprocessing_hands_a_feature_pixels_and_a_coordinate_by_reference() -> None:
    import dask.array as da

    from tests.geodata.features.conftest import sentinel

    scene = sentinel({"time": 1, "y": 16, "x": 16})
    model = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {
                "optical": {"variables": ["B10"], "coordinates": ["sun_azimuth"]}
            },
            "preprocessing": {
                "cloud": {
                    "call": "geosave_engine.geodata.features.cirrus_cloud_mask",
                    "kwargs": {"b10": Ref("optical.B10")},
                },
                "shadow": {
                    "call": "geosave_engine.geodata.features.shadow_mask",
                    "kwargs": {
                        "cloud": Ref("cloud"),
                        "sun_azimuth": Ref("optical.sun_azimuth"),
                        "shadow_distance_m": 30,
                    },
                },
                "image": {
                    "call": Ref("optical.assign"),
                    "kwargs": {"cloud": Ref("cloud"), "shadow": Ref("shadow")},
                },
            },
            "inputs": {"image": Ref("image")},
        }
    )

    image = model.preprocess({"optical": scene})["image"]

    assert list(image.data_vars) == ["B10", "cloud", "shadow"]
    assert isinstance(image.shadow.data, da.Array)
    assert image.shadow.dtype == bool
    assert image.gs.geobox == scene.gs.geobox

    with pytest.raises(KeyError, match="sun_azimuth"):
        model.preprocess({"optical": scene.drop_vars("sun_azimuth")})
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/model/spec/test_execution.py -q -k by_reference`
Expected: PASS. This test pins behaviour Tasks 1-2 already built, so it has no red phase of its own; if it fails, the failure names which link of the YAML path is broken (reference resolution, coordinate survival through `select_raster`, or `assign` as a call).
- [ ] **Step 3: Show the pattern in the template**

In `src/geosave_engine/templates/workspaces/segmentation/configs/model_spec.yaml`, add after the `preprocessing:` block's `image:` entry and before `chips:`:

```yaml
# A feature takes bands and coordinates by reference and is named where it is
# put. A coordinate comes from ingest: list it under `coordinates`, and load it
# under a plain name, since a `!ref` path cannot hold the colon in a STAC key.
#   rasters.sentinel_2_l2a.stac.load.with_properties:
#     - {key: "view:sun_azimuth", name: sun_azimuth}
#   preprocessing:
#     shadow:
#       call: geosave_engine.geodata.features.shadow_mask
#       kwargs: {cloud: !ref cloud, sun_azimuth: !ref sentinel_2_l2a.sun_azimuth}
#     image:
#       call: !ref valid_pixels.assign
#       kwargs: {shadow: !ref shadow}
```

- [ ] **Step 4: Run everything**

Run: `uv run pytest tests/geodata tests/ml tests/model -q && uv run ruff check src tests`
Expected: 0 failed; ruff clean. Report the actual pass count against the 1519 baseline (the feature tests are reorganised, so the number changes).

- [ ] **Step 5: Smoke-test from real YAML**

```bash
uv run python - <<'EOF'
import yaml
from pathlib import Path
from geosave_engine.model.spec import ModelSpec
from tests.geodata.features.conftest import sentinel

text = """
schema_version: 2
rasters:
  optical:
    variables: [B10]
    coordinates: [sun_azimuth]
preprocessing:
  cloud:
    call: geosave_engine.geodata.features.cirrus_cloud_mask
    kwargs: {b10: !ref optical.B10}
  shadow:
    call: geosave_engine.geodata.features.shadow_mask
    kwargs: {cloud: !ref cloud, sun_azimuth: !ref optical.sun_azimuth, shadow_distance_m: 30}
  image:
    call: !ref optical.assign
    kwargs: {shadow: !ref shadow}
inputs:
  image: !ref image
"""
path = Path("/tmp/claude-1000/-home-uwu-Projects-geosave-engine/72043f56-3049-4b26-b353-b365537429b9/scratchpad/model_spec.yaml")
path.write_text(text)
image = ModelSpec.load(path).preprocess({"optical": sentinel({"time": 1, "y": 16, "x": 16})})["image"]
print(list(image.data_vars), image.shadow.dtype, type(image.shadow.data).__name__)
EOF
```

Expected: `['B10', 'shadow'] bool Array`.

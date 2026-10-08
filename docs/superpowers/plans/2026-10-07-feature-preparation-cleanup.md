# Feature Preparation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native execution, or superpowers:subagent-driven-development if the user selects delegation. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove the obsolete feature preparation helper while preserving feature values, calculation dtypes, metadata, and lazy execution.

**Architecture:** `.gs.mask_and_scale()` owns storage decoding. Features select variables from a Dataset and calculate from the supplied values using native xarray operations. Numerical kernels own their calculation dtype; feature outputs discard source-variable names and attrs while retaining coordinates.

**Tech Stack:** Existing xarray, NumPy, Dask, SciPy, s2cloudless, and pytest dependencies.

**Spec:** The agreed in-chat direction is restated in [Design](#design) below. This plan covers the remaining cleanup after the Dataset feature API and `utils/dask_mapping.py` were implemented.

## Design

The caller API stays:

```python
reflectance = scene.gs.mask_and_scale()
scene = scene.assign(
    ndvi=features.ndvi(reflectance, nir="B08", red="B04"),
)
```

A maintainer uses ordinary xarray selection, casting, and arithmetic:

```python
bands = scene[[nir, red]].astype("float32")
a, b = bands[nir], bands[red]
denominator = a + b + eps
result = (a - b) / denominator.where(denominator != 0)
return result.drop_attrs(deep=False).rename(None)
```

Delete `features/_raster.py`, including `reflectance_bands()` and `bare()`. Keep the existing formula helpers in `spectral_indices.py`; selection, numeric conversion, and output cleanup belong with the calculations that need them. Use native operations directly rather than introducing a replacement preparation or output wrapper.

Feature functions calculate from supplied pixel values and do not interpret packing or nodata attrs. Callers must decode stored data explicitly. Removing the current packed-input `ValueError` is intentional: passing stored numbers will calculate from those numbers rather than decode them or reject them.

Float32 input conversion remains part of spectral calculations. Casting only the final result permits unsigned integer underflow and changes numerical behavior. Neighborhood kernels also retain their existing float32 calculations. Cirrus retains float32 comparison behavior; SCL membership preserves its input category values.

Derived outputs remain unnamed DataArrays with empty variable attrs. Coordinate attrs, indexes, CRS/grid information, and laziness are preserved. Strip attrs shallowly, with `drop_attrs(deep=False)`, so coordinate metadata survives.

## Global Constraints

- Preserve all current public feature signatures: Dataset input and keyword-only string selectors.
- Preserve float32 spectral-index outputs and boolean mask outputs.
- Preserve formulas, coefficients, thresholds, denominator handling, halo depths, and boundary policies.
- Keep decoding explicit on the existing `.gs.mask_and_scale()` accessor.
- Use the existing Dask mapping helper for neighborhood execution, including its dimension ordering and native chunk handling.
- Keep source objects unchanged; retain lazy execution for Dask inputs.
- Use existing dependencies and source-mirrored tests. Ignore notebooks and generated consumer workspaces.
- Work from the live checkout: it contains unrelated unstaged changes. Review task-owned changes separately; avoid whole-file staging or restoring files from HEAD. Commit only task-owned changes if requested.

## Review Focus

1. Storage attrs on supplied values must not trigger validation, scaling, or masking; pin this in Task 1 and Task 2.
2. Unsigned inputs must be cast before subtraction; pin the NDVI result to `-1` in Task 1.
3. Descriptive and storage attrs must not leak onto derived variables; coordinate metadata and the source must survive in both tasks.
4. Float64 cirrus values just above a threshold must retain the existing float32 comparison behavior; pin this in Task 2.
5. Cloud kernels must receive selectors in semantic order, including repeated variable names, and match across chunk edges; pin this in Task 2.

## Files and Responsibilities

| File | Change |
| --- | --- |
| `src/geosave_engine/geodata/features/spectral_indices.py` | Select and cast bands locally; clean derived outputs; update error documentation. |
| `src/geosave_engine/geodata/features/cloud_mask.py` | Select bands directly; retain kernel and comparison dtypes; clean direct comparison/membership outputs. |
| `src/geosave_engine/geodata/features/_raster.py` | Delete after both callers stop importing it. |
| `tests/geodata/features/test_spectral_indices.py` | Replace packed-input rejection expectations; cover calculation and output contracts. |
| `tests/geodata/features/test_cloud_mask.py` | Replace packed-input rejection expectations; cover threshold, selector, and execution contracts. |
| `docs/guides/architecture.md` | State that callers decode and features calculate supplied values. |

## Task 1: Move Spectral Calculations to Native Selection

**Interfaces:** Consume the existing Dataset/string-selector API and `.gs.mask_and_scale()`. Produce the same unnamed float32 DataArray results; remove the packed-input exception contract for spectral indices.

- [x] Rewrite the existing stored-value rejection test as `test_an_index_uses_supplied_values_without_interpreting_storage_attrs`. Use float32 `nir=0.6`, `red=0.2`; attach `scale_factor=0.0001`, `add_offset=1.0`, or `_FillValue=0.6` in parameterized cases. Assert NDVI is `0.5`, output name is None, output attrs are empty, and the source is identical to a saved copy. This explicitly tests the new supplied-value contract.
- [x] Add `test_ndvi_casts_before_unsigned_subtraction`: use uint16 `nir=0`, `red=1`, without packing/nodata metadata, and assert float32 NDVI equals `-1`. Extend existing metadata/laziness coverage to assert the source remains unchanged.
- [x] Run `uv run pytest -q tests/geodata/features/test_spectral_indices.py`. Expect the supplied-value test to fail on the existing metadata guard; existing arithmetic and metadata tests provide the preservation baseline.
- [x] Replace `reflectance_bands()` calls with selected Dataset subsets cast to float32 before arithmetic. Keep `_ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray` for zero-denominator handling and `_normalized_difference(scene: xr.Dataset, a: str, b: str, eps: float) -> xr.DataArray` for the shared formula.
- [x] Clean completed derived outputs with native `drop_attrs(deep=False).rename(None)`. Normalized-difference functions may share this cleanup through their existing formula helper. Other formulas clean their result where it is returned.
- [x] Remove the `_raster` import from spectral indices and remove their documented stored-value `ValueError`. Preserve native missing-selector `KeyError` and explicit decoding guidance.
- [x] Run the spectral test file again. Expect all formula, denominator, selector, dtype, metadata, source-preservation, and lazy-execution cases to pass.

## Task 2: Simplify Masks and Delete the Obsolete Module

**Interfaces:** Consume the existing cloud-mask signatures and `map_spatial_overlap(func, *fields, depth, dtype, boundary, **func_kwargs) -> xr.DataArray`. Produce the same unnamed boolean masks and remove the remaining feature-preparation module.

- [x] Replace `test_s2cloudless_mask_refuses_stored_values_before_scheduling` with a supplied-values test: attach storage attrs to an otherwise prepared test scene, schedule the mask, and assert it stays lazy. Compare its computed result with the same scene without those attrs, using the real model rather than a mock.
- [x] Add a cirrus test using float32 supplied value `0.02`, `_FillValue=0.02`, and threshold `0.01`. Assert True, empty output attrs, an unnamed result, preserved coordinates, and an unchanged source. This catches implicit masking and metadata validation.
- [x] Add a cirrus precision test with a float64 value `np.nextafter(np.float64(0.01), np.inf)` and threshold `0.01`. Assert False after the existing float32 conversion, for both eager and chunked inputs.
- [x] Add a CDI dtype test with uint16 `b07=1`, `b8a=1`, and a deterministic spatial `b08` field of zeros and ones. Compare its eager/chunked masks with the same supplied values cast to float32. This catches NaN-buffer allocation before converting `b08` to floating point.
- [x] Add an eager/lazy s2cloudless comparison using the same variable for all ten selectors. Assert boolean outputs with the scene's dimensions and matching values. This catches selection code that deduplicates channels. Keep the existing CDI chunk-edge and missing-pixel tests.
- [x] Run `uv run pytest -q tests/geodata/features/test_cloud_mask.py`. Expect the supplied-values cases to fail against the current metadata guard.
- [x] In `s2cloudless_mask`, select DataArrays directly in the explicit order `b01, b02, b04, b05, b08, b8a, b09, b10, b11, b12`. Preserve repeated selectors. `_s2cloudless_block` already casts its stacked input to float32; retain that conversion.
- [x] In `cdi_cloud_mask`, select `b07`, `b08`, and `b8a` directly. In `_cdi_block`, cast all three inputs to float32 before calculation or NaN-buffer allocation; add the missing initial `b08` cast now that the shared input cast is removed. Preserve the existing numerical kernel and mapping parameters.
- [x] In `cirrus_cloud_mask`, cast the selected band to float32 before comparison, then clean the result. In `scl_valid_mask`, calculate membership on the selected category band, then clean the result. Mapping already creates clean outputs for neighborhood kernels.
- [x] Remove the `_raster` import from cloud masks, delete `features/_raster.py`, and remove the documented stored-value exceptions. Update `docs/guides/architecture.md` with the explicit caller-decoding contract.
- [x] Run `uv run pytest -q tests/geodata/features tests/geodata/utils/test_dask_mapping.py tests/model/spec/test_execution.py tests/model/spec/test_examples.py tests/geodata/core/test_base.py tests/geodata/stac/test_dataflow.py`. Use local socket access if Zarr/STAC fixtures stall under restricted execution. Expect all cases to pass.
- [x] Run `uv run ruff check src tests`, scoped BasedPyright on the changed feature modules/tests, and `git diff --check`. Report pre-existing diagnostics separately rather than changing unrelated code.
- [x] Run the default full suite with `uv run pytest -q`; record counts and any failures. Check `git status` afterward for fixture changes, preserving unrelated work. Report the removal of packed-input exceptions as the behavior change.

## Planning Evidence

A read-only smoke probe compared direct Dataset selection, float32 arithmetic, and output cleanup with the current NDVI on a chunked Sentinel scene. Values were identical and the result retained its grid and laziness. A uint16 probe produced `65535` without an input cast and `-1` with float32 input conversion, confirming the placement of the cast.

The plan intentionally preserves the implemented Dataset API, decoder, Dask mapping, and shadow-mask design. This refactor removes the remaining preparation bundle without expanding those interfaces.

## Implementation Evidence

Native execution completed. Spectral tests: 35 passed; mask tests: 15 passed; focused integration checks: 115 passed. Ruff and scoped BasedPyright passed, and `git diff --check` was clean. Full default suite: 1635 passed, 31 deselected, 84 warnings in 162.72 seconds. Final independent review found no issues.

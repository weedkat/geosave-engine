# Model Workflows Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the
> independently owned contract/preparation and acquisition tasks. The controller
> implements inference and integrates the flows, then requests fresh review.

**Goal:** Ship executable YAML-driven ingestion and prediction workflows using
native DataTrees, temporal windows, and Prefect stage tasks.

**Architecture:** Ordinary domain functions own numerical behavior. Plain Prefect
flows accept primitive settings and paths, and coordinate stage tasks. Native
methods and custom import paths own operation behavior. Model settings use YAML.

**Tech Stack:** Python >=3.12; existing Pydantic, PyYAML, xarray, Dask, PyTorch,
Prefect, odc-geo and tiler dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-model-workflows.md`

## Global Constraints

- Existing dependencies only; preserve unrelated working-tree changes.
- Raster operations stay lazy until sampling or explicit persistence.
- Preserve CRS, coordinates, spatial dimensions, transform, nodata, dtype and
  variable identity. Test persistence through native round trips.
- No registry compatibility aliases or obsolete execution paths.
- Keep this checkout's uncommitted prerequisite code; do not commit it.

## Review Focus

- Branching operations must not mutate raw data or siblings.
- Duplicate YAML keys, invalid callable arguments, reference cycles and missing
  rasters must fail before pixel computation.
- Static/temporal modalities must preserve channel order, grid and time labels.
- Context collisions, invalid model outputs and model exceptions must fail
  clearly and restore model state.
- Saved results must contain computed pixels; ordinary lazy task completion
  must not be described as a persisted checkpoint.

### Task 1: YAML contract and native preparation

**Files:** `workflow/spec/`, `workflow/preprocessing.py`,
`workflow/examples/`, `tests/workflow/spec/`,
`tests/workflow/test_preprocessing.py`, fixture imports in `tests/workflow/conftest.py`.

**Produces:** `ModelSpec`, `InferenceSpec`, `TimeWindowSpec`, `OperationSpec`,
`PreprocessingSpec`, retained tensor/tiling/segmentation/attrs requirements;
`preprocess(raw, *, spec) -> xr.DataTree`.

- [x] Replace stale tests with behavioral tests, observe failures:
  `restored = ModelSpec.load(spec.save(tmp_path)); assert restored == spec`;
  `prepared = preprocess(raw, spec=spec); assert isinstance(prepared, xr.DataTree)`.
- [x] Implement YAML contract, callable validation and topological preparation.
- [x] Cover Review Focus 1-2, resolution checks and native names.
- [x] Select ordered recipe variables and invoke native Dataset/accessor methods.
- [x] Use native YAML serialization and primitive validation without a separate
  serialization layer; document attrs models, foreign metadata and coordinates.
- [x] Run `uv run --no-sync pytest -q tests/workflow/spec tests/workflow/test_preprocessing.py`.

### Task 2: Acquisition and local persistence

**Files:** `workflow/ingestion.py`, `workflow/io.py`, `tests/workflow/test_ingestion.py`,
`tests/workflow/test_io.py`.

**Produces:** `acquire(sources, anchor, *, requirements=None) -> xr.DataTree`,
`stac_config(requirement, *, defaults=None)`, file opening and completed local
Zarr persistence helpers. Flow wrappers are integrated by the controller.

- [x] Update real local STAC tests to assert DataTree output, caller isolation,
  source selection and metadata validation; observe expected missing API failures.
- [x] Implement native acquisition and local file/path handling.
- [x] Test GeoTIFF/stack inputs, completed Zarr round trips and URI rejection.
- [x] Expose odc-stac property loading as time coordinates, with native aliases,
  dtype and units; verify real local acquisitions remain lazy.
- [x] Stage raw/prepared writes and clean up failed writes; preserve backend close
  callbacks and open file inputs with Dask chunks.
- [x] Run `uv run --no-sync pytest -q tests/workflow/test_ingestion.py tests/workflow/test_io.py`.

### Task 3: Sampling, numerical inference and stage flows

**Files:** `workflow/sampling.py`, `workflow/inference.py`,
`workflow/postprocessing.py`, `workflow/flows.py`, `workflow/runtime.py`,
`workflow/__init__.py`, `tests/workflow/test_prediction.py`,
`tests/workflow/test_prefect.py`, `tests/workflow/test_sampling.py`.

**Consumes:** Task 1's ModelSpec/InferenceSpec and DataTree preprocessing; Task 2's
native acquisition and I/O; delete the superseded prediction stub. **Produces:** the signatures and two flows in the spec.

- [x] Write a toy temporal model accepting `image`, a static modality and context;
  assert one output per window and exact reconstructed pixels before implementation.
- [x] Implement batching through PyTorch DataLoader, native Tiles/TileMerger,
  explicit tensor encoding and existing segmentation postprocessing.
- [x] Add model state, context, shape, metadata and window association tests.
- [x] Compose bounded ingestion and stage prediction tasks; exercise a real
  temporary Prefect server, failure propagation and saved prepared/final outputs.
- [x] Accept primitive source settings, explicit coordinate/GeoJSON anchors,
  raster paths and saved model paths at the public flow boundary. Open native
  objects inside workers using existing constructors and model loaders.

### Task 4: Cleanup, integration and review

- [x] Replace workflow docs with runnable examples and the final boundaries.
- [x] Delete superseded workflow plans/reviews and obsolete examples; search
  for stale import/class names and broken example links.
- [x] Run focused Ruff checks and complete `uv run --no-sync pytest`.
- [x] Review the full change against the pre-change snapshot, resolve material
  findings, and report the actual checks and remaining scope limits.

## Verification results

- Default test suite: 792 passed, including real local STAC, saved-model Prefect
  flows and native persistence. Local socket access was enabled for Prefect/Zarr.
- Scoped Ruff check and format check passed for all workflow files and touched
  native STAC/I/O files; whitespace and local Markdown link checks passed.
- Offline reflectance example and primitive ingestion YAML smoke checks passed.
- Independent review verified STAC property coordinates through raw persistence,
  native preprocessing and temporal model context; no material findings remain.

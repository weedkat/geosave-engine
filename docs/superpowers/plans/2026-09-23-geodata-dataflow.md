# Geodata Data Flow Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Track completed work here. Preserve the existing dirty workspace; review against `/tmp/geosave-dataflow-baseline-20260923`, not the unrelated HEAD diff.

**Goal:** Untangle geodata around native user operations, remove stale execution paths, and verify each improvement before moving to the next module.

**Architecture:** Existing attrs models describe native xarray objects. One metadata write flow validates then applies edits; source adapters, pixel transforms, features, and persistence each own their respective representation changes.

**Tech Stack:** xarray, Dask, Pydantic, ODC, rasterio, GeoPandas, pytest.

**Spec:** [Geodata data flow and ownership](../specs/2026-09-23-geodata-dataflow-design.md).

## Global constraints

- Preserve unrelated working-tree changes; no automatic commits or branch changes.
- Use existing dependencies; no new dependency is required.
- Keep native xarray objects and lazy arrays in public Interfaces.
- Reprojection, resampling, casting, and eager computation remain explicit.
- Preserve applicable CRS, coordinates, spatial dimensions, transform, nodata, dtype, and band/variable identity.
- Persistence changes require round-trip tests; lazy operations require laziness checks.

## Review focus

- A failed later target or serialization must not leave earlier attrs edited.
- Shared fields must behave independently of model insertion order, including explicit clears.
- An explicit ordered patch must still override an earlier patch.
- Foreign None values must survive; they are not model field deletion markers.
- Pixel storage, coordinates, and original non-inplace inputs must remain unchanged.

## Step 1: attrs write ownership

**Modify:** `src/geosave_engine/geodata/attrs/xarray.py`, `namespace.py`, and stale metadata docstrings in `model.py`.

**Tests:** `tests/geodata/attrs/test_header_stamping.py` and `test_namespace.py`.

**Interface:** Keep `read`, `rebase`, model classes, namespace/header constructors,
and existing ordered bare-model patch behavior. Conflicting shared values in one
namespace now raise ValueError.

- [x] Add failing tests for partial model/namespace writes and conflicting header
  serialization; assert the whole input remains identical after failure.

  ```python
  before = ds.copy(deep=True)
  with pytest.raises(ValueError):
      rebase(ds, Nodata(fill_value=0), target=["red", "missing"], inplace=True)
  xr.testing.assert_identical(ds, before)
  ```

- [x] Add namespace tests for contradictory shared keys in both insertion orders,
  mutation after parsing, explicit clears, matching values, and foreign None.

  ```python
  namespace = AttrsNamespace.from_attrs({"units": "metre"})
  namespace.get(CFVariable).units = "kilometre"
  with pytest.raises(ValueError, match="units"):
      namespace.to_attrs()
  ```

- [x] Run `uv run pytest tests/geodata/attrs -q`; confirm the new assertions fail.
- [x] Flatten all write requests to per-target mappings, retaining the distinction
  between replacement and patching. Prepare all final mappings before writing.
  Remove `_rebase_header`, `_rebase_namespace`, and `_rebase_models` as separate
  mutation paths. Serialize each model patch once, retaining ordered overrides.
- [x] In `AttrsNamespace.to_attrs`, compare duplicate emitted keys using
  `values_agree` before dropping model None values; report the conflicting key
  and model names. Preserve foreign values.
- [x] Remove stale descriptions of per-model merge requirements and removed band
  statistics behavior. Keep comments about current constraints only.
- [x] Verify ordered patches, foreign None, lazy pixel identity, coordinate attrs,
  header restoration, and DataArray/DataTree writes through existing and new tests.
- [x] Run attrs tests, affected integration tests, and Ruff on changed Python files.
  Review the diff against the saved working-tree baseline.

## Ordered continuation

Refine and implement these steps sequentially against their actual callers:

2. Core representation and jointly computed statistics.
3. STAC native transport and effective band metadata.
4. Transform codec normalization and structure preservation.
5. Feature input agreement, coordinates, and explicit reflectance preparation.
6. Pipeline stale exports and portable Manifest paths.

The detailed acceptance example is chosen from the existing user workflows before
these steps are expanded. Record each completed step and its test evidence below.

## Execution record

- Baseline: snapshot of current geodata source/tests saved before editing.
- Design: user requested implementation after refining the plan; work proceeds
  sequentially in the existing workspace, with tests before each behavior change.

### Step 1 evidence

Eight new regression cases failed on the baseline. All 85 attrs tests pass after
removing the three independent write implementations; the attrs write module
shrinks from 438 to 349 lines. Geodata plus workflow specification tests: 546
passed, 31 warnings. Ruff passes for the changed files.

## Step 2: core representation

**Files:** core/array.py, core/raster.py, attrs/models/band.py;
tests/geodata/test_raster.py and a Zarr round-trip regression.

- [x] Jointly compute the five scalar statistics in one native xarray Dataset;
  a delayed source counter must change from five reads to one without collecting
  the full raster in memory.
- [x] Extend BandVariables with only Dataset values shadowed by shared band attrs.
  Its restoration owns the distinction between band keys and Dataset keys;
  preserve overlapping original root values and edits to non-band root values.
  Keep selected-band behavior and JSON-native persistence.
- [x] Reproduce loss of root units overlapping a band's units; verify restored
  band units, root units, and an edited root title independently.
- [x] Check lazy conversion and Zarr round trips, then run focused tests and Ruff.

Ruling: retain the existing conversion Interface and BandVariables model. Root
metadata overwritten by shared band metadata is restored from its recorded scope;
non-band array attrs retain their current edit behavior. This avoids introducing
another raster wrapper or guessing ownership from key subtraction in core.

## Step 3: STAC acquisition ownership

- Keep supplied pystac-client transports untouched; configure retries when the
  built-in factories create a session.
- Move conversion to native ODC load kwargs onto StacSourceConfig. The source
  coordinates search/load/stamping without mirroring configuration fields again.
- Let the loaded Dataset own effective nodata and units. Resolve output aliases
  and multiband assets with ODC's parsed collection metadata; attach only the
  corresponding source packing and description. Keep original source metadata
  in provenance, distinct from output decoding.
- Use real local GeoTIFFs to verify nodata overrides, aliases, multiband packing,
  default lazy reads, and eager mode. Preserve errors for incompatible packing.

## Step 4: transform structure and codec boundaries

- Normalize accepted nodata aliases to CF before delegating masking to xarray's
  codec. Update Dataset variables with native assign so independent coordinates
  survive masking and unpacking.
- Apply same-grid masking through DataTree.map_over_datasets, preserving root
  metadata and coordinates. Rebuilt stacks from concat, warp, and time windows
  retain root attrs; changed spatial and time axes remain operation-owned.
- Regressions cover ODC-only nodata, lazy pixels, disconnected coordinates,
  root metadata, and input immutability.

## Step 5: feature-owned kernels and prepared inputs

- Move the halo executor from general utils into private feature support; remove
  its dependency on core.array and rebuild outputs with native dims/coordinates.
- Check exact indexes and CRS/grid agreement before feature arithmetic. Share
  normalized-difference implementation while retaining readable public formulas.
- Require explicit to_nan/unpack for reflectance features; remove s2cloudless's
  separate automatic unpack path. Do not infer units or eagerly inspect pixels.
- Verify auxiliary coordinates, CRS, laziness, chunk-edge parity, shifted inputs,
  different CRSs, and actual reflectance numerics.

## Step 6: honest pipeline surface and portable paths

- Delete publish.py and its two exports, and remove Manifest.update: all three
  are unimplemented and have no live callers. Native publication can be added
  when a concrete consumer exists.
- Keep the implemented GeoParquet Manifest distinct from workflow's per-label
  resume ledger. Normalize root and Path values before validating containment;
  reject parent traversal and symlinks escaping the root before replacing a row.
- Test actual save/reopen/relocation, equivalent relative path spelling, empty
  frames, and invalid path edits that must leave existing rows intact.


## Completed implementation and review

All six steps were implemented in the requested order. Each behavior regression
was observed failing before its fix. Changes are compared with the saved dirty
workspace snapshot, not with HEAD.

| Step | Removed or consolidated | Result |
| --- | --- | --- |
| attrs | Three independent writers → one prepare/apply path | Atomic failures, explicit namespace conflicts, independent mutable model patches |
| core | Attrs lifting/restoration → BandVariables; five executions → one scalar computation | Correct root/band ownership and one delayed source read |
| STAC | Config field mirroring and private session replacement | Native load kwargs, caller transport ownership, resolved output-band metadata |
| transform | Fresh Dataset construction and manual tree masking rebuild | Independent coordinates and tree roots survive; native CF nodata decoding |
| features | General utils halo executor and repeated normalized differences | Feature-owned kernels, preserved CF coordinates, exact grid checks, explicit preparation |
| pipeline | Unimplemented publish module, exports, and update method | Implemented Manifest only; normalized, portable path references |

Review refinements are included: mutable attrs independence; equivalent shared
model merges; canonical nodata aliases; root edits outside lifted keys; non-finite
metadata through stacking and Zarr; CRS links stored in either attrs or encoding;
and CDI missing pixels with finite-weighted spatial filters. CDI's radius-8 halo
covers its radius-4 Gaussian and radius-3 variance neighborhood.

The concrete acceptance case is
[`test_dataflow.py`](../../../tests/geodata/stac/test_dataflow.py): local multiband
GeoTIFF STAC assets → alias resolution → explicit masking/unpacking → NDVI →
composite → Zarr → lazy reopening with grid, values, and source identity intact.

Verification:

- Geodata + workflow: 678 passed, 34 warnings before the last review refinements (superseded by the full run below).
- Focused attrs/core/features/persistence/STAC flow after those refinements:
  131 passed, 3 warnings.
- Full default repository suite: **843 passed, 46 warnings in 100.20s**.
  Slow/external-service tests remain excluded by the project configuration.
- Ruff check and formatting check pass on all 33 changed Python files.
- Separate reviewers verified attrs, core/STAC, and transform/features/pipeline;
  their concrete findings were reproduced and fixed, and re-review found no
  remaining blocker.

### Compatibility changes

- Contradictory models in one AttrsNamespace now raise ValueError. Ordered bare
  model patches continue to express explicit overrides.
- Reflectance features reject mismatched grids and stored Packing or finite
  nodata markers. Prepare using `.gs.to_nan().gs.unpack()` and align explicitly.
  s2cloudless no longer unpacks implicitly.
- `geodata.utils.dask` was removed; its only repository consumers are now inside
  features and use private support.
- `pipeline.push_to_hub`, `pipeline.push_to_bucket`, and `Manifest.update` were
  removed; each previously raised NotImplementedError.
- STAC's internal `read_header` now takes the loaded Dataset; `_load_options`
  was replaced by `StacSourceConfig.to_load_kwargs()`.
- Manifest rejects paths resolving outside its root, including escaping symlinks.

### Remaining scope

This is the first complete implementation pass across the six requested Modules.
It keeps working tiling, native file readers/writers, visualization, and the live
workflow implementation. STAC 1.1 field translation, broader acquisition-property
handling, and root coordinates whose dimensions change in multi-group warp/time
operations remain separate design work; this pass does not promise to solve
those pre-existing cases. No compatibility aliases or new framework were added.


The library-source diff against the session baseline is 496 added and 567 removed
lines (71 fewer lines overall), including moving the existing halo implementation.
Tests exercise public operations and real local persistence; no external service
was needed for the acceptance flow. Unrelated working-tree changes were retained.

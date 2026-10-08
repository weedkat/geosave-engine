# Attrs Key Dispatch Implementation Plan

> **Superseded:** The user chose independent sequential model parsing instead
> of a key-to-model index. Each model reads the same mapping; a cached set of
> scope-owned keys identifies foreign attrs. No parallel executor is needed
> for the current small registry. The tasks below record the abandoned proposal.

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Preserve the shared working tree; do not commit unrelated changes.

**Goal:** Identify metadata groups by looking up each incoming attr key, without scanning absent models on every parse.

**Architecture:** Keep `AttrsHeader` responsible for xarray locations, `AttrsNamespace` responsible for grouping and foreign attrs, and `AttrsModel` responsible for typed group validation, aliases, serialization, and merging. Derive one immutable key-to-model index per scope from the existing registry at import time. Namespace parsing uses that index and constructs only groups present in the input.

**Tech Stack:** Python, Pydantic, xarray, pytest, Ruff, BasedPyright; no new dependencies.

**Spec:** The agreed design and constraints recorded below.

## Agreed Design and Global Constraints

- `MODELS` remains the single declaration of registered models and configuration names.
- Stored key ownership comes from each model's existing `attr_keys()`; do not maintain a second list of keys.
- Within one scope, every stored key has exactly one owner. Reusing a key in different scopes is valid.
- Build the index once; no runtime registration, generic registry class, compatibility layer, or new public wrapper.
- Keep `AttrsModel.from_attrs()` as the group parser, including equivalent-alias normalization and conflicting-alias errors.
- Preserve source mappings, foreign values, key string normalization, and existing invalid-value behavior.
- Preserve name/class lookup, model-spec configuration names, serialization, and merge semantics.
- Field-validator consistency is separate work and is outside this change.
- Keep unrelated working-tree changes, including the user's GDAL color changes.

## Review Focus

1. The same `units` key must select different models in variable and coordinate scopes.
2. Duplicate ownership in a scope must raise during index construction, rather than depend on declaration order.
3. Both nodata aliases must reach the same model; equivalent text/numeric values succeed and different values fail.
4. Empty, foreign-only, and recognized-null mappings must retain existing results without modifying the source.
5. Invalid recognized values must raise; they must never be demoted to foreign attrs.

## Evidence Before Implementation

A temporary indexed-parser smoke test produced identical flattened results for empty attrs, one recognized key, a typical band, and 100 foreign keys. It found no ownership collisions in the current registry. Local median timings in microseconds were scan/index: 5.7/2.7, 17.5/14.5, 150.9/145.9, and 26.5/26.8 respectively. These support simpler routing and a small sparse-input improvement, not a claim about end-to-end raster performance.

### Task 1: Derive and Validate Stored-Key Ownership

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/models/__init__.py`
- Test: `tests/geodata/attrs/test_models.py`

**Interfaces:**
- Consumes: `MODELS: Mapping[Scope, Mapping[str, type[AttrsModel]]]` and `AttrsModel.attr_keys() -> tuple[str, ...]`.
- Produces: `KEY_MODELS: Mapping[Scope, Mapping[str, type[AttrsModel]]]`, immutable at both levels, for internal package use.
- Builder: `_build_key_models(models: Mapping[Scope, Mapping[str, type[AttrsModel]]]) -> Mapping[Scope, Mapping[str, type[AttrsModel]]]`.

- [ ] Add tests asserting `KEY_MODELS['variable']['units'] is CFVariable`, `KEY_MODELS['coordinate']['units'] is CFCoordinate`, and both variable nodata keys select `Nodata`. Assert each registered stored key appears under its declared scope and owner.
- [ ] Add a construction test using two temporary `AttrsModel` subclasses with the same `shared` field in one scope. Assert `ValueError` identifies the scope, key, and both owners. Verify the same key in separate scopes succeeds.
- [ ] Run `uv run pytest tests/geodata/attrs/test_models.py -q`; confirm new tests fail because the index/builder does not exist.
- [ ] Implement the builder with explicit loops and a concise Google-style docstring describing the nested mapping and collision error. Derive `KEY_MODELS` once from `MODELS`; use `MappingProxyType` for outer and inner mappings.
- [ ] Keep `resolve_model()` and `model_scope()` behavior intact. Make `scope_keys()` read the index keys rather than rediscover ownership; retain its existing return type.
- [ ] Run `uv run pytest tests/geodata/attrs/test_models.py -q`; expect all tests to pass.

### Task 2: Route Incoming Attrs Through the Index

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/namespace.py`
- Modify: `docs/guides/architecture.md`
- Test: `tests/geodata/attrs/test_namespace.py`

**Interfaces:**
- Consumes: Task 1's `KEY_MODELS[scope]`.
- Preserves: `AttrsNamespace.from_attrs(attrs: Mapping[Any, Any], scope: Scope) -> Self` and all Header/Model interfaces.

- [ ] Add a test that monkeypatches an absent model's `attr_keys()` to fail after registry initialization, then parses `{'units': '1'}` in variable scope successfully. This pins the requirement that parsing does not rediscover absent groups.
- [ ] Add cases for empty input, foreign-only input with a nested value, and `{'units': None}`. Assert empty/present model behavior matches the current parser, foreign values survive, and the source is unchanged.
- [ ] Extend grouping tests to cover conflicting nodata aliases. Retain existing equivalent text/numeric aliases, wrong-scope keys, invalid recognized values, and source-preservation assertions.
- [ ] Run `uv run pytest tests/geodata/attrs/test_namespace.py -q`; confirm the absent-model discovery test fails on the old parser.
- [ ] Replace the model scan with one pass over a string-normalized copy of the input. Route each key into a model's flat group or `foreign`, then call `model.from_attrs(group)` only for present groups. Normalizing into a copy first preserves existing behavior if distinct input keys stringify to the same key. Retain the existing `None` guard for group parsing.
- [ ] Update the parser docstring and architecture guide to describe indexed ownership and the three existing responsibilities. Do not add a field-extraction API or move alias validation out of `AttrsModel`.
- [ ] Run `uv run pytest tests/geodata/attrs tests/model/spec/test_rasters.py tests/geodata/core/test_raster.py tests/geodata/io/test_zarr.py tests/geodata/io/test_zarr_stack.py tests/geodata/transform/test_concat.py tests/geodata/transform/test_mosaic.py -q`; expect all tests to pass, including persistence round trips and merge consumers.
- [ ] Repeat a small local parsing benchmark for empty, sparse, representative, and foreign-heavy mappings. Check flattened outputs match and report results without timing assertions or a permanent benchmark framework.
- [ ] Run `uv run ruff check src tests`, scoped BasedPyright on `geodata/attrs`, and `git diff --check`; expect no new diagnostics.
- [ ] Run `uv run pytest -q --tb=short --disable-warnings`. Baseline is 1,557 passed, 16 deselected, and two existing `tests/geodata/core/test_stack.py` failures: restored group order and the NetCDF output parent directory. Investigate new failures; leave these unrelated baseline issues unchanged. Persistence tests may require local runtime access for Zarr.
- [ ] Inspect the final diff and status. Report changes, checks, benchmark limitations, and any new breaking changes; none are intended by this plan.

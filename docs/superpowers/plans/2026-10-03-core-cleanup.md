# Core Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove duplication from `geodata.core` so each behaviour has one implementation, and give DataArrays the operations Datasets already have.

**Architecture:**
- `array()` builds through `raster()`.
- The `gs.rebase` accessor accepts all three `rebase` forms.
- A shared `GeoRasterAccessor` base holds the methods `GeoRaster` and `GeoArray` delegate identically.
- `crop` moves to `transform.vector`, so both accessors get it.
- Single-model reads use `Model.from_attrs`.
- IO writers import lazily.

**Tech Stack:** Python 3.12, xarray, odc-geo, pytest

**Spec:** the design agreed in conversation on 2026-10-03, items 1–6 of the core review. This plan is its record.

## Global Constraints

- Public names stay: `array`, `raster`, `ds.gs.*`, `da.gs.*`.
- No compatibility aliases; no new dependencies; no commits.
- Laziness is preserved: no new eager computation.

## Review Focus

- `array()` errors now name the variable `'pixels'`; Task 3 updates the tests that match the old wording.
- `da.gs.crop` on a DataArray carrying no fill value with `mask=True` must raise the same error the Dataset path does; Task 5 tests this.
- `ds.gs.rebase(header)` and `ds.gs.rebase(namespace, target=...)` behave exactly like `attrs.rebase`; Task 4 tests this.

---

### Task 1: Single-model reads in `GeoArray`, and `colorize` tidy
`plot`, `statistics`, and `colorize` read `attrs.Legend` and `attrs.Nodata` through `Model.from_attrs(self._data.attrs)`. `GeoRaster.plot` drops the duplicated `flags` name, and `colorize` drops its redundant `None` checks. These are no-behaviour edits, so the existing `tests/geodata/core` and `tests/geodata/viz` suites are the guard.

### Task 2: Lazy GeoTIFF writers in `GeoArray`
Move `write_cog` and `write_gtiff` imports into `to_cog` and `to_gtiff`. Add a test that importing `geosave_engine.geodata.core` leaves `geosave_engine.geodata.utils.io.geotiff` out of `sys.modules`. Expected: RED, then GREEN.

### Task 3: `array()` through `raster()`
`array` returns `raster({"pixels": pixels}, geobox, nodata=nodata, **coords)["pixels"].rename(None)`. Add a test that `array(x, g, time=t)` equals `raster({"a": x}, g, time=t)["a"]` apart from the name. Run the array tests and update the error-message expectations.

### Task 4: The `gs.rebase` accessor takes all three forms
Failing tests: `ds.gs.rebase(attrs.merge([ds, ds]))` restores the header, and `ds.gs.rebase(ds.gs.attrs.data_vars["red"], target="red")` patches it. Then the accessor's signature and overloads mirror `attrs.rebase`.

### Task 5: Shared raster accessor and `crop` in `transform.vector`
- Failing test: `ds["red"].gs.crop(vector)` crops and masks a DataArray.
- Move the crop logic into `transform.vector.crop(data, vector, *, mask=True)`.
- Add `GeoRasterAccessor[DataT]` in `core/base.py` with `unpack`, `mask`, `to_nan`, `reproject`, and `crop`, each delegating to `transform`.
- `GeoRaster` and `GeoArray` inherit it and drop their copies.

### Task 6: Verification
Run `uv run pytest -q`, `uv run ruff check src tests`, basedpyright on `core`, and the docstring checker on touched files.

# GeoVector conventions Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task.

**Goal:** Normalize spatial records consistently and simplify raster catalog I/O.

**Architecture:** GeoVector owns spatial record conventions and Item conversion.
Readers/factories invoke that contract. Existing STAC Item construction owns
identity bindings; catalog I/O uses actual physical writer inventories.

**Tech Stack:** GeoPandas, pandas, PySTAC, stac-geoparquet, xarray, existing I/O.

**Spec:** ../specs/2026-10-07-geovector-conventions.md

## Global constraints

Preserve native objects, laziness, windows and file-close ownership. No new table
module, dependencies, invented timestamps, compatibility aliases or commits.
Execute in the current workspace containing earlier authorized changes.

## Review focus

- Geometry-only and empty records must have missing time without losing selection.
- Mixed instants/ranges/undated records and renamed geometry retain their meaning.
- STAC JSON must preserve nested assets, relative hrefs and top-level fields.
- External date mappings must not silently overwrite canonical dates.
- Writer inventories must preserve sparse/time-independent stack groups and order.

## Task 1: Spatial record conventions

Files: core/vector.py; tests/geodata/core/test_vector.py.
Produces: GeoVector.normalize(frame, *, datetime_column=None), .timespan,
.to_anchor(resolution=None, shape=None, pad=0, crs=None, anchor='edge', timespan=None).

- [x] Add failing tests for UTC geometry properties, undated query selection,
  explicit date mapping, empty IDs discovery, and row/selection anchor coverage.
- [x] Observe failures with pytest tests/geodata/core/test_vector.py.
- [x] Implement shared normalization, time bounds/query and anchor delegation.
- [x] Run the focused tests and retain native index, CRS and geometry names.

## Task 2: Item conversion and reader consistency

Files: core/vector.py; io/geojson.py, geoparquet.py, geopackage.py, readers.py;
tests/geodata/io/test_geojson.py; tests/geodata/core/test_vector.py.
Produces: GeoVector.to_items() -> tuple[pystac.Item, ...]; reader date mapping.

- [x] Add failing tests for Item metadata/JSON round trips, undated export refusal,
  geometry-only explicit dates and IDs, relative STAC hrefs and filtered reading.
- [x] Observe failures; implement native Item conversion and STAC JSON I/O.
- [x] Normalize direct readers and file dispatch, with datetime_column explicit.
- [x] Verify plain GeoJSON/GeoParquet/GeoPackage and STAC JSON round trips.

## Task 3: Catalog simplification and names

Files: io/catalog.py, stac/item.py, core/raster.py, core/stack.py, io/__init__.py;
tests/geodata/io/test_catalog.py; existing catalog/STAC/core tests.
Produces: catalog.write(data, path, *, driver='cog', collection=None, id=None,
**options) and describe(paths or group->paths, *, id=None, collection=None).

- [x] Add failing tests for describe without rewriting and write/read identity.
- [x] Replace export helpers with native inventories and one description flow.
- [x] Move Item bindings into existing stac/item.py, preserving order and scenes.
- [x] Migrate io.catalog.to_items callers; retain raster accessor conveniences.
- [x] Verify catalog, stack, STAC and source-cleanup tests across all formats.

## Task 4: Documentation and final verification

Files: architecture guide, relevant public docstrings and current examples.

- [x] Explain record conventions, missing time, anchors, JSON and write/describe.
- [x] Run focused vector/I/O/core/ML checks, Ruff, scoped type checks and full pytest.
- [x] Obtain fresh read-only review; fix important findings with regressions.
- [x] Record results and API removals; git diff --check and inspect fixture status.

## Execution ledger

Tasks 1-3 implemented with failing tests observed before production changes.
Fresh read-only review found nullable asset conversion, duplicate selected
geometry, and nanosecond acquisition collisions. All reproduced and fixed with
regressions. Native STAC conversion has microsecond-only paths; retain UTC values
and restore their precision around those library calls rather than replace the
library's property/asset conversion.

Ruling: scalar COG acquisitions now receive timestamp Item IDs like dimensional
COG acquisitions. Inventory description reads temporal meaning from native
headers, independently of the original xarray dimension shape. This is an
intentional alpha naming change; existing files and raster values are unchanged.

Ruling: normalize missing/empty asset cells to None, and supply {} during Item
conversion. This preserves ordinary GeoParquet persistence without empty structs.
Work remains in the shared workspace, without committing earlier user changes.

The first full-suite run exposed store date-coverage and missing-assets error
regressions (1632 passed, four failed). Store description now retains whole-date
and resampled coverage while preserving nanosecond boundary labels. Raster
loading explicitly reports a missing assets column, independently of identity
discovery. The focused regression suite passed: 87 passed, one deselected.

Final verification: 1636 passed, 16 deselected in the complete pytest suite.
Ruff passed for src/tests; scoped BasedPyright reported zero errors or warnings;
git diff --check passed. Fresh read-only review found no remaining important
regressions. Tracked test fixtures were unchanged. No commits were created.

Breaking changes: io.catalog.to_items is replaced by write/describe; scalar COG
Item IDs include acquisition timestamps; ordinary vector readers/factories now
include a UTC datetime column with NaT for unknown dates.

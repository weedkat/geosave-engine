# Raster Catalog Implementation Plan

**Goal:** Save and reopen Dataset rasters through a shared STAC GeoParquet catalog.
**Architecture:** PySTAC owns Items; stac-geoparquet owns serialization. The existing
GeoVector accessor exposes discovery/loading, backed by format-aware catalog I/O.
**Spec:** ../specs/2026-10-06-cog-leaves-catalog-design.md

## Constraints

Preserve unrelated workspace edits. Keep stack and row APIs working. Default IDs
use gs.anchor.stem. COG acquires one row per date; NetCDF/Zarr use a complete-store
row. Keep lazy arrays, metadata, precise dates, explicit dtype/grid invariants,
relative hrefs, deferred publication, and stable upserts.

## Tasks

- [x] Add failing behavioral tests in tests/geodata/io/test_catalog.py and
  tests/geodata/core/test_catalog.py for all three formats, discovery, filtering,
  append/replace, moved catalogs, default IDs, deferred writes and failure cleanup.
- [x] Replace manual STAC assembly in geodata/stac/item.py and metadata stamping
  in geodata/io/geoparquet.py with library conversions; migrate callers/tests of
  the removed manual columns. Verify edited/null/heterogeneous frames.
- [x] Implement geodata/io/catalog.py for Dataset publication, COG acquisitions,
  compatibility validation, lazy reconstruction and resource closure. Expose
  raster_ids/to_raster on core/vector.py and wire Dataset writers in core/raster.py.
  Keep stack COG writing on its existing low-level layout writer.
- [x] Migrate old Dataset COG call sites and tests, add guide examples, and run
  focused persistence/catalog tests, lint/type checks and the complete suite.
- [x] Review the final changes and address correctness findings.

## Review focus

Catalog writes must preserve unrelated rows; changed grids/bands/formats must fail
before pixels are replaced. Scalar vs dimensional time and subsecond identity
must survive. Edited/reordered frames must have current bbox metadata. Failed or
deferred pixel writes must not publish. Loaded arrays must close every file.

## Execution notes

User explicitly requested implementation after approving the design. Work stays
in the existing checkout because it contains required in-progress geodata changes;
no unrelated edits will be reverted or committed.

## Verification ledger

- Behavioral tests first exposed colliding COG paths, missing discovery, and
  missing NetCDF catalog arguments. Dataset catalog tests now pass (22 cases).
- First full suite: 1465 passed, 3 failed. Fixed native asset-list normalization
  for workflow validation and removed the obsolete manual column-order assertion.
  Targeted vector/workflow regression run: 111 passed.
- Independent review found encoded-dtype upsert and NetCDF backend propagation
  defects. Both reproduced in tests and fixed; their regression tests pass.
- Added passing failure/cleanup, incompatible schema, cross-ID destination
  ownership, and same-ID format-change tests. No extra compatibility aliases.
- Ruling: preserve stack saving through its existing low-level layout writer;
  the Dataset API deliberately changes destination semantics. Stack grouping
  remains deferred by the approved spec.
- Ruling: leave user changes and implementation uncommitted in this shared dirty
  checkout. No unrelated files were reverted.
- Scoped BasedPyright: 0 errors/warnings. Ruff src/tests and diff checks pass.
- Final full suite: 1474 passed, 26 deselected, 92 warnings in 188.35 seconds.
  Additional metadata assertions: all 22 Dataset catalog tests pass.

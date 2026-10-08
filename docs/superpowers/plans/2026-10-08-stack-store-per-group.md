# One Store Per Stack Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Save a stack's Zarr or NetCDF as one store per group, so every asset opens by its href and nothing has to name a group inside a store.

**Architecture:** `stack.gs.to_zarr` and `to_netcdf` write `<destination>/<group>.zarr` or `.nc` and return `{group: path}`. `read_stack` reads a folder or a mapping. The format modules write and read one Dataset only.

**Tech Stack:** xarray, existing `io.zarr`, `io.netcdf`, `io.readers`.

**Spec:** This document. Design agreed in conversation 2026-10-08.

## Design

```python
paths = sample.gs.to_zarr("samples/s0")
# {'optical': PosixPath('samples/s0/optical.zarr'), 'label': PosixPath('samples/s0/label.zarr')}

items = stac.create_stack_items(paths, name="s0")     # one Item per group
sample = read_stack("samples/s0")                     # folder of stores, as today
```

```text
before                     after
s0.zarr/                   s0/
  optical/                   optical.zarr
  label/                     label.zarr
```

Why: the group selector was the only reader option that had to be stored in a
catalog. It rode in `xarray:open_kwargs`, from an extension its own repository
marks deprecated, and the Zarr extension that replaces it has no group
selector. A stack Zarr also lost its group order on read.
`workflow/tasks/sample.py` already writes one raster per group.

## Global Constraints

- `stack.gs.to_zarr` / `to_netcdf` return `dict[str, Path | str]`, or a delayed task resolving to it when `compute=False`. One path per group, not a tuple: `read_stack` joins a tuple along `time`, which would turn a scalar `time` into a dimension.
- `io.zarr.write` / `io.netcdf.write` take a Dataset only; `read` takes no `group`; `read_stack` is removed from both modules.
- `read_stack` on a `.zarr` or NetCDF path raises and points at `read_raster`.
- No `xarray:open_kwargs` is written or read.
- No compatibility aliases. No commits.
- Baseline: `uv run pytest` gives 1747 passed.

## Review Focus

1. A stack's group order survives a round trip through the returned mapping; a folder reads in name order. (Task 1)
2. `to_zarr(..., compute=False)` writes nothing until computed, then yields the same mapping. (Task 1)
3. A group that already exists refuses without `overwrite=True`, and no other group is left half written. (Task 1)
4. A dated group and a timeless group round trip through the table and load back as a stack. (Task 3)
5. `read_stack("x.zarr")` fails with a message naming `read_raster`. (Task 2)

---

### Task 1: Stack writers write one store per group

**Files:** Modify `src/geosave_engine/geodata/core/stack.py`; Test `tests/geodata/core/test_stack.py`.

- [x] Failing tests: `to_zarr("s0")` and `to_netcdf("s0")` return `{group: path}` in stack order with files at `s0/<group>.zarr|.nc`; `read_stack("s0").gs.groups` equals the written order with equal pixels; an existing group refuses; `compute=False` writes nothing until computed.
- [x] Implement over `self.rasters` and the Dataset writers.
- [x] `uv run pytest tests/geodata/core/test_stack.py -q`

### Task 2: Format modules and `read_stack` drop grouped stores

**Files:** Modify `src/geosave_engine/geodata/io/raster/zarr.py`, `netcdf.py`, `src/geosave_engine/geodata/io/readers.py`; Tests `tests/geodata/io/raster/test_zarr_stack.py`, `test_zarr.py`, `test_netcdf.py`, `tests/geodata/io/test_readers.py`.

- [x] Failing test: `read_stack("scene.zarr")` raises `ValueError` matching `read_raster`.
- [x] Remove the DataTree write path, `read_stack`, the `group` parameter and the grouped branch of `fill_encoding`; delete tests of grouped stores and keep those that test a folder of stores.
- [x] `uv run pytest tests/geodata/io -q`

### Task 3: STAC drops the group selector

**Files:** Modify `src/geosave_engine/geodata/stac/item.py`, `table.py`; Tests `tests/geodata/stac/test_item.py`, `test_table.py`, `tests/geodata/io/test_remote.py`, `tests/ml/segmentation/supervised/test_data.py`.

- [x] Failing test: a stack saved with `to_zarr` gives one Item per group whose asset has no `xarray:open_kwargs`, and each group loads through `table.load`.
- [x] `create_stack_items` takes the mapping only; `_Saved` and `DataAsset` lose their options; the open-options check goes.
- [x] `uv run pytest`, `uv run ruff check src tests`, BasedPyright and `scripts/check_docstrings.py` on the changed modules; update `docs/guides`.

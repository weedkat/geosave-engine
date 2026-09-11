# Core module sweep — session summary

Sweeping `geodata/core` for bugs and stale code, module by module. `anchor`,
`array`, and `vector` were frozen before this session. **This session settled
`GeoRaster` (`core/raster.py`).** Next up is `GeoStack` (`core/stack.py`).

---

## READ THIS FIRST: 51 tests fail on purpose

```
51 failed, 227 passed
  41  tests/geodata/transform/test_time.py
   5  tests/geodata/transform/test_grid.py
   4  tests/geodata/test_stack.py
   1  tests/geodata/test_core_smoke.py
```

**Every one traces to a single deleted property.** `GeoRaster.categorical` was
removed as flawed (see Decisions). Its three callers all live in `transform/`,
which is a **dead module pending rework** — the user's explicit instruction was
to delete from core and *let transform break* rather than patch it.

```
src/geosave_engine/geodata/transform/time.py:194: AttributeError:
    'GeoRaster' object has no attribute 'categorical'
```

The 4 `test_stack.py` and 1 `test_core_smoke.py` failures are **not** stack or
core defects — those tests build fixtures by calling `transform.time.resample`,
so they inherit the breakage. Core-only run is green:

```bash
pytest tests/geodata tests/ml -q --ignore=tests/geodata/transform
# 151 passed
```

Use that as the working check until transform is reworked.

---

## Ground rules established this session

- **`transform/` is dead.** Treat it as stubs to be rewritten. Do **not** patch
  it to keep core clean — let it break. (I violated this once by adding a helper
  to `transform/variables.py`; it was reverted.)
- **`stitcher.py` is shelved** pending the tiling redesign, but keep its tests
  passing so real regressions stay visible.
- **Names come from the package, the wrapped library, or the domain** — never
  invented to fill a gap. Fix a vague name with a *truer* word, not a longer one.
- **Don't add API for one caller.** Prefer deleting over moving/wrapping.
- **Measure before claiming.** Several "obvious" fixes were wrong until measured.

---

## What changed

### `core/raster.py` (1216 → 1205 lines)

**Model IO — `to_array` / `to_numpy` / `to_tensor` / `_stacked`**

- Deduplicated into one private `_stacked(var_names, dtype)` that owns the whole
  contract: absent names, empty selection, band-axis collision, grid coverage,
  axis agreement, dtype agreement, layout.
- **Fixed silent data loss:** `to_array` destroyed the Dataset's own attrs.
  It called `attrs.combine([ds[n] for n in names])` — a *cross-object* join — to
  do an *intra-object* one, then `stamp`ed the result over everything. ACDD
  `title`/`institution` vanished on the way to `to_cog`. Now it merges only the
  selected variables' namespaces onto the root that `Dataset.to_array()` already
  preserves. Side effect: **1390 µs → 634 µs**.
- **Fixed silent broadcast:** a variable with no `y`/`x` (e.g. `sun_elevation`
  with dims `('time',)`) was broadcast across every pixel and stacked as a band.
  The old check compared only *non-grid* dims, so `('time',)` vs `('time',)`
  matched. Now `ungridded` variables are refused.
- **Fixed `IndexError`:** `to_array([])` / `to_numpy([])` / `to_tensor([])` died
  on `var_names[0]` with `tuple index out of range`. Now an actionable error.
- **Unified dtype:** `to_array` silently promoted mixed dtypes (measurably lossy:
  `int64(2**53+1)` + float32 → `2**53`) while `to_numpy` refused the same input.
  Both now refuse and both accept `dtype=`.
- **Fixed `to_tensor` dead end:** on mixed dtypes it raised "pass dtype=", but its
  `dtype` is torch-side and never reached `to_numpy`, so the advice was
  unfollowable. Now forwards the numpy equivalent; `bfloat16` (no numpy
  counterpart) stacks as float32 and narrows.
- **Layout changed** — see Decisions.

**Readers**

- `timespan` 262 → 90 µs, `categorical` 167 → 75 µs. Both parsed all 10
  registered models across every variable and coord to read **one** model off
  **one** of them.
- `categorical` **deleted** (see Decisions).
- `times` **kept** — zero internal callers, but it's public API and returns
  `None` for timeless data, which is what lets `timespan` compose with it.

**Attrs writers**

- `write_nodata` collapsed to a single `rebase` call after `Packing` gained the
  mirror. Both fill-value constants (`_FillValue`, `nodata`) are now gone from
  raster.py entirely.
- `write_crs`, `rebase` — inspected, unchanged. See Non-defects.

**`plot`**

- Three near-identical `draw(...)` calls → one. The chain now resolves *which
  variables*; *how to draw* happens once after it.
- Dropped two redundant `isinstance` checks — `.get(Model)` is overloaded to
  return `Model | None`, so they guarded a narrowing the type already gives.
- `Legend` now read from the variable's own namespace, not the stacked array's
  root (one namespace instead of a full header read).

**Misc**

- Deleted dead `pbar` param, a no-op `cast`, and `_leading_dims` (a helper whose
  name asserted a position it never checked).
- Fixed stale doctest `tile_id` → `tile_index`.
- `to_cog` was constructing *both* layouts on every call to use one.

### `attrs/models/packing.py` — `nodata` mirror

`Packing` now owns **both** spellings and keeps them in sync, modelled on
`Legend._sync_flags`:

```python
Packing(fill_value=0)   -> {'_FillValue': 0, 'nodata': 0}
Packing(nodata=9)       -> {'_FillValue': 9, 'nodata': 9}   # reads odc-only sources
Packing(fill_value=None)-> {'_FillValue': None, 'nodata': None}   # rebase pops both
{'_FillValue':0,'nodata':9} -> ValidationError
```

**Why:** odc reads `nodata` *before* `_FillValue`
(`NODATA_ATTRIBUTES = ("nodata", "_FillValue")`). With both present and
disagreeing, `odc` saw 9 while `Packing.fill_value` said 0 — so `crop(mask=True)`
filled with the wrong value. The divergence is now unrepresentable.

**Gotcha:** the recursion guards must use `values_agree`, not `!=`. `nan != nan`
is always true, so a NaN fill re-assigns forever under `validate_assignment=True`
(`RecursionError`). `Legend._sync_flags`'s odd-looking `if x != y:` guards are
the same defence — they are load-bearing, not stylistic.

### `attrs/xarray.py` — `read()`

Was building a whole `DataArray` per coordinate just to read its `.attrs` dict.
Now reads the `Variable` mapping: **8.7× cheaper** on Dataset, **12.6×** on
DataArray, identical output. Helps every attrs read in the library.

Note: the *write* path (`rebase`/`stamp`) still uses `obj[name]` deliberately —
`Variable.copy(deep=False)` shares the attrs **dict object**, so rebinding
(`.attrs = {...}`) is safe but mutating (`.attrs.pop(...)`) leaks into the
caller's Dataset.

### `band` vocabulary (4 files)

One spelling now, `BAND_DIMENSION` in `core/array.py`:

| was | |
|---|---|
| `BAND_DIMENSION` in io/gdal (12 uses: gdal, geotiff, layout) | the incumbent name |
| `_VARIABLE_DIMENSION` in core/raster (4) | → imported |
| `_CHANNEL_DIMENSION` in core/array (1) | → renamed |
| bare `"band"` ×3 in viz/plot | → imported |

Home is `core/array.py` because it already owns `UNPLACED_DIMENSIONS`,
`_CRS_COORDINATE`, `_TIME_COORDINATE`. Layering verified: `io/dispatch.py`
already imports `core.vector` at module level, and `core/array.py` reaches io and
viz only lazily inside methods. Also removed a re-export chain — geotiff and
layout were getting the constant via `from .gdal import`.

### Other files

- `core/stitcher.py` — `_close` unflattened band-first; fixed for the new layout.
- `core/stack.py`, `core/array.py` — layout docstrings; `colorize` moveaxis.
- `tests/geodata/test_model_io.py` (3), `tests/geodata/attrs/test_header_stamping.py` (2)
  — assertions updated to deliberate new behaviour, not worked around.

---

## Decisions (please don't re-litigate)

**Model-input layout is now `(*axes, band, y, x)`, i.e. torchgeo's `[T, C, H, W]`.**
Previously `(band, *axes, y, x)`. torchgeo — the project's declared ML dependency
— uses `[T, C, H, W]` across ~12 dataset modules. `nn.Conv3d` does require
`(N, C, D, H, W)` (verified: `(N,T,C,H,W)` raises), but that's a *model-boundary*
concern fixed by one `permute`, which is exactly why torchgeo ships datasets the
other way. Timeless rasters are unchanged at `(band, y, x)`.

**`GeoRaster.categorical` deleted.** All three callers wanted a *guard*
("would this kernel blend class codes?"), not a reader, and it smuggled a dtype
validation into attribute access — firing on every read, far from the `rebase`
that wrote the bad `Legend`. A user wanting classes already has
`ds.gs.attrs.data_vars[name].get(Legend).class_map`.

**`to_cog` meaning two things is fine.** `GeoRaster.to_cog` takes a `Layout` and
returns `None`; `GeoArray.to_cog` takes a path and returns `Path`. Different
input structures need different mechanisms; both produce COGs.

**`times` stays.** Zero internal callers is weak evidence for a *public* reader.

---

## Open items

**Two decisions for the user:**

1. **The Legend-on-float check has no home.** Deleting `categorical` removed the
   refusal of a `class_map` on a non-integer variable. The check was real; its
   right home is validation *when a Legend is written* (`rebase` or `Legend`),
   not a reader. No test covered it.
2. **`encoding["grid_mapping"]` is dropped by `to_array`.** The `spatial_ref`
   coord survives and the geobox resolves, so odc is fine, but the CF pointer
   dangles — and `raster.py`'s module docstring cites it as part of what "placed"
   means.

**Frozen until `transform/` is reworked:**

- 7 delegating methods (`reproject`, `resample`, `interpolate`, `rename_vars`,
  `tile`, `time_window`) + `crop`. Their `cast("Dataset", ...)` calls are
  **debt markers, not narrowing** — the transform functions under-declare their
  return as `xr.Dataset`. When transform is rewritten, annotate it `-> Dataset`
  and the casts delete themselves.
- `crop` does its odc call **inline** while the other seven delegate.
- 3 dangling `GeoRaster.categorical` call sites (`grid.py:229`, `time.py:194`,
  `time.py:300`).
- `odc.reproject` re-stamps `nodata` alongside `_FillValue` (`_xr_interop.py:1003`)
  and float-coerces both. No longer a correctness bug now that `Packing` mirrors
  them, but still worth handling.
- The three duplicated `_declared_fill` helpers (transform/tiles, transform/grid,
  stitcher) are now redundant — `Packing.fill_value` answers for either spelling.

**Elsewhere:**

- `core/__init__.py` re-exports `StitchWindow` from `.stitcher`, which doesn't
  define it (it lives in `attrs.models.tiling` and is *also* exported from
  `attrs`). Two public paths, one type.
- `Tiling.count`, `.padded_shape`, `.bounds`, `.tiles()`, `.merger()` have **zero
  call sites** anywhere — behaviour added ahead of a consumer. The stitcher was
  the intended consumer and doesn't use them.

---

## Non-defects (checked, leave alone)

- **`GeoRaster.rebase`'s `if inplace:` branch is correct**, not a pass-through
  pretending to be logic. mypy confirms a bare `bool` matches neither
  `Literal[True]` nor `Literal[False]`, so the branch narrows to a literal per
  call. Collapsing it would force a cast.
- **`to_tensor`'s `torch.empty(0, dtype=...).numpy()` probe** is a capability
  query, not swallowed error handling. Measured at 0.5% of the smallest realistic
  call, and it's fixed cost against pixel-proportional work.
- **xarray's own stacking is not the bottleneck.** Hand-rolled `np.stack` was
  *slower* in 2 of 3 measured cases; memory copy dominates.

---

## Next: `GeoStack` (`core/stack.py`, 539 lines)

Flagged in the original sweep, not yet investigated in depth:

- `insert` vs `merge` — does `insert` earn its place? Both duplicate the grid
  check, and `stack()` already builds. Deleting it removes ~33 lines.
- `stack()` names a local `anchor` for a Dataset — `GeoAnchor` is a domain type
  in the same package.
- `_GRID_MAPPING_COORDINATE = "spatial_ref"` duplicates `core/array.py`'s
  `_CRS_COORDINATE`. Same collapse as `BAND_DIMENSION`.
- `stack()` picks `time_bnds` from the *first* group that has one — arbitrary.
- `_cast_for[T]` — 14-line generic helper used twice for
  `dtype.get(group) if Mapping else dtype`.
- Its `cast(...)` calls are the same species as raster's; most should dissolve
  when transform is annotated.

Expect the same pattern that held all session: **every vague name had a real bug
behind it.** `carried` → silent pixel broadcast. `names[0]` → `IndexError`. The
`stamp` call → destroyed Dataset attrs.

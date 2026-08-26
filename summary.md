# Session Handoff — GeoHeader / GeoExtension unification

Session collapsed `GeoAttrs` (pydantic model, mixed dedicated fields + open plugin bag) into `GeoHeader` (plain dataclass, one field, every namespace a `GeoExtension`). Fundamental layer only — `extensions/`, `spatial/header.py`, `spatial/anchor.py`, `spatial/_array.py`. All tested, all passing. `spatial/raster.py`/`mosaic.py`/`stack.py`/`stitch.py` **not** touched beyond mechanical import fixes, confirmed broken — see §3.

Full writeup with hook-contract table and construction-flow diagram: `docs/concept/geoheader.md`.

---

## 1. What changed

**`GeoAttrs` → `GeoHeader`.** Not pydantic — plain `@dataclass(frozen=True)`, one field: `extensions: dict[str, GeoExtension]`. `start_datetime`/`end_datetime`, `tags`, `tiling`, `timespec` no longer dedicated fields — each real registered `GeoExtension` subclass (`TimeSpan`, `Tags`, `TilingInfo`, `TimeSpec`) living in same dict as `render`/`stac`. No special-cased fields left anywhere in header.

**One hook contract, five methods, on every `GeoExtension`:** `decode(value)`, `check(data)`, `combine(values)`, `encode()`, `SETTABLE`. Each defaults sensibly; subclass overrides only where semantics differ (full matrix in doc). `check(data)` — was `validate_against_data`, renamed: collided with real inherited `pydantic.BaseModel.validate` classmethod (deprecated v1 API, still present), different signature/return type, genuine LSP violation caught by type checker, not style nit.

**`GeoHeader(extensions, data=data)`** — `InitVar`-based constructor. Positional dict construction works (`GeoHeader({"tags": {...}})`). When `data` given, every namespace's `check(data)` runs during construction — no separate `.revalidate()` call needed. Idempotent: twice on already-consistent pair = no-op both times. `_SpatialArray.__post_init__` now calls this once; no more inline per-namespace loop.

**Eager stamp.** Every `_SpatialArray` construction (every transform — `rebase`, `crop`, `reproject`, etc. all end in `dataclasses.replace()`, reruns `__post_init__`) writes `encode_attrs(data.attrs, anchor.header, "json")` back onto `data.attrs`. Reason: caller reading `.data` directly (interop, plotting, `xr.concat` outside library) used to see empty dict — `.attrs` carried no header in memory, only stamped at disk write. Now `data.attrs` always mirrors `anchor.header`, recomputed fresh each time (not incrementally patched), so direct external mutation of returned dict still silently diverges on next read — accepted, same "consenting adults" limitation as mutating array off nominally-frozen object.

**Per-namespace on-disk encoding.** No more single reserved `"geosave"` key holding one opaque blob. Every namespace present gets own top-level attrs key — `data.attrs["tags"]`, `data.attrs["render"]`, etc. — nested dict for JSON-capable stores (zarr), one JSON string for text-only (GDAL/netCDF). Tradeoff accepted, not fixed: unregistered/stale namespace data now indistinguishable from coincidental foreign attr on decode; old design could warn on that, new one can't.

**`SETTABLE` guard.** `TimeSpec.SETTABLE = False` only one. `GeoHeader.rebase()` rejects with one clear message. Fixes two bugs: known misleading `GeoAnchor.rebase(timespec=...)` → `UnknownExtensionError` (old design, `_RESERVED_NAMESPACES` conflated "can't register" with "can't set"), and new one found this session — `GeoRaster.rebase(timespec=...)` silently succeeding by leaking through `**extensions` into `_SpatialArray._rebase`'s own explicit param. `_with_timespec` (private) is sanctioned bypass — only `resample_time`/`concat` reach it.

**Renames.** `attrs` field/property → `header`, on both `GeoAnchor` and `_SpatialArray` (collided with xarray's own `.attrs` dict — two different things, same name, sibling objects). `GeoAnchor.time` property and `rebase(time=...)`/`from_bbox(time=...)`/`from_coordinate`/`from_vector`/`from_geometry` kwarg → `timespan`, matching `TimeSpan`'s name; `TimeSpan.NAMESPACE` renamed `"time"` → `"timespan"` (changes on-disk key). **Not done, deliberately deferred:** `_SpatialArray.time`/`.start`/`.end` (the `GeoRaster`/`GeoTile`-level property names) still say `.time`. `explore(time=...)` (pick one instant) and `data.isel(time=...)` (xarray's own dim name) unrelated, correctly untouched.

---

## 2. Tests

New `tests/geodata/spatial/test_header.py` — 28 tests + 1 perf guard (`@pytest.mark.slow`), all pass. Covers: `rebase()` merge/replace/drop/`SETTABLE` semantics, positional construction, `combine()` per extension (`TimeSpec` equal-keep/else-None, `TilingInfo`/`TimeSpan` always-None, default equal-or-raise), encode/decode round-trip (both encodings, foreign-key survival, empty-namespace omission), `check()` hook end-to-end through real `GeoAnchor.to_geotile()` construction, idempotency, `timespan` rename. `conftest.py`'s `make_anchor` needed one line fixed (`rebase(time=...)` → `rebase(timespan=...)`) — would break every fixture-dependent test otherwise.

1000 `GeoTile` constructions (2 extensions each, eager stamp included): **~2.5ms/construction**, no perf red flag from eager-stamp change.

---

## 3. Confirmed broken — `spatial/raster.py` (not fixed this session)

Full `tests/geodata/spatial/` suite: **141 failed, 62 passed, 23 errors.** Root cause one bug, not scattered regressions: `GeoRaster.rebase()` (raster.py, still pre-refactor signature — explicit `tags=`/`tiling=` params) unconditionally forwards those params — even at `UNSET` default — into `_SpatialArray._rebase()`, which no longer declares `tags=`/`tiling=` at all. They land in `_rebase`'s `**extensions`, get forwarded again into `GeoAnchor.rebase(**extensions)`, land in *its* `**extensions` too, and `GeoHeader.rebase()` tries `{**current.model_dump(), **UNSET}` — `TypeError`, `UNSET` isn't a mapping. Every test using `make_raster()` fixture hits this, since `make_raster` always calls `.rebase(nodata=...)`. `GeoRaster.rebase()` needs signature brought in line with `GeoAnchor.rebase()`/`_SpatialArray._rebase()` (drop `tags=`/`tiling=`, rename `time=` → `timespan=`) before anything downstream usable.

Also broken, not yet reached: `open()`/`_write_tiff()`/`_cf_encoded()` (old `HEADER_KEY`/`GeoAttrs` references), and once fixed, `_concat_band()`/`GeoMosaic.result()` need switch from public `.rebase()` to `._rebase()` + pop `timespec` for private bypass — same pattern `_concat_time()` already needs, since `combine_extensions`/`GeoHeader.combine` folds `timespec` uniformly and `SETTABLE` guard rejects it through public path.

---

## 4. Open for next session

1. **Fix `spatial/raster.py`.** Bring `GeoRaster.rebase()` in line with new `GeoAnchor.rebase()`/`_SpatialArray._rebase()` shape, fix `open()`/write paths to use `GeoHeader`/`encode_attrs`/`decode_attrs` new signatures, fix `_concat_time`/`_concat_band`/`GeoMosaic.result()` per §3. Same for `mosaic.py`/`stack.py`/`stitch.py`.
2. **`Tags.combine()`** has no override — composing rasters with differing tags will raise (default equal-or-raise) once composition wired back up, vs today's silent first-wins. Undecided: accept raise, or add dict-union override.
3. **`TimeSpan.from_input`** bespoke method, not part of five-hook contract every other extension follows. Tentative: promote to formal hook, default = defer to `decode`, since no other extension needs real translation logic.
4. **`_SpatialArray.time`/`.start`/`.end`** → `.timespan`/`.start`/`.end`, property-name half of rename `GeoAnchor` already got. Trivial, deliberately deferred, touches existing tests (`raster.time` assertions) still passing under old name.
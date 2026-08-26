# Session Handoff — GeoHeader / GeoExtension unification

This session collapsed `GeoAttrs` (pydantic model, mixed dedicated fields + open plugin bag)
into `GeoHeader` (plain dataclass, one field, every namespace a `GeoExtension`). Fundamental
layer only — `extensions/`, `spatial/header.py`, `spatial/anchor.py`, `spatial/_array.py`. All
tested, all passing. `spatial/raster.py`/`mosaic.py`/`stack.py`/`stitch.py` were **not**
touched beyond mechanical import fixes and are confirmed broken — see §3.

Full writeup with a hook-contract table and a construction-flow diagram:
`docs/concept/geoheader.md`.

---

## 1. What changed

**`GeoAttrs` → `GeoHeader`.** No longer pydantic — a plain `@dataclass(frozen=True)` with one
field, `extensions: dict[str, GeoExtension]`. `start_datetime`/`end_datetime`, `tags`, `tiling`,
`timespec` are no longer dedicated fields — each is a real, registered `GeoExtension` subclass
(`TimeSpan`, `Tags`, `TilingInfo`, `TimeSpec`) living in the same dict as `render`/`stac`. No
special-cased fields left anywhere in the header.

**One hook contract, five methods, on every `GeoExtension`:**
`decode(value)`, `check(data)`, `combine(values)`, `encode()`, `SETTABLE`. Each defaults
sensibly; a subclass overrides only where its own semantics differ (full matrix in the doc).
`check(data)` — was `validate_against_data`, renamed: that name collided with a real inherited
`pydantic.BaseModel.validate` classmethod (deprecated v1 API, still present), different
signature/return type, genuine LSP violation caught by the type checker, not a style nit.

**`GeoHeader(extensions, data=data)`** — an `InitVar`-based constructor. Positional dict
construction works (`GeoHeader({"tags": {...}})`). When `data` is given, every namespace's
`check(data)` runs as part of construction itself — no separate `.revalidate()` call needed
anywhere. Idempotent: calling it twice on an already-consistent pair is a no-op both times.
`_SpatialArray.__post_init__` now just calls this once; no more inline per-namespace loop.

**Eager stamp.** Every `_SpatialArray` construction (which is *every* transform — `rebase`,
`crop`, `reproject`, etc. all end in `dataclasses.replace()`, which reruns `__post_init__`) ends
by writing `encode_attrs(data.attrs, anchor.header, "json")` back onto `data.attrs`. Reason: a
caller reading `.data` directly (interop, plotting, `xr.concat` outside this library) used to see
an empty dict — `.attrs` carried no header in memory, on purpose, only stamped at actual disk
write. Now `data.attrs` always mirrors `anchor.header`, still recomputed fresh each time (not
incrementally patched), so a direct external mutation of the returned dict still silently
diverges on the next read — accepted, same "consenting adults" limitation as mutating any array
pulled off a nominally-frozen object.

**Per-namespace on-disk encoding.** No more single reserved `"geosave"` key holding one opaque
blob. Every namespace present gets its own top-level attrs key — `data.attrs["tags"]`,
`data.attrs["render"]`, etc. — nested dict for JSON-capable stores (zarr), one JSON string for
text-only ones (GDAL/netCDF). Tradeoff accepted, not fixed: an unregistered/stale namespace's
data is now indistinguishable from a coincidental foreign attr on decode; the old design could
warn on that, the new one can't.

**`SETTABLE` guard.** `TimeSpec.SETTABLE = False` is the only one. `GeoHeader.rebase()` rejects
it with one clear message. Fixes two bugs at once: the known misleading
`GeoAnchor.rebase(timespec=...)` → `UnknownExtensionError` (old design, `_RESERVED_NAMESPACES`
conflated "can't register" with "can't set"), and a new one found this session —
`GeoRaster.rebase(timespec=...)` silently succeeding by leaking through `**extensions` into
`_SpatialArray._rebase`'s own explicit param. `_with_timespec` (private) is the sanctioned
bypass — only `resample_time`/`concat` reach it.

**Renames.** `attrs` field/property → `header`, on both `GeoAnchor` and `_SpatialArray` (was
colliding with xarray's own `.attrs` dict — two different things, same name, on sibling
objects). `GeoAnchor.time` property and the `rebase(time=...)`/`from_bbox(time=...)`/
`from_coordinate`/`from_vector`/`from_geometry` kwarg → `timespan`, matching `TimeSpan`'s own
name; `TimeSpan.NAMESPACE` itself renamed `"time"` → `"timespan"` to match (changes the on-disk
key). **Not done, deliberately deferred:** `_SpatialArray.time`/`.start`/`.end` (the
`GeoRaster`/`GeoTile`-level property names) still say `.time` — same rename, one level up, not
yet applied. `explore(time=...)` (pick one instant) and `data.isel(time=...)` (xarray's own dim
name) are unrelated and correctly untouched throughout.

---

## 2. Tests

New `tests/geodata/spatial/test_header.py` — 28 tests + 1 perf guard (`@pytest.mark.slow`), all
pass. Covers: `rebase()` merge/replace/drop/`SETTABLE` semantics, positional construction,
`combine()` per extension (`TimeSpec` equal-keep/else-None, `TilingInfo`/`TimeSpan` always-None,
default equal-or-raise), encode/decode round-trip (both encodings, foreign-key survival, empty-
namespace omission), the `check()` hook end-to-end through real `GeoAnchor.to_geotile()`
construction, idempotency, and the `timespan` rename. `conftest.py`'s `make_anchor` needed one
line fixed (`rebase(time=...)` → `rebase(timespan=...)`) to keep working — was going to break
every fixture-dependent test otherwise.

1000 `GeoTile` constructions (2 extensions each, eager stamp included): **~2.5ms/construction**,
no perf red flag from the eager-stamp change.

---

## 3. Confirmed broken — `spatial/raster.py` (not fixed this session)

Ran the full `tests/geodata/spatial/` suite: **141 failed, 62 passed, 23 errors.** Root cause is
one bug, not scattered regressions: `GeoRaster.rebase()` (raster.py, still has the pre-refactor
signature — explicit `tags=`/`tiling=` params) unconditionally forwards those params — even at
their `UNSET` default — down into `_SpatialArray._rebase()`, which no longer declares `tags=`/
`tiling=` at all. They land in `_rebase`'s `**extensions`, get forwarded again into
`GeoAnchor.rebase(**extensions)`, land in *its* `**extensions` too, and `GeoHeader.rebase()`
tries `{**current.model_dump(), **UNSET}` — `TypeError`, `UNSET` isn't a mapping. Every test
using the `make_raster()` fixture hits this, since `make_raster` always calls
`.rebase(nodata=...)`. `GeoRaster.rebase()` needs its signature brought in line with
`GeoAnchor.rebase()`/`_SpatialArray._rebase()` (drop `tags=`/`tiling=`, rename `time=` →
`timespan=`) before anything downstream is usable again.

Also still broken, not yet reached: `open()`/`_write_tiff()`/`_cf_encoded()` (old `HEADER_KEY`/
`GeoAttrs` references), and once fixed, `_concat_band()`/`GeoMosaic.result()` will need to
switch from the public `.rebase()` to `._rebase()` + pop `timespec` for the private bypass —
same pattern `_concat_time()` already needs, since `combine_extensions`/`GeoHeader.combine`
folds `timespec` uniformly now and the `SETTABLE` guard will reject it through the public path.

---

## 4. Open for next session

1. **Fix `spatial/raster.py`.** The actual next pass: bring `GeoRaster.rebase()` in line with
   the new `GeoAnchor.rebase()`/`_SpatialArray._rebase()` shape, fix `open()`/write paths to use
   `GeoHeader`/`encode_attrs`/`decode_attrs`'s new signatures, fix `_concat_time`/`_concat_band`/
   `GeoMosaic.result()` per §3. Same category for `mosaic.py`/`stack.py`/`stitch.py`.
2. **`Tags.combine()`** has no override — composing rasters with differing tags will raise
   (default equal-or-raise) the moment composition is wired back up, versus today's silent
   first-wins. Undecided: accept the raise, or add a dict-union override.
3. **`TimeSpan.from_input`** is a bespoke method, not part of the five-hook contract every other
   extension follows. Tentative direction from this session: promote it to a formal hook,
   default = defer to `decode`, since no other extension needs real translation logic.
4. **`_SpatialArray.time`/`.start`/`.end`** → `.timespan`/`.start`/`.end`, the property-name half
   of the rename `GeoAnchor` already got. Trivial, deliberately deferred, touches existing tests
   (`raster.time` assertions) that currently still pass under the old name.

# GeoHeader and GeoExtension

> **Status:** Fundamental layer only. `extensions/`, `spatial/header.py`, `spatial/anchor.py`,
> and `spatial/_array.py` reflect this design. `spatial/raster.py`, `spatial/mosaic.py`,
> `spatial/stack.py`, and `spatial/stitch.py` still reference the pre-refactor API and will
> raise at call time until a follow-up pass reaches them.

## What GeoHeader is

`GeoHeader` is a collection of `GeoExtension`s — everything a raster carries besides pixels,
geobox, and vector. It has one field, `extensions: dict[str, GeoExtension]`. Every field that
used to live directly on the header (declared time span, tags, tiling, timespec) is now a
registered `GeoExtension` subclass under its own namespace, with the same rebase/combine/encode
machinery as a plugin extension like `render` or `stac` — no special-cased fields left.

Read a namespace through its own convenience property (`.tags`, `.timespan`, `.tiling`,
`.timespec`) rather than `.extensions` directly.

## The GeoExtension hook contract

Five hooks exist on every `GeoExtension` subclass. Most extensions take the default; a subclass
overrides only the ones where its own semantics differ.

| Extension     | `decode(value)` | `check(data)`                | `combine(values)`            | `encode()` | `SETTABLE` |
| ------------- | ---------------- | ----------------------------- | ------------------------------ | ----------- | ----------- |
| `Tags`        | default          | default                       | default                        | default     | `True`      |
| `TimeSpan`    | default          | default                       | always `None`                  | default     | `True`      |
| `TimeSpec`    | default          | default                       | equal across inputs → keep, else `None` | default | `False` |
| `TilingInfo`  | default          | default                       | always `None`                  | default     | `True`      |
| `RenderHints` | default          | drop stale `rgb_bands`        | default                        | default     | `True`      |
| `StacItems`   | default          | default                       | union by item id               | default     | `True`      |

- `decode(value) -> Self` — parse one stored namespace value (JSON string, dict, or already-live
  instance) back into this extension. Default: JSON-decode a string, then `model_validate`.
- `check(data: xr.DataArray) -> Self | None` — check this extension against the array it's
  actually attached to; return self unchanged, a corrected copy, or `None` to drop the
  namespace. Default is a no-op.
- `combine(values) -> Self | None` — merge this namespace's values when composing several
  arrays into one. Default: equal across inputs or raise.
- `encode() -> dict[str, Any] | None` — fields ready for one store's attrs. Default: JSON-mode
  dump, empty result omitted.
- `SETTABLE: ClassVar[bool]` — `False` keeps a namespace out of `GeoHeader.rebase()`'s generic
  merge path entirely; only the operation that legitimately produces it may set it. `TimeSpec`
  is the only one today — it records a bucketing that actually happened, set only by
  `resample_time`/`concat`, never by a caller.

`TimeSpan.from_input(value)` parses a caller-facing date string or `(start, end)` pair into
fields. It is **not** one of the five hooks above — no other extension needs a translation step,
since a plain field dict is already the ergonomic caller form for `tags=`, `render=`, `tiling=`.
Candidate for promoting to a named hook everywhere (default: defer to `decode`) — not done yet.

## Construction and revalidation flow

Every `_SpatialArray` build — first construction, or any `rebase`/`crop`/`reproject`/etc., since
they all end in `dataclasses.replace()` — runs the same sequence in `__post_init__`. The pass is
idempotent: running it twice on an already-consistent pair changes nothing the second time.

1. `validate_spatial(data)` + geobox match check.
2. `anchor._validate_time(data)` — hard `ValueError` if the declared span doesn't cover the
   data's own time labels. No auto-repair here; time is structural, not presentational.
3. `GeoHeader(anchor.header.extensions, data=data)` — the constructor's `InitVar` path:
   - resolve each namespace to its registered class (`decode` if not already an instance; warn
     and drop if nothing's registered under that namespace);
   - since `data` is given, run every resolved namespace's `check(data)` and drop/replace as
     each one says.
4. If the result differs from the current header, swap it onto the anchor.
5. A dated array whose anchor declares no span yet gets one backfilled from its own labels
   (`span_from_times`).
6. `encode_attrs(data.attrs, anchor.header, "json")` stamps the current header back onto
   `data.attrs`, one key per namespace — so a caller reading `.data` directly (interop,
   plotting, `xr.concat` outside this library) sees real metadata instead of an empty dict.
   Direct mutation of the returned `data.attrs` afterward still silently diverges — same
   "consenting adults" limitation as mutating any array pulled off a nominally-frozen object.

`GeoHeader.rebase(**extensions)` (the caller-facing path, e.g. `anchor.rebase(tags={...})`) does
**not** go through `decode()` — it merges a field dict straight onto the current value with
`extension_cls.model_validate(merged)`. Only the construction path above (raw `data.attrs`, or a
namespace already resolved elsewhere) goes through `decode()`.

## On-disk encoding

No single reserved key. Every namespace present lands under its own top-level attrs key —
`data.attrs["tags"]`, `data.attrs["render"]`, etc. — as a nested dict (`"json"` encoding, zarr)
or one JSON string (`"text"` encoding, GDAL/netCDF, which only hold strings). A namespace
nothing has registered is indistinguishable from a coincidental foreign attr on read; accepted
tradeoff, not fixed.

## Known gaps

- `spatial/raster.py`'s `_concat_band()` and `GeoMosaic.result()` call the *public* `.rebase()`,
  not `._rebase()`. Once `GeoHeader.combine()` folds `timespec` uniformly, they'll raise through
  the `SETTABLE` guard instead of leaking `timespec` through silently — needs switching to
  `._rebase()` and popping `timespec` out for the private bypass, same as `_concat_time`.
- `Tags` has no `combine()` override — composing rasters with differing tags will raise (default
  equal-or-raise) instead of today's silent first-wins. Undecided: accept the raise, or add a
  dict-union override.
- `TimeSpan.from_input` inconsistency noted above.

# Name format strings

Status: deferred on 2026-10-06; the keys need more refining before this is
built. No product code has changed.

## What the mechanism is

A **format string**: text with **replacement fields** in braces, each naming a
key and optionally carrying a **format spec** after a colon. It is Python's own
`str.format` mini-language (PEP 3101). A date key accepts `strftime`
directives as its format spec because `datetime` implements them, which is why
`{start:%Y-%m}` works without any code of ours.

```python
name = forest.gs.anchor.format("{start:%Y}-{start:%m}_{lon:.2f}_{lat:.2f}")
# '2025-06_111.00_-9.05'

forest.gs.to_cog(f"samples/{name}")           # the path names the raster
# samples/2025-06_111.00_-9.05/2025-06_111.00_-9.05_20250601T103031.tif

forest.gs.anchor.format("{site}_{start:%Y%m%d}", site="forest")
# 'forest_20250601'
```

## Rules

1. **Rendering only.** A name is produced from the object. Nothing parses a
   name back: time, grid and bands come from the file header, and the catalog
   holds `datetime` and `geometry` as columns to query.
2. **Keys describe the anchor**: where and when the data is. Anything else is a
   keyword the caller passes.
3. **Standard mini-language, no dialect.** `str.format` does the substitution.
   A key the object does not have raises `KeyError`, which covers a typo and
   a time key on timeless data alike.
4. **The name is the id.** A format with too few keys gives two rasters one
   name, and `upsert` replaces one with the other. For a time series the COG
   writer still appends `_<instant>` per scene.
5. **`GeoAnchor.stem` stays** as the default descriptive name and is built from
   the same key values.

## Public API

```python
# geodata/core/anchor.py: GeoAnchor
@property
def fields(self) -> dict[str, object]: ...
def format(self, template: str, **extra: object) -> str: ...
```

`fields` returns the keys below. `format(template, **extra)` is
`template.format(**self.fields, **extra)`.

## Keys

| Key | Type | Value | Example with a spec |
| --- | --- | --- | --- |
| `start` | `datetime` | first covered instant | `{start:%Y%m%d}` → `20250601` |
| `end` | `datetime` | last covered instant | `{end:%Y%m%d}` → `20250602` |
| `dates` | `str` | the period token `stem` uses today | `{dates}` → `20250601-20250602` |
| `lon`, `lat` | `float` | grid centroid in WGS84 degrees | `{lon:.2f}` → `111.00` |
| `centroid` | `str` | the centroid token `stem` uses today | `{centroid}` → `111.0001E_9.0465S` |
| `x`, `y` | `float` | grid centroid in the grid's CRS | `{x:.0f}` → `500010` |
| `epsg` | `int` | EPSG code of the grid's CRS | `{epsg}` → `32749` |
| `res` | `str` | pixel size token | `{res}` → `10m` |
| `extent` | `str` | ground extent token | `{extent}` → `20mx20m` |
| `width`, `height` | `int` | grid size in pixels | `{width}x{height}` → `512x512` |

`start`, `end` and `dates` are absent from `fields` for timeless data. `epsg` is
absent for a CRS with no EPSG code.

Not included, each for a stated reason:

- **Tiling keys** such as an MGRS tile, a UTM zone or an H3 cell. Each is a
  choice of tiling scheme and most need a new dependency. They are added one at
  a time when a workflow needs one.
- **Place names** from `anchor.location`. They need a network call to a
  geocoder, so a name could differ between runs.
- **Band or variable names.** An anchor describes a grid and a time span, not
  its variables. The COG writer already names split-band files by band.
- **Sample or site identifiers.** They are not derivable from the data, so they
  are passed as keywords.

## Verification

`tests/geodata/core/test_anchor.py` (new or existing):

- every key renders for a dated raster, with and without a format spec;
- `stem` equals the join of `centroid`, `extent`, `dates` and `res`, and of the
  three remaining keys for timeless data;
- a caller keyword renders, and overrides nothing silently: a keyword that
  repeats a key raises `TypeError`;
- an unknown key, and `{start}` on timeless data, raise `KeyError`;
- a rendered name passed to `to_cog` becomes the folder name and the Item id
  prefix.

## Smoke evidence, 2026-10-06

`"{site}_{start:%Y}-{start:%m}_{lon:.2f}_{lat:.2f}_{epsg}".format(...)` with
values read from `build_raster(times=2).gs.anchor` gave
`forest_2025-06_111.00_-9.05_32749`.

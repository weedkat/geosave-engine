# GeoTIFF metadata synchronization and stacked attrs

## Purpose

GeoSave works with time-aware xarray cubes while GeoTIFF stores one instant as
one banded raster. Writing a cube therefore cuts it into scenes, and writing a
stacked `DataArray` first restores its `band` labels to Dataset variables. Both
paths must preserve the metadata a GIS reads from each GeoTIFF band.

This change keeps typed metadata readable without adding compatibility aliases
or a second persistence path.

## Shared collection validation

Add `geodata/attrs/validate.py` for validators reused by several attrs models.
Its first function is `parse_collection_text(value)`: non-string values pass
through, JSON text is decoded, and invalid JSON passes through for the field's
normal Pydantic validation to reject.

Collection-valued attrs fields declare the conversion explicitly with
`Annotated[..., BeforeValidator(parse_collection_text)]`. This applies to
Legend listings and palettes, STAC item tuples, Zarr variable order, and the
stacked-attrs dictionaries. Model-specific validators remain beside their
models; for example, Legend alone owns spelling `class_map` as CF fields.

Unknown attrs are not decoded. GeoSave cannot distinguish a foreign string that
looks like JSON from a collection serialized by an unknown producer.

## Individual field parsing

Rename `read_field` to `parse_field_value`. It converts one stored attr value
to the Python type declared by one model field without constructing a partial
model. This is needed when two attr keys spell the same field and when workflow
requirements validate one value at a time.

Each `AttrsModel` builds its field parsers once when the subclass registers.
`parse_field_value` only looks up the parser and applies it; calling the function
does not populate or mutate a global cache. Its documentation explains the
single-field use case and says plainly that field annotations are applied while
model-level and decorator validators are not. No `read_field` compatibility
alias remains.

## Stacked array metadata

Rename `BandVariables` to `StackedAttrs`. It describes the conversion that made
the metadata necessary, rather than treating variables as a new band concept.
Its fields become `variable_attrs` and `dataset_attrs`.

`GeoRaster.to_array()` stores `StackedAttrs` on the `band` coordinate.
`GeoArray.to_raster()` consumes it to restore each variable's attrs and any
Dataset attrs shadowed while shared variable metadata was lifted. Selection may
drop bands; only metadata for labels still present is restored.

This model remains necessary because xarray cannot represent different attrs
for several Dataset variables after they become one DataArray. In particular,
`DataArray.gs.to_cog()` converts back to a Dataset before writing and needs the
original per-variable units, packing, nodata, descriptions, and colour
interpretations. No compatibility alias for `BandVariables` is retained during
Alpha development.

## GeoTIFF tag synchronization

Add `GeoTIFFTags.from_xarray(obj, *, map_scale=None)`. The method starts with
the TIFF-only tags already carried by the object, then synchronizes redundant
tags from their authoritative native metadata:

- `ACDD.summary` becomes `TIFFTAG_IMAGEDESCRIPTION` when present.
- A scalar `time` coordinate becomes `TIFFTAG_DATETIME` and retains the current
  whole-second validation.
- `map_scale` derives `TIFFTAG_XRESOLUTION` and `TIFFTAG_YRESOLUTION` in pixels
  per centimetre and sets `TIFFTAG_RESOLUTIONUNIT` to `3`.

The GeoTIFF writer invokes this method after reducing a one-step time dimension
to a scalar coordinate and before dropping that coordinate. Generic `rebase()`
does not trigger synchronization: editing format-neutral ACDD metadata must not
silently introduce format-specific TIFF tags onto an in-memory raster.

An authoritative ACDD summary, scalar time, or calculated physical resolution
overwrites a stale carried copy of its TIFF tag. TIFF-only values such as artist,
copyright, document name, software, and host computer remain unchanged.

## Physical resolution

`map_scale` is a positive finite representative-fraction denominator. For grid
resolution `g` expressed in metres per pixel and scale denominator `s`, each
axis is written as:

```text
pixels_per_centimetre = 0.01 * s / abs(g)
```

Projected CRS axis-unit conversion factors convert each grid resolution to
metres first, so metre and foot-based projected rasters both work. Calculation
is refused for an absent grid, a geographic CRS, non-linear or unavailable CRS
axis units, or a non-positive/non-finite map scale. Without `map_scale`, carried
TIFF resolution tags remain untouched.

`map_scale` is an explicit keyword on Dataset, DataArray, stack, layout, COG,
and plain GeoTIFF writing interfaces. It is persistence metadata, not a GDAL
creation option, and propagates to every leaf of a time cube.

## Verification

Tests will establish the behavior before implementation:

- Collection attrs accept native collections and their GDAL JSON text spelling;
  invalid text is rejected by the target field.
- Model-specific Legend validators remain local and continue to enforce CF
  class-map invariants.
- Dataset to DataArray to GeoTIFF restores distinct per-variable metadata,
  including GDAL colour interpretation and CF packing.
- ACDD summary and scalar time synchronize to TIFF tags at write time while
  unrelated carried TIFF tags survive.
- A projected metre grid and a projected foot grid calculate the expected
  pixels per centimetre at a supplied scale.
- Geographic grids, invalid scales, and missing grids fail without writing a
  misleading physical resolution.
- A time cube propagates one map scale to each GeoTIFF leaf and reads back with
  its time and band metadata intact.

Focused attrs, raster, GeoTIFF, and layout tests run first. Ruff checks the
changed files. Persistence tests that cannot complete under the sandbox are
reported explicitly rather than treated as passing.

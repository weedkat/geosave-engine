# Attrs Header Factories

> Updated 2026-10-03: `headers/xarray.py` merged into `attrs/xarray.py`,
> which now holds every read and write of xarray objects; `headers/` keeps
> only foreign sources (GDAL, GeoBox, STAC). `attrs.create_header` is unchanged.

## Purpose

Make metadata construction easier to read without weakening the existing
`AttrsHeader` and `rebase` foundation. Code that gathers metadata from an
external context should say that it creates a header. `AttrsHeader` should
remain a context-free value object, and `rebase` should remain the convenient
interface for applying complete headers or targeted metadata edits.

## Interface

The common xarray path is exported from `geosave_engine.geodata.attrs`:

```python
header = attrs.create_header(dataset)
restored = attrs.rebase(target, header)
```

Specialized contexts use module-qualified factories:

```python
from geosave_engine.geodata.attrs.headers import geobox, stac

grid_header = geobox.create_header(grid)
load_header = stac.create_header(items, collection, loaded, groupby="solar_day")
```

There is no literal-selected dispatcher. Xarray objects, GeoBoxes, and STAC
loads require different inputs, validation, and dependencies; their modules
make those differences explicit while giving every factory the same verb.

The existing public names `read` and the two context-specific `read_header`
functions are removed rather than retained as compatibility aliases. This is
an Alpha interface and the obsolete names add no behavior.

## Modules and ownership

```text
geodata/attrs/
├── header.py
├── headers/
│   ├── __init__.py
│   ├── xarray.py
│   ├── geobox.py
│   └── stac.py
├── model.py
├── namespace.py
└── xarray.py
```

`headers/xarray.py` gathers the root, data-variable, and coordinate mappings
from one Dataset, DataArray, or DataTree node. The package-level
`attrs.create_header` is a direct re-export of this factory because xarray is
the default GeoSave raster representation.

`headers/geobox.py` translates a GeoBox CRS and dimension names into CF
coordinate metadata. `headers/stac.py` owns the complete STAC-to-attrs
translation, including asset and band field interpretation, shared-field
rules, loader-authoritative metadata, and provenance assembly. Keeping that
logic beside the factory gives the translation one place to understand and
test.

Every factory finishes by calling `AttrsHeader.from_attrs(...)`.
`AttrsHeader` therefore continues to own only the context-free conversion of
already-shaped flat mappings into `AttrsNamespace` values, plus header merge
behavior. It does not import or dispatch on xarray, GeoBox, or STAC types.

## Rebase simplification

`rebase` retains its current public forms:

- an `AttrsHeader` replaces the root and every named variable or coordinate;
- an `AttrsNamespace` overlays one or more targets;
- ordered `AttrsModel` values and keyword model edits update or clear one or
  more targets.

The implementation will prepare all target mappings before changing the
object, preserving atomic in-place validation. Model edits will be applied
directly in their declared order. A keyword model set to `None` clears the
keys declared by that model without constructing a synthetic model whose
fields are all `None`. The private `_model_patch` function is removed.

Copying remains shallow so lazy pixel arrays are not computed or duplicated.
Mutable metadata values are independently copied when one edit targets
multiple variables.

## Integration changes

Callers that inspect an xarray object's typed metadata use either
`attrs.create_header(obj)` or the existing `obj.gs.attrs` property. Header
round trips continue to use `attrs.rebase(obj, header)`.

GeoBox-aware constructors and CRS writing import
`attrs.headers.geobox.create_header`. STAC loading imports
`attrs.headers.stac.create_header`. Transform modules use the package-level
xarray factory when preserving a header across an operation.

## Verification

Tests first adopt the new factory names and demonstrate the intended import
paths. Existing behavioral coverage must continue to prove:

- typed and foreign attrs round-trip through a header;
- replacement and patching validate every target before in-place mutation;
- ordered model edits and model removal retain their semantics;
- bulk edits do not share mutable metadata;
- metadata-only work preserves lazy pixel arrays;
- GeoBox coordinate models and STAC load metadata are unchanged;
- old public names have no compatibility aliases.

Focused attrs, core, STAC, transform, and raster I/O tests run before the full
test suite. Ruff checks all changed Python files.

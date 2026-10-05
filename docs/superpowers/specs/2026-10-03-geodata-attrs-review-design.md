# Geodata attrs review

## Intent

`geodata.attrs` gives xarray attrs typed meaning: it decodes flat attrs
mappings into registered models, merges them across joins, and writes them
back. This review checks that package for correctness, stale code, and
avoidable complexity before the rest of `geodata` is reviewed in this order:
attrs, core, stac, transform, features, viz.

Each section below states the contract users can rely on, what is wrong
today, and what changes. Sections are grouped into plans by isolation: plan A
touches one function per task and changes no public signature; plan B
changes the model set, merge policy, and stacked round trip.

## Contracts

### Packing fields from a STAC load

Superseded by `2026-10-03-attrs-header-factories-cleanup-design.md`: items
that disagree drop the field with a `DroppedAttrsWarning` instead of raising.
The history below is kept for context.

One loaded variable carries one `scale_factor` and one `add_offset`, so every
item loaded into it must publish the same value for the asset's `scale` and
`offset`, and either every item publishes the field or none does. A load that
breaks either rule raises `ValueError` naming the items that disagree, because
the pixels of at least one item would otherwise decode wrongly.

Labelling fields (`unit` → `units`, `description` → `long_name`) describe
rather than decode pixels. They are kept when every item publishes the same
value and dropped otherwise.

**Today:** an item that omits `scale` while another publishes `0.0001` drops
`scale_factor` silently (`attrs/headers/stac.py:_shared_fields`). The item that
published `0.0001` then decodes ten thousand times too large.

**Change (plan A):** absence counts as a value for packing fields. Labelling
behaviour is unchanged.

### Restoring a header

`rebase(obj, header)` replaces the root attrs and the attrs of every variable
and coordinate the header names, as decided in
`2026-09-24-attrs-header-factories-design.md`. Variables and coordinates the
header does not name keep their attrs.

**Today:** the behaviour matches the contract, but the `rebase` docstring says
a header "replaces the object's attrs whole", and no test pins the unnamed
case.

**Change (plan A):** a test pins the contract and the docstring states it.

### GDAL band identity

`GDALVariable` carries `variable_name` and `colorinterp`. A GDAL band
description is CF's `long_name` and is read and written through `CFVariable`.

**Today:** `GDALVariable` also declares an undocumented `description` field.
Nothing writes or reads it, but every attr named `description` is captured into
it instead of staying a foreign attr.

**Change (plan A):** the field is removed; a `description` attr is foreign.

### Model set, scopes, merge policy, and stacking (plan B)

Superseded by `2026-10-03-attrs-scoped-models-design.md`. `CFVariable` and
`CFCoordinate` stay separate: each model is scoped to the dataset, variable,
or coordinate mappings it belongs to, which removes the shared-key code
without merging the two. The same spec moves the pixel-semantics merge policy
onto model fields and fixes the stacked-array round trip.

## Withdrawn

Moving the source-specific header factories (`headers/stac.py`,
`headers/gdal.py`), `GeoTIFFTags.from_xarray`, and `GDALVariable.rgb_indices`
out of `attrs` is withdrawn. The first two contradict
`2026-09-24-attrs-header-factories-design.md` and
`2026-09-24-geotiff-metadata-design.md`. The isolation benefit is already
delivered: `test_importing_attrs_does_not_import_stac_adapters` proves that
importing `attrs` loads no STAC adapter.

## Datasets

`geodata.datasets.TileDataset` is a torch `Dataset`. Its only library consumer
is `ml.lightning.tasks.semantic_segmentation`. As
`2026-09-22-geodata-review.md` proposes, it moves to `ml.data` together with
the model-aware sample adapter. That adapter does not exist yet, so the move
is deferred and stays outside these plans.

Once `TileDataset` leaves, `geodata.datasets` is reserved for benchmark
fetchers. Each fetcher downloads a benchmark and returns native xarray or
GeoDataFrame objects plus a manifest, not torch datasets. No fetcher is
planned until a benchmark is needed.

## Verification

Plan A is test-first: every change starts with a failing test in the mirrored
test file. Plans run the focused attrs, STAC, and raster I/O suites, then the
full suite and `ruff check`.

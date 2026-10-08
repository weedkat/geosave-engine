# Catalog raster identity and table loading

Execute inline in the current workspace. Preserve unrelated changes.

## Contract

- A GeoDataFrame row remains one STAC Item.
- Dataset exports carry `geosave:raster` on Items and data assets.
- Stack exports carry `geosave:stack`, ordered `geosave:groups`, and asset-level
  `geosave:group` / `geosave:raster` bindings. Groups are not Collections.
- `to_items(..., id=...)` names the logical Dataset or stack; defaults use the
  output name. Item IDs retain per-scene timestamps for COGs.
- GeoVector exposes `raster_ids`, `stack_ids`, `to_raster(raster_id=None,
  assets=None)`, and `to_stack(stack_id=None, groups=None)`.
- Unspecified IDs are accepted only when selection has at most one known
  identity. Explicit missing IDs raise KeyError; ambiguous identities raise
  ValueError. Untagged external records remain loadable as selected tables.
- Stack loading uses persisted bindings or an explicit group-to-asset mapping.
  Never infer groups from arbitrary asset keys or Collection IDs.
- Windows apply per record before combining. Shared timeless files are loaded
  once for the same window. Preserve laziness and file-close ownership.
- Remove GeoRow and migrate training, tests, and current documentation.

## Tasks

1. Add failing catalog tests for COG/store identity, selection of two rasters in
   one Collection, filtered time subsets, and stack group/time reconstruction.
2. Extend io.catalog export metadata and accessor delegates. Preserve native
   stack persistence. Ensure stack store output directories exist; support
   split-band COG groups without overwriting assets.
3. Extract native Dataset combination in io.readers, implement catalog readers
   and index discovery, and add thin GeoVector methods. Test windows before
   combination, asset filtering, repeated timeless assets, and resource closure.
4. Move record window interpretation to transform.chip.crop_record; migrate
   prepared/virtual ML consumers and all active Series accessor usage. Rename
   asset/group internals consistently; retain native GeoPackage layer names.
5. Run focused I/O, core, chip, and ML tests, static checks, then the full suite.
   Document intentional API removals and any remaining failures.

## Review focus

- Logical raster identity must not depend on Item ID parsing or Collection names.
- External catalogs require explicit stack group mappings.
- Group order, variable identity, temporal axes, nodata, and grid survive storage.
- A pixel window must never be skipped or applied twice.
- Closing combined outputs closes all sources; failures close partial opens.

## Completion

All five tasks implemented. Read-only review covered mixed raster selection,
nullable window fields, selected scenes with absent groups, split-band COGs,
timeless references, variable order, and source cleanup. Reproduced gaps have
focused regression coverage. Saved groups absent from selected rows are omitted;
variables with incompatible temporal coverage still raise instead of introducing
missing pixels implicitly.

Verification: full suite 1608 passed, 16 deselected; three additional GeoParquet
identity/order round trips passed. Ruff on src/tests, scoped BasedPyright on
changed implementation modules, and git diff --check passed.

Intentional removals: GeoRow and pandas Series .gs. Select single records with
iloc[[position]], map external assets to groups explicitly, and use crop_record
for prepared parents. Collection IDs and native GeoPackage layers retain their
existing meanings.

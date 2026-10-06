# Raster reference cleanup

User approved implementation after reviewing the shared native reference-table
contract. Dataset splits are separate Parquet files, never a generated split column.

- [x] Move STAC registration factories to stac/item.py and use transform.vector's
  existing from_layouts directly; remove redundant GeoVector factories and update
  production, tests and maintained docs.
- [x] Use one asset opener for rows, stacks and acquisition catalogs. Native file
  readers own names/timestamps; acquisition assembly only restores ordering and
  scalar/dimensional shape, with metadata explicitly naming the logical raster.
- [x] Keep tile references as ordinary GeoParquet with optional asset pointers;
  distinguish valid STAC-shaped tables from references and preserve pointer
  relocation independently of STAC encoding.
- [x] Return pixel writing to format/layout modules. Keep catalog code focused
  on record selection, validation and publication, without format guessing.
- [x] Test persisted virtual windows with annotations, shuffled/separate split
  files, lazy reads and independent saved-tile reopening without double cropping.
  Run existing persistence/training tests and whole-suite regression verification.

Preserve existing user work. No new wrapper, compatibility aliases, split field,
model recipe serialization or automatic preprocessing replay.


Verification and review:
- Initial smoke tests: 4 failures / 1 pass, confirming missing domain factories,
  generic-reference encoding, and acquisition row loading.
- Focused persistence suite: 44 passed, 1 deselected.
- Fresh review: pixel-only no-catalog store writes and virtual COG references
  needed regression fixes. Both were reproduced failing before implementation.
- Nullable window columns were treated as a correctness fix: whole-raster rows
  must remain readable when a shared reference schema includes null windows.
- Reviewed fixes plus row/catalog/import-boundary checks: 51 passed.
- Ruff passes; scoped BasedPyright reports zero errors. Final full suite: 1484 passed, 26 deselected (155.77s).
- No compatibility aliases or commits; unrelated working-tree edits preserved.

- Final review fixes verified RED to GREEN: unindexed pixel raster persistence
  (NetCDF/Zarr), virtual COG logical image (split/multiband), and nullable windows.
  No outstanding review findings.

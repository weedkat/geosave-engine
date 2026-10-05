# Native raster storage and GeoVector records

Status: implemented and verified.

The user rejected preparation-job abstractions and whole-catalog conversion.
Start from the existing data path: read_raster/read_stack -> native xarray ->
format writer -> GeoVector record -> GeoParquet -> selected asset reopening.
Acquisition, preprocessing, training, and Prefect can compose that path later.
The proposed prepare_catalog job and load_rasters overloads are withdrawn.

## Existing caller flow

This uses current APIs and works for a co-registered, dated stack stored as COGs:

```python
data = read_stack("input/")
saved = data.gs.to_cog("dataset/sample-01", split_bands=True)
record = GeoVector.from_xarray(read_stack(saved), id="sample-01")
record.gs.to_geoparquet("dataset/catalog.parquet")

catalog = read_vector("dataset/catalog.parquet")
restored = catalog.iloc[0].gs.to_xarray()
```

For a saved Dataset, use read_raster and GeoVector.from_assets or from_xarray.
For multiple records, use existing GeoVector.concat/upsert and to_geoparquet.
Parquet holds records and pointers; assets hold pixels. Relative hrefs beneath
its directory already allow the directory and catalog to move together.

The selected-row reader lives on the native Series accessor. Select with
`catalog.iloc[position]`, a filter, or an explicit `set_index("id").loc[id]`;
there is no requirement to set a catalog index. `row.gs.to_xarray()` opens
the row's saved assets and applies any stored pixel window.
`row.gs.crop(parent)` applies that window to explicit prepared xarray data.
The former table-level reader and `data=` substitution have been removed.
Generic GeoJSON selection remains independent of raster asset fields.

## Optional companion catalog

Extend existing format writers rather than introduce another persistence job.
Caller API:

```python
saved = image.gs.to_cog(
    "dataset/image.tif", catalog="dataset/catalog.parquet", id="sample-01"
)
saved = data.gs.to_zarr(
    "dataset/sample-01.zarr", catalog="dataset/catalog.parquet", id="sample-01"
)
saved = data.gs.to_cog(
    "dataset/sample-01", split_bands=True,
    catalog="dataset/catalog.parquet", id="sample-01"
)
```

Without catalog, writing remains an asset-only operation. With catalog, complete
asset writing, construct the GeoVector record from actual saved assets using
existing registration, then publish it through existing GeoParquet persistence.
Do not register original input paths for transformed pixels. Do not rediscover
identity or dates by interpreting filenames when saved metadata supplies them.
The optional record id follows the existing GeoVector factory convention.

Format writers return the saved Path. With `compute=False`, Zarr returns a
delayed task that resolves to the Path after pixels and any catalog finish. For multiple records, the caller
uses GeoVector.concat/upsert explicitly. The companion option does not silently
append to an existing catalog. Existing overwrite rules apply. This is not an
atomic transaction across several assets and the Parquet file.

## Physical storage and pointers

Keep the existing distinction between a Dataset and a DataTree stack. Asset
reading reuses existing format readers. COG, Zarr, and mixed-format records are
valid. Native directories of COG leaves already read through read_raster;
directories of named raster groups read through read_stack.

Current split-band COG registration points to a group's directory; GeoSave's
reader reconstructs that Dataset from its leaves. This is the current tested
representation, not a claim that every external STAC reader loads directory
assets. A conventional catalog listing each leaf COG needs an explicit grouping
contract before being added; it does not justify replacing the current path.

A multi-group Zarr registers named assets pointing to one store with explicit
`group` selectors. Registration and row reopening use the same asset reader.
The Zarr reader opens the native DataTree and selects a group with inherited
coordinates, preserving the root grid without recreating it. Every opened
group retains its store and group in xarray encoding for later registration.
Groups need not be rewritten as independent stores.

Current registration also requires a shared georeferenced grid and a time or
explicit datetime. Native stacks can contain independent grids. That restriction
must remain explicit until per-asset projection metadata and record footprints
are supported; do not pretend the companion option resolves it.

## Ownership and checks

- Existing format I/O owns pixel persistence and reopening.
- GeoVector owns records, metadata, geometry, identity, queries and asset pointers.
- Existing GeoParquet/storage code owns relative paths and filesystem handling.
- Processing operates on native xarray; derived results are written before their
  new pointers are registered. Model inputs and batching remain outside geodata.

A focused smoke test covers a two-date optical raster plus a static label:
split-band COG writing, read_stack, GeoVector registration, relative GeoParquet,
copying the directory, and reopening by record id. Assert pixel values, per-group
grids, time coordinates, dtype and lazy arrays. Preserve existing COG/Zarr tests.
Add behavior tests for optional catalog writing and multi-group Zarr registration
before changing those APIs. Verify failure never publishes a new unfinished
record, and test relocation rather than only the original working directory.


The group reader requires xarray >= 2026.4.0 for native `inherit="all_coords"`;
the declared minimum and lockfile reflect that API. Shared asset reads own their
opened file handles; closing the returned stack, including a cropped window,
closes those handles. Crop and partial-open failures also close completed reads.

# Remote Persistence Design

## Intent

GeoSave persistence accepts local paths and remote filesystem URLs without
turning GeoSave into a storage SDK. GeoParquet is the first remote-capable
format. The same internal filesystem seam must later support direct Zarr
stores and format-specific GeoTIFF and NetCDF strategies.

The first supported live target is the Hugging Face bucket
`hf://buckets/fatmur/test`. Hugging Face access uses its native
`HfFileSystem`, which is already fsspec-compatible. S3-compatible access is a
later integration over the same seam, not the basis of the abstraction.

## Principles

- Public APIs continue to be named for formats: `read_vector`,
  `to_geoparquet`, `to_zarr`, `to_cog`, and `to_netcdf`.
- URLs select storage. GeoSave does not introduce a provider registry,
  credential model, storage client, or remote manifest.
- Each format module owns encoding and decoding. The shared storage module
  owns only filesystem resolution and location arithmetic.
- Credentials remain transient. The underlying filesystem reads its normal
  environment, login cache, or explicit `storage_options`; GeoSave never
  persists credentials.
- A catalog's `path` column is a pointer, not an ownership boundary. Writing a
  catalog never copies or deletes referenced assets.
- Local behavior and return values remain unchanged where possible.

## Public API

GeoParquet read and write gain one explicit provider-neutral option:

```python
catalog.to_geoparquet(
    "hf://buckets/fatmur/test/catalog.parquet",
    storage_options={"token": token},
)

catalog = read_vector(
    "hf://buckets/fatmur/test/catalog.parquet",
    storage_options={"token": token},
)
```

`storage_options` is `Mapping[str, object] | None`. When omitted,
`HfFileSystem` resolves authentication through its normal environment and
login state, including `HF_TOKEN` and `hf auth login`. An explicit token may
be passed as `{"token": token}`. Provider-specific top-level parameters such
as `token=`, `endpoint_url=`, or `access_key=` are not added to GeoSave APIs.

Local writes continue to return `Path`. Remote writes return the destination
URL as `str`, so persistence methods return `Path | str`. Readers continue to
return the native GeoSave geodata object.

The initial implementation does not add `load_asset`. Callers choose the
existing reader because a `.zarr` reference alone cannot distinguish a raster
Dataset from a DataTree stack:

```python
catalog = read_vector("hf://buckets/fatmur/test/catalog.parquet")
row = catalog.query(prediction).gdf.iloc[0]
raster = read_raster(row["path"])
# Or, when the registered asset is a stack:
stack = read_stack(row["path"])
```

## Asset References

The canonical optional `path` column holds one asset pointer.

- A relative pointer is interpreted relative to the catalog's parent.
- An absolute local path remains absolute.
- An absolute URL remains unchanged, including URLs on another provider,
  bucket, or prefix.
- Catalog persistence never verifies that a referenced asset exists and never
  uploads it implicitly.

When writing, an absolute reference already below the catalog's destination
parent may be shortened to a relative POSIX reference. References outside that
parent remain absolute. A caller-supplied relative reference remains relative.

When reading, relative references are expanded against the catalog's parent.
Absolute paths and URLs pass through unchanged. Consequently, every materialized
catalog row contains a pointer that can be handed directly to `read_raster` or
`read_stack`.

This replaces `GeoVector.source` and `GeoVector.resolve_path`. Resolution
belongs at the vector I/O boundary, not on the in-memory spatial collection.
The previous containment rule is not retained: an asset pointer may
intentionally name another directory or bucket, and reading the catalog alone
does not dereference it.

## Filesystem Boundary

Add a small internal `geodata.utils.io.storage` module built on fsspec. It
contains functions rather than a provider hierarchy or GeoSave filesystem
class. Its responsibilities are:

1. Resolve a local path or URL plus `storage_options` into the native fsspec
   filesystem and its protocol-free internal path.
2. Identify local filesystems so existing `Path` and `os.replace` behavior can
   remain native.
3. Join a relative asset pointer to a catalog parent and reconstruct a usable
   local absolute path or remote URL.
4. Relativize an absolute pointer only when it is below the destination parent
   on the same filesystem.

The module does not import GeoPandas, xarray, Rasterio, or any GeoSave geodata
type. Format modules consume its resolved filesystem and path. `fsspec` becomes
an explicit project dependency because it is now an owned runtime boundary
rather than a transitive dependency.

## GeoParquet Implementation

### Reading

`geoparquet.read` resolves the source through the storage module and passes the
native filesystem to `geopandas.read_parquet`. PyArrow therefore performs
range reads and retains GeoParquet column and covering-bbox filtering; GeoSave
does not download the whole file first.

After the frame is read, `read_vector` expands only relative values in its
`path` column against the catalog parent. The resulting `GeoVector` has no
source state.

### Writing

`geoparquet.write` validates the suffix on the filesystem's internal path.
For a local filesystem, it retains the existing staging file and atomic
`os.replace` behavior.

For a remote filesystem, it checks existence when `overwrite=False` and lets
GeoPandas/PyArrow write through the resolved fsspec filesystem. Serialization
stays in GeoPandas and PyArrow; the storage module does not buffer or encode
GeoParquet.

Remote stores do not share local filesystem rename semantics. GeoSave promises
that a successful close completes the write, but it does not promise atomic
remote replacement across providers. Failure cleanup is best-effort and must
never delete an older destination that GeoSave did not create during the
failed call.

Before encoding, asset pointers are normalized according to the rules above.
This transformation does not contact the referenced assets.

## Authentication and Providers

For `hf://`, fsspec discovers `HfFileSystem` from `huggingface_hub`. With no
options, that library owns token discovery. Explicit options are forwarded
only while resolving the filesystem and are not stored on `GeoVector`.

Future `s3://` support uses the same URL and `storage_options` contract through
an optional `s3fs` dependency. Hugging Face's S3 endpoint configuration is a
consumer concern passed through fsspec; no Hugging Face-specific S3 client is
added to GeoSave.

Missing protocol implementations raise an actionable dependency error. Native
authentication and permission errors retain their provider details rather
than being replaced by a generic GeoSave exception.

## Scaling to Other Formats

The shared boundary does not force every format through one byte-transfer
strategy:

- **Zarr:** xarray receives a direct fsspec-backed store. Reads stay lazy and
  writes remain chunked; no local staging is introduced.
- **GeoTIFF:** Rasterio keeps ownership of GDAL encoding. Protocols Rasterio
  cannot write receive a completed local temporary file followed by a storage
  upload. Remote lazy/windowed reading requires a separately verified GDAL or
  seekable-file path and is not implied by GeoParquet support.
- **NetCDF:** xarray retains its selected backend. A backend that requires a
  local random-access file uses local staging; a verified file-object backend
  may write directly.

These later phases reuse location resolution and credentials but remain in
their format modules. There is no generic `write_remote(data, format)` API.

## Error and Overwrite Semantics

- Unsupported suffixes fail before creating a destination.
- `overwrite=False` refuses an existing local or remote catalog.
- Local replacement remains atomic.
- Remote existence checks cannot eliminate races on providers without
  conditional object creation; the first implementation documents this rather
  than simulating an unreliable lock.
- Relative asset expansion never reads the asset.
- Invalid or null `path` values remain null or fail with a clear scalar-type
  error; array-like and opaque property values are not interpreted as paths.
- Credentials are never included in error text, object reprs, catalog columns,
  or logs.

## Testing

Unit tests use fsspec's in-memory filesystem to cover remote path resolution,
read/write round trips, overwrite refusal, failure cleanup, relative pointer
expansion, absolute external pointers, bbox filtering, and column filtering.
Local tests retain the existing atomic-replacement and relocation cases.

A separately marked Hugging Face integration test targets a unique prefix
below `hf://buckets/fatmur/test`. It runs only when the environment can access
the bucket, writes one small catalog, reads it with bbox and column filtering,
and removes only its unique test prefix in `finally`. Tests never embed a token
or print credential-bearing options.

The current environment cannot list `fatmur/test`, so live verification needs
an authenticated `HF_TOKEN` or prior `hf auth login` before implementation is
declared integrated with that bucket.

## Delivery Order and Cost

1. Remove `GeoVector.source` and `resolve_path`; resolve local asset pointers at
   `read_vector` instead.
2. Add the small fsspec storage module and direct dependency.
3. Add remote GeoParquet read/write while preserving local behavior.
4. Verify with in-memory, local, and authenticated Hugging Face tests.
5. Add Zarr remote stores in a separate follow-up.
6. Add GeoTIFF and NetCDF strategies only after their native remote behavior is
   proven independently.

GeoParquet is a moderate change concentrated in the storage utility,
GeoParquet I/O, vector dispatch, and tests. Zarr is another moderate phase.
GeoTIFF and NetCDF are higher-cost phases because their random-access and
staging behavior must be validated independently rather than hidden behind the
GeoParquet implementation.

## Non-goals

- Uploading referenced raster assets when a catalog is written.
- Inferring whether an asset is a Dataset or DataTree.
- Storing provider credentials or inventing a GeoSave authentication API.
- A provider registry, storage plugin system, or universal persistence class.
- Universal atomic replacement on object stores.
- Remote GeoTIFF, NetCDF, Zarr, or S3 support in the first implementation.

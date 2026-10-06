# Remote storage for raster I/O

Status: implemented on 2026-10-06, with rule 5 revised during the work (see
"Revision" below). Nothing is committed. Builds on
`2026-10-06-catalog-loop-design.md`.

## Caller view

```python
options = {"token": token}

paths = forest.gs.to_cog("hf://buckets/me/samples/forest", storage_options=options)
store = forest.gs.to_zarr("s3://bucket/samples/forest.zarr", storage_options=options)

catalog = GeoVector.from_rasters(paths, storage_options=options)
catalog.gs.to_geoparquet(
    "hf://buckets/me/samples/catalog.parquet", stac=True, storage_options=options
)

catalog = read_vector("hf://buckets/me/samples/catalog.parquet", storage_options=options)
forest = catalog.gs.to_raster(storage_options=options)
```

## Rules

1. **One vocabulary.** A location is a local path or an fsspec URL.
   `storage_options` is the only credentials argument, the same name xarray,
   GeoPandas, fsspec and LitData use. `storage.filesystem_path` is the single
   resolver. No path wrapper class.
2. **A writer never guesses.** A URL a format cannot write raises; it is never
   turned into a local folder. Today `ds.gs.to_zarr("memory://b/y.zarr")`
   writes a local folder named `memory:`.
3. **Zarr reads and writes through fsspec directly**, which xarray supports.
4. **COG and NetCDF are written locally, then uploaded.** Both libraries need a
   local file. One step in `storage.py` owns "write here, publish there".
5. **TIFF reads use GDAL's own remote access**: `s3://`, `gs://`, `az://` and
   `https://`, configured with `configure_gdal`. An `hf://buckets/` URL is read
   through the bucket's S3 gateway: `storage.gdal_path` spells it
   `s3://<namespace>/<bucket>/<key>`.
6. **Return values.** A local write returns `Path`; a remote write returns the
   URL `str` it was given, as `to_geoparquet` already does.
7. **New backends cost no GeoSave code.** Installing an fsspec implementation
   adds its protocol.

## Public API after the change

```python
# geodata/io/storage.py
@contextmanager
def local_target(
    location: str | PathLike[str],
    *,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> Iterator[Path]: ...

# geodata/io/geotiff.py
def write_cog(ds, path, *, map_scale=None, overwrite=False,
              storage_options=None, **options) -> Path | str: ...
def write_gtiff(ds, path, *, map_scale=None, overwrite=False,
                storage_options=None, **options) -> Path | str: ...

# geodata/io/cogs.py
def write(ds, path, *, split_bands=False, map_scale=None, overwrite=False,
          storage_options=None, **options) -> tuple[Path | str, ...]: ...

# geodata/io/zarr.py, geodata/io/netcdf.py
def write(raster_or_stack, destination, *, compute=True, overwrite=False,
          storage_options=None, ...) -> Path | str | Delayed: ...

# geodata/io/gdal.py
def read(source, *, chunks=None, mask_and_scale=False,
         storage_options=None, **open_options) -> Dataset: ...

# geodata/stac/asset.py, geodata/stac/item.py, GeoVector
def from_path(path, *, storage_options=None) -> pystac.Asset: ...
def from_paths(paths, *, storage_options=None) -> tuple[pystac.Item, ...]: ...
def from_rasters(cls, paths, *, storage_options=None) -> GeoDataFrame: ...
```

The `gs` writers (`to_cog`, `to_zarr`, `to_netcdf`, `to_gtiff` on Dataset,
DataArray and DataTree) gain `storage_options=None` and forward it. `read_raster`,
`read_stack`, `to_raster` and `to_stack` already forward `**options`, so
`storage_options` reaches each format reader unchanged.

## How each format moves bytes

`storage.local_target(location)` yields a local path to write:

- for a local location, the location itself;
- for a URL, a path with the same name inside a temporary folder. On a clean
  exit the file is uploaded with `filesystem.put`; on an error nothing is
  uploaded. An existing remote object raises `FileExistsError` unless
  `overwrite` is true.

| Format | Write | Read |
| --- | --- | --- |
| COG / GeoTIFF | `local_target`, then the existing GDAL write | `rasterio.open(path)`, or `rasterio.open(path, opener=filesystem.open)` when `storage_options` is passed |
| NetCDF | `local_target`, then the existing xarray write; `compute=False` with a URL raises | local only; a URL raises |
| Zarr | `to_zarr(url, storage_options=...)` | as today |
| GeoParquet | unchanged | unchanged |

`cogs.write` builds each file's location by joining names with `/`, so a URL
keeps its `scheme://`. A raster written to `s3://bucket/samples/forest` gives
`s3://bucket/samples/forest/forest_20250601T103031.tif`.

`stac.asset.from_path` passes `storage_options` to `read_raster` and to the
rasterio open that tells a COG from a plain GeoTIFF. hrefs stay URLs, and the
Item id rule works on them unchanged.

## Limits

- **NetCDF cannot be read from a URL.** It needs a second engine and a file
  object. Writing by upload is supported; reading raises with that reason.
- **`read_raster(folder)` is local only.** The catalog is the record of what a
  write produced.
- **One `storage_options` per call.** Rows whose assets sit on two filesystems
  needing different credentials do not read in one `to_raster` call.
- **An upload is not atomic** on every backend, and a failed multi-file COG
  write can leave earlier scenes uploaded. The returned tuple lists only what
  was written.

## Verification

Tests use fsspec's `memory://` filesystem, plus one `integration` test on an
`hf://` bucket beside the existing GeoParquet one.

- `tests/geodata/io/test_storage.py`: `local_target` for local and remote
  locations, existing object, failed write uploads nothing.
- `tests/geodata/io/test_zarr.py`: remote round trip; the returned URL; no
  local folder is created.
- `tests/geodata/io/test_geotiff.py`, `test_cogs.py`: remote COG write and
  read back through `read_raster(..., storage_options={})`; file URLs for the
  four arrangements.
- `tests/geodata/io/test_netcdf.py`: remote write; `compute=False` and remote
  read raise.
- `tests/geodata/core/test_catalog.py`: the whole loop on `memory://`.

## Smoke evidence, 2026-10-06

On `memory://` with rasterio 1.5.0, zarr 3.2.1, fsspec 2026.3.0:

- `xarray.to_zarr("memory://...")`, `io.zarr.read` and `read_raster` on that
  URL worked.
- `ds.gs.to_zarr("memory://b/y.zarr")` returned `memory:/b/y.zarr` and created
  a local folder `memory:`.
- A COG uploaded with `filesystem.put` opened with
  `rasterio.open(path, opener=filesystem.open)`; pixels and the
  `GEOSAVE_DATETIME` tag matched. `rioxarray.open_rasterio(path, opener=...)`
  failed, so the reader opens with rasterio and hands rioxarray the open
  dataset, as `gdal.read` already does.
- `io.gdal.read("memory://...")` and `xarray.to_netcdf("memory://...")` failed.
- LitData 0.2.67 takes `storage_options` on `StreamingDataset` and `optimize`.

## Revision, 2026-10-06

The first design handed an fsspec filesystem to rasterio as its `opener` when
`storage_options` was passed. rasterio documents that hook, and rioxarray reads
lazily through it when given a path. It fails with `band_as_variable=True` and
when rioxarray is handed an open dataset, because rioxarray reopens the file
under the opener's internal name; `gdal.read` does both. Since GDAL reads S3
natively and Hugging Face buckets have an S3 gateway, the opener was dropped:

- `gdal.read`, `stac.asset.from_path`, `stac.item.from_paths` and
  `GeoVector.from_rasters` take no `storage_options`. The sections above that
  list it for them, and the `opener` row of the format table, no longer apply.
- TIFF reads from a Hugging Face bucket need S3 credentials generated from the
  HF token, set through `configure_gdal(aws_access_key_id=...,
  aws_secret_access_key=..., aws_s3_endpoint="s3.hf.co",
  aws_virtual_hosting=False)`. Writes, Zarr and GeoParquet on `hf://` keep
  using the HF login through fsspec.
- `storage.local_target` checks "exists and not overwrite" for local files as
  well, so the GeoTIFF and NetCDF writers no longer repeat it.
- zarr maps `memory://` to its own store and refuses `storage_options` there,
  so the `memory://` tests omit the argument.

Verified on a real `hf://` bucket: Zarr write, register, table write and read;
and the same loop for split-band COGs, read back through the S3 gateway with
range requests (HTTP 206). The gateway needs
`gdal_disable_readdir_on_open=True`: without it GDAL lists the folder with
ListObjectsV1, which the gateway does not serve, and reports the file missing.
Putting the namespace in the endpoint (`s3.hf.co/<namespace>`) fails GDAL's
request signing, so the namespace is the bucket in the `s3://` form.

Reading over HTTPS with the HF login token also worked, but GDAL's bearer
setting is process-wide and would send the token to every HTTPS host it reads,
so it was not adopted.

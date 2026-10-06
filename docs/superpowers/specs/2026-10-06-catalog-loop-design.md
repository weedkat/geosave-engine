# Catalog loop: save, register, query, read

Status: approved and implemented on 2026-10-06; see the execution notes in
`plans/2026-10-06-catalog-loop.md`. Nothing is committed. Supersedes the catalog, row-reader and
writer-return parts of `2026-10-06-cog-leaves-catalog-design.md`,
`2026-10-06-persistence-redesign-design.md` and the writer section of
`plans/2026-10-06-chip-index.md`.

## The loop

```python
paths = forest.gs.to_cog("samples/forest", split_bands=True)   # tuple[Path, ...]
catalog = GeoVector.from_rasters(paths)                        # one row per scene
catalog.gs.to_geoparquet("samples/catalog.parquet", stac=True)

catalog = read_vector("samples/catalog.parquet")
forest = catalog.gs.query(aoi).gs.to_raster()                  # lazy Dataset
```

Adding a raster to an existing catalog, and registering a batch once:

```python
scenes = GeoVector.from_rasters(water.gs.to_zarr("samples/water.zarr"))
catalog = read_vector("samples/catalog.parquet").gs.upsert(scenes, on="id")
catalog.gs.to_geoparquet("samples/catalog.parquet", stac=True, overwrite=True)

catalog = GeoVector.concat([GeoVector.from_rasters(paths) for paths in written])
```

A training sample is one row whose assets are its layers:

```python
sample = stac.item.from_assets(
    {"optical": stac.asset.from_path("s0/optical.zarr"),
     "label": stac.asset.from_path("s0/label.tif")},
    id="s0",
)
tree = GeoVector.from_items([sample]).iloc[0].gs.to_stack(layers=["optical", "label"])
```

## Rules

1. **Save, then register.** Writers write pixels and return paths. They take no
   catalog or id argument. Registration reads the saved files.
2. **The path names the raster.** A writer's destination stem is the raster's
   name, the COG writer carries it into every scene's file or folder name, and a
   registered Item takes its id from the path its files share.
3. **An asset is one saved file or store** that `read_raster` opens by its href.
   It is built only from that file's header, never from an unsaved object. No
   directory assets.
4. **A row is one STAC Item.** For a registered raster it is one scene: the files
   sharing one instant. A Zarr or NetCDF store spanning several instants is one
   row with `start_datetime` and `end_datetime`.
5. **Asset keys.** `from_paths` keys a single-variable file by its variable name
   and a multi-variable file or store as `image`. The key has to be the same in
   every scene so other STAC loaders can line scenes up. `from_assets` uses the
   caller's keys, which is how a sample names its layers.
6. **Bands, time and grid come from headers.** Reading opens assets with
   `read_raster`. Only an Item's id comes from a path.
7. **hrefs are absolute in memory and relative in the table** when the asset and
   the table share a filesystem, so a catalog moved with its assets still opens.
8. **Catalogs stay native.** A catalog is a GeoDataFrame; pandas and `gs.query`
   select rows; `to_geoparquet` persists it. No wrapper, no custom STAC fields.

## Ownership

| Owner | Responsibility |
| --- | --- |
| PySTAC | Item and Asset objects; Projection, Raster and EO extension fields; href arithmetic (`pystac.utils`) |
| stac-geoparquet | Item to Arrow conversion, STAC GeoParquet serialization |
| xarray | Merging variables and joining instants (`combine_by_coords`) |
| GeoSave format readers and writers | Pixels, file headers, attrs round trip |
| GeoSave `stac.asset` / `stac.item` | Which fields a saved file yields, and which Items one write yields |
| GeoSave `io.cogs` | How a Dataset is arranged as COG files |

## Public API after the change

```python
# geodata/core/raster.py: GeoRaster
def to_cog(self, path, *, split_bands=False, map_scale=None, overwrite=False,
           **options) -> tuple[Path, ...]: ...
def to_zarr(self, destination, *, compute=True, overwrite=False,
            **write_options) -> Path | Delayed: ...
def to_netcdf(self, destination, *, compute=True, engine="netcdf4",
              overwrite=False, **write_options) -> Path | Delayed: ...

# geodata/core/stack.py: GeoStack (to_zarr / to_netcdf lose catalog= and id= the same way)
def to_cog(self, destination, *, split_bands=False, map_scale=None,
           overwrite=False, **options) -> dict[str, tuple[Path, ...]]: ...

# geodata/core/array.py: GeoArray
def to_cog(self, path, *, map_scale=None, overwrite=False, **options) -> Path: ...

# geodata/io/cogs.py (replaces io/layout.py)
def write(ds, path, *, split_bands=False, map_scale=None, overwrite=False,
          **options) -> tuple[Path, ...]: ...

# geodata/io/__init__.py
def read_raster(source: str | PathLike[str] | Sequence[str | PathLike[str]],
                **options) -> Dataset: ...
def read_stack(source: str | PathLike[str] | Mapping[str, RasterSource],
               **options) -> DataTree: ...   # RasterSource: what read_raster accepts

# geodata/stac/asset.py
def from_path(path: str | PathLike[str]) -> pystac.Asset: ...

# geodata/stac/item.py
def from_assets(assets: Mapping[str, pystac.Asset], *, id: str,
                datetime: DateTime | None = None) -> pystac.Item: ...
def from_paths(paths: Iterable[str | PathLike[str]]) -> tuple[pystac.Item, ...]: ...

# geodata/core/vector.py: GeoVector
@classmethod
def from_rasters(cls, paths: Iterable[str | PathLike[str]]) -> GeoDataFrame: ...
def to_raster(self, **options) -> Dataset: ...

# geodata/core/row.py: GeoRow
@property
def hrefs(self) -> dict[str, str]: ...        # data assets: key -> href
def to_raster(self, *, layer: str | None = None, **options) -> Dataset: ...
def to_stack(self, *, layers: Collection[str] | str | None = None,
             **options) -> DataTree: ...

# geodata/io/geoparquet.py
def relative_hrefs(frame, table: str) -> gpd.GeoDataFrame: ...
def absolute_hrefs(frame, table: str) -> gpd.GeoDataFrame: ...

# geodata/transform/vector.py
def chip_windows(parents, tilers, *, padding=None) -> gpd.GeoDataFrame: ...
```

## Writing

Every Dataset writer takes its destination: `to_cog("samples/forest")`,
`to_zarr("samples/forest.zarr")`, `to_netcdf("samples/forest.nc")`. The stem
`forest` is the raster's name. A caller wanting the descriptive name passes
`folder / ds.gs.anchor.stem` itself.

`io.cogs.write(ds, path)` is the only place that arranges a Dataset as files.
With `name = path.name`:

```text
path.tif                              no time dimension, joined bands
path/<band>.tif                       no time dimension, split bands
path/<name>_<instant>.tif             time dimension, joined bands
path/<name>_<instant>/<band>.tif      time dimension, split bands
```

- A raster with no `time` dimension is timeless or carries one scalar instant,
  which its file records in its tags. Its file is named by `path` alone.
- A raster with one variable is written as joined whatever `split_bands` says,
  so one file always carries its scene's name.
- `cogs.write` always reads `path` as a name and appends `.tif` itself, so a
  stack group called `image.tif` is written as `image.tif.tif`.
- `GeoRaster.to_cog(path)` passes `path` to `cogs.write` and refuses a path
  ending in `.tif` or `.tiff`, pointing to `io.geotiff.write_cog` for one
  file. `GeoStack.to_cog` calls `cogs.write` with `destination / group` per
  group. Neither repeats the arrangement.

What a write returns, and what registration makes of it:

```python
forest.gs.to_cog("samples/forest", split_bands=True)
# (samples/forest/forest_20250601T000000/red.tif,     asset "red"  -> Item forest_20250601T000000
#  samples/forest/forest_20250601T000000/nir.tif,     asset "nir"  -> Item forest_20250601T000000
#  samples/forest/forest_20250602T000000/red.tif,     asset "red"  -> Item forest_20250602T000000
#  samples/forest/forest_20250602T000000/nir.tif)     asset "nir"  -> Item forest_20250602T000000

forest.gs.to_cog("samples/forest")
# (samples/forest/forest_20250601T000000.tif,         asset "image" -> Item forest_20250601T000000
#  samples/forest/forest_20250602T000000.tif)         asset "image" -> Item forest_20250602T000000

forest.gs.to_zarr("samples/forest.zarr")
# samples/forest.zarr                                  asset "image" -> Item forest
```

The return value is a flat sequence of the files this call wrote, in time then
variable order. One file or store returns `Path`; a write that can produce
several COGs returns `tuple[Path, ...]`; a stack returns
`dict[str, tuple[Path, ...]]` keyed by group. Each path is one asset.

The instant token comes from one function in `geodata/utils/datetime.py`.
Today `format_instant` is unused and `layout._scenes` builds its own token with a
nanosecond fraction; the two become one function.

## Registering

`stac.asset.from_path(path)` opens the file with `read_raster` (header only) and
returns an Asset holding:

| Field | Source |
| --- | --- |
| `href` | absolute location of `path` |
| `type` | COG when GDAL reports `LAYOUT=COG`, else GeoTIFF; Zarr; NetCDF |
| `roles` | `["data"]` |
| `proj:code` / `proj:wkt2`, `proj:shape`, `proj:transform` | `gs.geobox` |
| `raster:bands` | per variable: `data_type`, `nodata`, `scale`, `offset`, `unit` |
| `eo:bands` | per variable: `name`, `description` |
| `start_datetime`, `end_datetime` | the file's instant (equal) when `time` is scalar; `gs.timespan` when `time` is a dimension; absent when timeless |

Only variable attrs that have a standard STAC band field are written: the
variable name, dtype, `_FillValue`, `scale_factor`, `add_offset`, `units` and
`long_name`. The attr-to-field names are the inverse of `_ATTR_KEYS` in
`attrs/headers/stac.py` and live beside it. Every other attr stays in the file
and returns through `read_raster`; the catalog is an index, not a copy of the
header. A value a caller wants to search on is added as a column.

A file with no locatable grid raises `ValueError`: pixel-only rasters stay in
ordinary GeoParquet reference tables.

`stac.item.from_assets(assets, id=..., datetime=None)` builds one Item from
named Assets without opening files. The footprint is the first asset's grid in
WGS84. Time is the union of the dated assets' spans: `datetime` when it is one
instant, else `start_datetime` and `end_datetime`. When no asset is dated,
`datetime=` is required. The Projection, Raster and EO extensions are declared.

`stac.item.from_paths(paths)` describes the files of one write. It calls
`asset.from_path` once per path, groups assets sharing one span into an Item,
keys them by rule 5, and names the Item by the path its files share: the file's
stem for one asset, their folder's name for several. Files of several rasters
go through one call per raster.

- A timeless file raises `ValueError` naming `from_assets(..., datetime=...)`,
  since STAC requires a time.
- Two assets wanting one key in one Item, or two Items wanting one id, raise
  `ValueError`.

`GeoVector.from_rasters(paths)` is `cls.from_items(stac.item.from_paths(paths))`.
Caller annotations are ordinary columns added to the returned frame.

What can be registered:

| Written by | Registered with |
| --- | --- |
| `GeoRaster.to_cog` / `to_zarr` / `to_netcdf` | `GeoVector.from_rasters(paths)` |
| `GeoArray.to_cog` | `GeoVector.from_rasters([path])`; its COG is an ordinary raster file |
| any raster file from elsewhere | the same, named after its own path |
| `GeoStack` groups, one file or store per group | `stac.item.from_assets({group: asset.from_path(path)}, id=...)` as one sample row, or `from_rasters` per group as separate rasters |

`from_rasters` takes paths only. It never takes a Dataset, DataArray or
DataTree, because an asset describes what is on disk.

## Catalog table

`geoparquet.write`:

- relative hrefs are written for STAC and ordinary tables alike, through
  `pystac.utils.make_relative_href`; `read_vector` restores them with
  `make_absolute_href`. `storage.stored_asset_path` and `resolve_asset_path` are
  deleted. Stored local hrefs gain pystac's `./` prefix, and an asset outside
  the table's folder is stored as `../x` instead of absolute.
- `stac=True` derives the `bbox` column from the current geometry before
  serializing, because GeoPandas drops the covering `bbox` column on read.
- the serializer (stac-geoparquet or GeoPandas) is chosen once; the local and
  remote branches differ only in atomic replace versus cleanup. `_write_stac`
  goes away. Local writes keep temp-file-then-`os.replace`.

`gs.query` filters by time whenever the table has `datetime`; `start_datetime`
and `end_datetime` are used where present.

`gs.upsert` keeps its `GeoDataFrame | pystac.Item` input.

## Reading

`read_raster(sources)` with several sources reads them as one raster:

- each source goes through its own format reader, so COG, Zarr and NetCDF mix;
- variables merge and dated files join along `time`, which is always a
  dimension in the result;
- every source must sit on one grid (`gs.geobox` equal), else `ValueError`;
- no two sources may hold one variable at one instant, else `ValueError`:
  xarray would let one win without reading pixels;
- attrs merge through the existing `attrs.merge` / `rebase`;
- the result owns every opened file and closes them together.

A directory is read as its `*.tif` files in path order through the same code.
`layout.read_tree` is deleted.

`read_stack(sources)` with a mapping reads each value through `read_raster` into
the group its key names. A directory of layers goes through the same code, so
what a writer returns is what a reader accepts:
`read_raster(ds.gs.to_cog(path))` and `read_stack(tree.gs.to_cog(path))`.

`GeoRow.hrefs` maps each data asset's key to its href.

`GeoVector.to_raster(**options)` reads the data assets of every row in the frame
as one raster with `read_raster(hrefs, **options)`. Data assets are those whose
`roles` contain `data` or are absent. An empty frame raises `ValueError`. Rows
are selected before the call; the method takes no id.

`GeoRow.to_raster(layer=None, **options)` reads one asset by key, or every data
asset of the row as one raster when `layer` is None.

`GeoRow.to_stack(layers=None, **options)` reads the named assets with
`read_stack({key: href})`, one group per asset. None reads every data asset. A
missing key raises `KeyError`.

Both row readers apply the row's pixel window through the existing `crop`, which
returns the data unchanged for a row without one. Catalog and row readers open
with `chunks={}` unless the caller passes `chunks`.

## Gaps found by the smoke runs and how each closes

| Gap | Resolution |
| --- | --- |
| A STAC table read and written back loses `bbox` | `bbox` derived from geometry on every STAC write |
| `stac=True` stores absolute hrefs; a moved catalog breaks | relative hrefs for STAC tables too |
| `gs.query` skips its time filter on tables with only `datetime` | filter on `datetime`, widened by span columns where present |
| Reading files on two grids silently concatenates them along `x` | one-grid check raises before combining |
| Two files holding one variable at one instant merge silently, one winning | repeated-plane check raises before combining |
| `row.gs.to_stack` is a stub the training dataset calls | implemented; the 22 `xfail` markers and one `skip` in `tests/ml/segmentation/supervised/test_data.py` removed |
| `stac.asset` / `stac.item` builders are stubs; `workflow/tasks/labels.py` calls them | builders implemented; `read_labels` uses `from_assets({"label": from_path(path)}, id=...)` |
| The COG path rule lives in `layout.write_cog`, `layout._leaves` and `GeoStack.to_cog` | `cogs.write` only |
| Scene files are named by instant alone, so a file carries no raster name and an Item id cannot come from its path | scene files and folders are named `<name>_<instant>` |
| Two instant-token implementations | one function |
| Plain GeoTIFF would be labelled COG by suffix | media type from GDAL's `LAYOUT` |
| `catalog=` and `id=` reserved on seven writers, the ingest flow and its CLI flag | removed everywhere, including `id=` on `GeoRaster.to_cog` |

## Known limits, stated in docstrings

- **Split-band variable order.** Files of one scene are combined in asset-key
  order, which Parquet stores alphabetically. Joined COGs and stores keep their
  written order. Select by name where order matters.
- **Names are ids.** Two rasters saved under the same name in different folders
  get the same ids, and `upsert` replaces one with the other. Give rasters that
  share a catalog different names.
- **Catalog time has microsecond precision.** The file keeps the exact instant,
  and readers take time from the file.
- **A stack row holds one file or store per layer.** A multi-date layer is a
  Zarr or NetCDF store. A layer written as several COGs is registered as a
  raster, not as one layer.
- **One writer per table.** Registering is read-modify-write; concurrent writers
  to one Parquet file are not supported. Batches register once.

## Names

Vocabulary used in code and docstrings:

- **scene**: one raster at one instant; one Item.
- **asset**: one saved file or store, located by its href.
- **COG**: one saved GeoTIFF. Replaces "leaf"; "tree" is dropped.
- **chip**: one model-sized pixel window of a parent. `Tiler` remains the
  library's class name.
- **catalog**: the STAC table of saved rasters. Replaces "reference table" in the
  io docstrings.

| Today | After |
| --- | --- |
| `io.layout` (`write_cog`, `write_tree`, `write_leaf`, `read_tree`, `_leaves`) | `io.cogs` (`write`); reading through `read_raster` |
| `LAYOUTS`, `nested`, `flat`, `LeafPath`, `layout=` | removed: one arrangement |
| `GeoRaster.to_cog(path, id=...)` with `path` as the parent folder | `to_cog(path)` with `path` as the destination |
| `stac.asset.from_xarray(data, href, ...)` | `stac.asset.from_path(path)` |
| `stac.item.from_xarray(data, assets=..., ...)` | `stac.item.from_assets`, `stac.item.from_paths` |
| `GeoVector.to_raster(id)` | `GeoVector.to_raster(**options)` |
| `geoparquet.stored_assets`, `resolved_assets` | `relative_hrefs`, `absolute_hrefs` |
| `storage.stored_asset_path`, `resolve_asset_path` | removed; `pystac.utils` |
| `catalog_path` parameter | `table` |
| `transform.vector.from_tilers` | `chip_windows` |

Locals in the touched functions follow the code-style rule of naming what a value
holds: `spelled` to `href`, `joined` to `path`, `planned` to `cogs`, `cut` to
`scenes`, `beyond` to `extra_dims`, `draft` to `temp`, `existed` to `exists`,
`retained` to `others`, `resolved` and `entry` in `from_items` to `clones` and
`clone`. Code outside the touched functions is left alone.

## Out of scope

- The training and inference workflow redesign: `workflow/tasks/dense.py`,
  `prepare_dense_data`, a persisted chip table, the chip columns (`tile_id`,
  `parent_id`, `raster_metadata`), `Dataset.reference`, and "manifest" in `ml`.
- A stack constructor taking `GeoStack.to_cog`'s return value, and rows for
  multi-group Zarr or NetCDF stores.
- A one-call `register(paths, table)` that also writes the table.
- `odc.stac.load` for local catalogs. `StacSource.load` keeps it for provider
  catalogs. The written catalog remains loadable by it.
- Concurrent catalog writers, deleting assets, and atomic remote writes.

## Verification

Tests mirror the source tree. Each behaviour gets a failing test first.

- `tests/geodata/io/test_cogs.py` (from `test_layout.py`): the four
  arrangements with their exact returned tuples, a one-variable raster with
  `split_bands=True`, explicit TIFF filename, dotted names, overwrite.
- `tests/geodata/io/test_read_raster.py`: several sources equal the directory
  read; COG and Zarr sources; one scene keeps a length-one `time`; the one-grid
  `ValueError`; no pixel computation while opening.
- `tests/geodata/stac/test_asset.py`, `test_item.py`: fields against real COG,
  GeoTIFF, Zarr and NetCDF files; packed and unpacked bands; scene grouping and
  ids for split and joined writes, single scenes, stores and dotted names;
  duplicate key and id errors; timeless files; `Item.validate()`.
- `tests/geodata/core/test_vector.py`, `test_catalog.py`, `test_row.py`:
  `from_rasters`, `to_raster` round trip for split, joined and store rows with
  attrs equal to `read_raster` on the files; `to_stack` on a sample row; window
  applied once; `gs.query` on `datetime`-only and span tables.
- `tests/geodata/io/test_geoparquet.py`, `test_reference.py`: `bbox` survives
  read, upsert and write; relative hrefs for STAC and ordinary tables; a
  catalog moved with its assets still reads; remote (`memory://`) tables.
- `tests/geodata/io/test_storage.py`: href tests removed with the helpers.
- `tests/workflow/tasks/test_labels.py`, `tests/workflow/flows/test_ingest.py`,
  `tests/cli/commands/test_workflow.py`: label directory to table; no `catalog`.
- `tests/ml/segmentation/supervised/test_data.py`: `xfail` and `skip` markers
  removed.
- One interoperability test: `odc.stac.load` reads a catalog GeoSave wrote and
  returns the same pixels, times and grid.

Then `uv run pytest`, `uv run ruff check .`, and
`python scripts/check_docstrings.py` on the touched files.

Docs updated: `docs/guides/architecture.md`, `docs/guides/workflows.md`,
`src/geosave_engine/templates/workspaces/segmentation/README.md`,
`src/geosave_engine/model/README.md`.

## Smoke evidence, 2026-10-06

Throwaway scripts on real files (two dates, two uint16 bands, split and joined
COGs, one Zarr store); pystac 1.14.3, stac-geoparquet 0.8.2, geopandas 1.1.3,
odc-stac 0.5.2.

- Split and joined COGs registered from their paths gave two rows each, with
  `red` / `nir` or `image` assets. A Zarr store gave one row with a span. All
  seven Items passed `Item.validate()` (STAC 1.1.0, projection v2.0.0, raster
  v1.1.0, eo v1.1.0). Scene and store rows shared one table.
- Reading catalog hrefs through `read_raster` per file and `combine_by_coords`
  was identical to today's directory read, lazy, with equal pixels and times.
  The store read through the same path was identical to `read_raster(store)`.
  Split-band variables came back as `nir, red`.
- Two rasters on adjacent grids combined into one four-column raster with
  either `join` setting; hence the one-grid check.
- A dict `bbox` column with `xmin, ymin, xmax, ymax` was written as the STAC
  struct with the covering declared. Relative hrefs survived moving the folder.
- The time filter kept one of two rows once it read `datetime` alone.
- With the proposed `to_stack` patched in, `test_data.py`, `test_module.py` and
  `test_inputs.py` passed with `--runxfail`: 88 passed.
- `odc.stac.load` read the written catalog with matching names, dtype, nodata,
  times, grid and pixels.
- The shared-path id rule was checked as a pure function over every arrangement
  above, a remote href and a dotted name. It failed only for a one-variable
  raster written with split bands, which is why such a raster is written as
  joined. The new file names themselves have not been written to disk yet.
- rio-stac 0.12 was rejected: `datetime` is the current time, no band names,
  plain GeoTIFF media type, and band metadata needs pixel reads.
- An fsspec transaction left a file behind after a failed local write, so the
  local temp-file write stays.

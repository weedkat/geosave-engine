# One raster accessor base and a GeoVector accessor

## Intent

Vectors become what rasters already are: a native ecosystem object carrying a
typed `gs` accessor. A `GeoDataFrame` replaces the `GeoVector` wrapper
instance, and `GeoVector` becomes the accessor class behind `gdf.gs`. The three
xarray accessors share one base instead of two.

The STAC table stays the single protocol between data preparation, training,
and later mapping: one row is one STAC item, and each asset is one raster file
keyed by its layer name. Time lives inside an asset where a sample has a
series, so an asset may be a `.zarr` beside a `.tif`.

```python
manifest = read_vector("manifest.parquet")                  # GeoDataFrame
hits = manifest.gs.query(scene)                              # same ground, same time
sample = manifest.gs.to_xarray(sample_id)                   # lazy stack, one group per asset

row = GeoVector.from_assets(paths, id=sample_id, properties={"quality": 0.97})
manifest = manifest.gs.upsert(row, on="id")
manifest.gs.to_geoparquet("manifest.parquet", overwrite=True)

mask = plots.gs.rasterize(scene, column="class")
regions = GeoVector.vectorize(prediction)
unpacked = stack.gs.unpack()                                # a stack gains the pixel operations
```

The work lands as two plans. Part A touches only rasters and changes no
existing behaviour. Part B is breaking for every caller holding a `GeoVector`
instance; no compatibility alias is kept.

## Part A: one raster accessor base

### Structure

`GeoAccessor` and `GeoRasterAccessor` merge into one class in `core/base.py`:

```python
class GeoRasterAccessor[DataT: xr.Dataset | xr.DataArray | xr.DataTree]:
    # geobox, is_georeferenced, crs, crs_name, bounds, resolution
    # attrs, rebase, timespan, anchor
    # unpack, mask, to_nan, reproject, crop

class GeoArray(GeoRasterAccessor["DataArray"])
class GeoRaster(GeoRasterAccessor["Dataset"])
class GeoStack(GeoRasterAccessor["DataTree"])
```

`GeoStack` keeps its own `timespan` and `anchor`, which read across groups.

### Pixel operations on a stack

`nodata.mask` and `warp.reproject` already accept a `DataTree`. `packing.unpack`,
`nodata.to_nan`, and `vector.crop` gain the same, so every method of the merged
base works on a stack. The accessor methods keep delegating without branching.

The transforms apply a change to each group in three different ways today:

| Where | How | Root attrs |
| --- | --- | --- |
| `nodata.mask` | `tree.map_over_datasets(...)` | kept |
| `warp.reproject` | `stack({...})`, then `attrs.rebase(warped, tree.gs.attrs.root)` | restored |
| `tiling._map_groups` | `stack({...})` | dropped |

Two ways remain, chosen by whether the change moves the grid. A change that
keeps the grid (`unpack`, `to_nan`, `mask`) maps with xarray's own
`map_over_datasets`, which also keeps a coordinate the caller put on the root.
A change that moves the grid (`crop`) cannot be mapped, because the root
`spatial_ref` would no longer match the groups, so it rebuilds through one
helper in `core/stack.py`. A rebuild keeps root attrs but drops caller-added
root coordinates, as `reproject` and `align` already do:

```python
def map_groups(
    tree: xr.DataTree, change: Callable[[xr.Dataset], xr.Dataset]
) -> DataTree:
    """Apply one change to every group, keeping the stack's own attrs."""
    changed = stack({name: change(raster) for name, raster in tree.gs.rasters.items()})
    return attrs.rebase(changed, tree.gs.attrs.root)
```

`crop` uses it. `reproject` keeps its own loop, since each group takes its own
resampling kernel. `tiling` keeps its plain
rebuild: it runs once per tile in the training loop, where restoring root attrs
measured 4.4 ms against 2.6 ms per tile, and a tile is a model input that
needs none.

### Transform cleanup

The transform module mixes two typing styles. A function returning the kind it
was given is generic; overloads remain only where the input type differs from
the output, as in `reduce`, `resample`, and `interpolate`, which take xarray
resamplers.

```python
def unpack[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T: ...
def to_nan[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T: ...
def crop[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T, vector, *, mask=True) -> T: ...
```

The accessor methods then return `DataT` without `cast`, and the result still
resolves to the `Dataset`, `DataArray`, or `DataTree` stand-in that carries
`gs`.

- `crop` burns its mask with GeoSave's own `rasterize` instead of odc's, so the
  module has one burner, and reads the vector's `footprint` instead of
  recomputing the union.
- The two `basedpyright` errors in the module are fixed: the group mapping in
  `tiling.py` and the frame-table return in `time.py`.
- `mosaic`, `merge_bands`, `reduce`, `resample`, `interpolate`, and `frames`
  stay Dataset- or DataArray-only. A stack has `stack_frames`, and nothing
  calls the others on one.

`tensor()` moves from `base.py` to `core/array.py`, and `raster.py` imports it
from there.

## Part B: GeoVector as the GeoDataFrame accessor

### Registration and typing

```python
# core/vector.py
@pd.api.extensions.register_dataframe_accessor("gs")
class GeoVector:
    def __init__(self, data: gpd.GeoDataFrame) -> None: ...

# geodata/__init__.py
if TYPE_CHECKING:
    class GeoDataFrame(gpd.GeoDataFrame):
        gs: GeoVector
else:
    GeoDataFrame = gpd.GeoDataFrame
```

Every GeoSave function that returns a vector is annotated with the
`GeoDataFrame` stand-in, so `.gs` autocompletes on its result. A frame produced
by a native pandas or GeoPandas operation, such as `gdf[mask]` or `pd.concat`,
still has `.gs` at runtime but is untyped to the checker. This is the same
limit rasters have after a native xarray operation. `GeoVector.concat` and
`gs.query` are the typed ways to merge and select.

### Validation

- Accessor construction raises `AttributeError`, as pandas expects, when the
  object is not a `GeoDataFrame`, has no active geometry column, or has no CRS.
- `rasterize` refuses null, empty, and invalid geometries, since burning one
  writes wrong pixels silently. Nothing else validates geometries: GeoPandas
  and the file formats accept them.
- The GeoParquet writer keeps its rules for a table carrying `assets`: a
  unique non-null `id`, and longitude/latitude, which STAC and the
  stac-geoparquet format both define.

### Surface

| Member | Behaviour |
| --- | --- |
| `gs.crs` | odc `CRS` of the frame |
| `gs.footprint` | odc `Geometry`, the union of all geometries |
| `gs.query(target, *, predicate="intersects")` | rows related to an anchor in space, and in time where both state one |
| `gs.rasterize(like, *, column, fill, dtype, all_touched)` | unchanged |
| `gs.upsert(records, *, on)` | replace rows whose key matches and append the rest |
| `gs.to_xarray(id=None)` | open one row's assets as a lazy stack |
| `gs.to_geojson`, `gs.to_geopackage`, `gs.to_geoparquet` | unchanged |
| `GeoVector.from_assets(assets, *, id, datetime, geometry, properties)` | one STAC item row, read from the files |
| `GeoVector.from_xarray(data, *, id, datetime, geometry, properties)` | the same, for an object read from a file |
| `GeoVector.from_geometry(geometry, *, crs, properties)` | one-feature frame |
| `GeoVector.vectorize(flags, *, value_name, mask, connectivity)` | polygons per region |
| `GeoVector.concat(frames)` | typed concatenation on one CRS |
| `GeoVector.empty(crs)` | empty frame on one CRS |

Removed, because GeoPandas already provides them or nothing uses them:
`__len__`, `to_crs`, `from_anchor`, and the `assets`, `fields`, and `crs`
arguments of `from_xarray`. `read_vector` returns the `GeoDataFrame` stand-in.

### A row is always read from files

`GeoVector.from_assets` is how a row is made. It opens the named files lazily
and reads the row from them, so a row cannot describe pixels that are not on
disk and cannot disagree with the file it points at.

```python
row = GeoVector.from_assets(
    {"label": "prepared/s1/label.tif", "sentinel_2_l2a": "prepared/s1/s2.tif"},
    id="s1",
    properties={"quality": 0.97},
)
row = GeoVector.from_assets("prepared/s1/scene.tif")     # the key is the file stem
```

It always returns one STAC item in EPSG:4326. A value is a path or URL, or a
STAC asset mapping stating `href`. Layers that do not share a grid are
refused. `properties` is a mapping, so a caller column can share a name with
an argument; one that takes an item column's name raises.

`GeoVector.from_xarray(data)` is the seamless spelling: it finds the path
`data` was read from and calls `from_assets`. It takes a Dataset, or a stack
whose groups each name an asset by their group name.

```python
row = GeoVector.from_xarray(read_raster("prepared/s1/scene.tif"))
GeoVector.from_xarray(raster({"red": pixels}, geobox))
# ValueError: ... was not read from a file, so no path names it; write it,
#             then register it with GeoVector.from_assets(path)
```

The origin is xarray's own `encoding["source"]` on the Dataset. xarray sets it
for Zarr and NetCDF; GeoSave's GeoTIFF reader sets it too, and `read_tree`
sets the tree's root directory. `mask`, `to_nan`, `unpack`, and `crop` return
objects without it, as `reproject` already does, so an object changed by
GeoSave must be written before it can be registered. `from_xarray` also
compares the object with its file and raises where the grid, the variables, or
the shape differ, which catches a selection, a rename, or a concatenation.
Plain xarray arithmetic keeps the origin and the shape; the row is then still
read from the file and still true of it, but nothing reports that the changed
values were never saved. A stack read from one multi-group store is refused,
since a row names one store per layer.

One layer is one asset with one href, however many files hold it. A layer
written with `write_tree`, as split bands or one file per instant, is
registered by its root directory, and `read_raster` opens a directory through
`read_tree`. `write_tree` and `stack.gs.to_cog` return the path they wrote,
as the other writers do.

Provider provenance comes from the rasters themselves. Each raster loaded from
STAC carries `StacMetadata` in its attrs, with the item properties
`StacSource`'s `item_properties` selected. `from_assets` merges them across
the layers into a `sources` column: one entry per provider item, holding its
id, its datetime as ISO text, and its captured properties. A row whose rasters
carry none has a null `sources`.

Adding a layer to an existing row writes the layer and registers the row
again, so a layer on another grid is refused before a row exists:

```python
write_cog(prediction, "prepared/s1/prediction.tif")
row = GeoVector.from_assets({**hrefs, "prediction": "prepared/s1/prediction.tif"}, id="s1")
catalog = catalog.gs.upsert(row, on="id")
```

`upsert` appends replaced rows at the end, so row order is not preserved. Its
docstring states this.

In a table GeoSave writes, each asset key is a layer name and each asset is one
raster holding all of that layer's bands. A STAC table from another provider
keys its assets by the provider's own names, usually one band each; it reads
with `read_vector` but is not promised to be trainable. `StacSource` remains
the way to load provider catalogs.

### `to_xarray`

```python
def to_xarray(self, id: str | None = None) -> DataTree:
```

Opens every asset of one row with `read_raster(href, chunks="auto")` and
returns `stack({key: raster})`. `id` may be omitted on a one-row frame. It
raises `KeyError` when the frame has no `assets` column or no row with that
`id`, and `ValueError` when `id` is omitted on a frame of several rows. No
pixels are read.

The training `Dataset` opens samples through `to_xarray` and selects the layers
its spec names, replacing its own href loop. `open_sample` stays, because it
opens a sample directory before any manifest exists.

### `query`

Plain selection is GeoPandas' and pandas' own: `frame[frame.intersects(aoi)]`,
`frame.sjoin(...)`, `frame.query("quality > 0.9")`. `gs.query` takes what only
GeoSave has, an anchor, which states both where and when:

```python
manifest.gs.query(anchor)
manifest.gs.query(scene, predicate="within")    # a raster or stack: its anchor is used
```

Space is the anchor's grid extent, reprojected onto the table's CRS and
compared with the GeoPandas predicate, read from each row to the target. Time
is compared only when both sides state one: the anchor has a timespan and the
table has `datetime`, `start_datetime`, and `end_datetime`. A row then matches
when its span overlaps the anchor's, or its `datetime` falls in it where its
span is null. There is no `time` argument, and a bare geometry or another
vector is not a target.

`to_xarray` also takes `layers`, the asset keys to open; None opens every one.
The training `Dataset` passes the layers its spec names, so a worker opens no
file it will not read.

### STAC ownership

`geodata/stac/item.py` owns what a STAC row is: the STAC version, the
projection extension, the media types, the asset builder, the item column
names, and the provider sources. The table rules stay in the writer. `core/vector.py`,
`utils/io/geoparquet.py`, and `workflow/tasks/manifest.py` import from it and
drop their own copies. `core/vector.py` imports it inside `from_assets`, since
the `stac` package imports its client eagerly.

### Behaviour fixes

- `vectorize` repairs regions with `make_valid` before building the frame, so
  `connectivity=8` works on regions that touch at a corner. Such a region
  becomes one MultiPolygon with its area unchanged.
- One CRS rule: a vector follows the raster. `query`, `rasterize`, and `crop`
  reproject the geometry onto the raster's CRS; `crop` no longer raises on a
  CRS mismatch. `concat` and `upsert` combine vectors and raise on a mismatch.

### Callers

`GeoAnchor.from_geometry`, `gs.crop`, the anchor workflow config, the manifest
task, and the training `Dataset` move from `vector.gdf` and `GeoVector(...)` to
the frame and `frame.gs`. `examples/data/dw_imagery/manifest.parquet` is
regenerated as a STAC table; it still has the pre-assets layout and the
`Dataset` cannot read it. `README.md`, `AGENTS.md`, and
`docs/guides/workflows.md` follow the new spelling.

## Out of scope

- A prediction catalog for mapping. It will reuse `from_assets`, `to_xarray`,
  `upsert`, and `query` and gets its own spec with the mapping workflow.
- `unpack` returning `float64` for `uint16` input.
- A `__getitem__` or other pandas mirrors on the accessor.
- Typed results after native pandas or GeoPandas operations.

## Verification

Part A:

- each of `unpack`, `to_nan`, `mask`, `crop`, `reproject` on a two-group stack
  returns a stack with the same groups and root attrs, and the first three
  keep it lazy;
- `crop` masks the same pixels as before on a Dataset and a DataArray;
- existing array, raster, and transform tests pass unchanged;
- `basedpyright` reports no error in `core` and `transform`, and `base.py`
  holds no `cast`.

Part B:

- `gdf.gs` raises `AttributeError` for a plain DataFrame and for a missing CRS;
- `rasterize` refuses null, empty, and invalid geometries;
- `from_xarray` raises for an object that was never read from a file, and
  for one a GeoSave transform changed;
- a layer written with `write_tree` registers and reopens by its directory;
- `sources` lists every provider item of every layer once;
- `from_assets` rows convert through `stac-geoparquet` and validate with
  `pystac` against item 1.1.0 and projection v2.0.0;
- GeoParquet round trip keeps relative hrefs on disk and resolved hrefs on
  read;
- `to_xarray` returns a lazy stack keyed by asset name and raises for an
  unknown or ambiguous `id`;
- registering a row again with a new layer and `upsert` keeps the row count, time,
  and geometry, and a layer on another grid is refused;
- `query` selects by space, and by time overlap where the anchor and the
  table both state one, falling back to a row's `datetime`;
- `vectorize(connectivity=8)` returns valid geometries on corner-touching
  regions;
- `crop` with a vector in another CRS equals `crop` with it reprojected first;
- `properties` may name `id`-like arguments, and a derived-column collision
  raises;
- dense preparation and the training `Dataset` tests pass on the new manifest
  path.

Scoped Ruff, `basedpyright`, the full test suite, and `git diff --check`
complete each part.

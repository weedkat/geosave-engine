# GeoAnchor, GeoTile, GeoStack

Three related types make up the data model the rest of the library is built
around. This page is the reference for all three; see
[pipeline.md](pipeline.md) for how they're built and
[model.md](model.md) for how they become training tensors, or
[workflow.md](../guide/workflow.md) for the ordered procedure.

| Type | Has pixel data? | Role |
| --- | --- | --- |
| `GeoAnchor` | No | "Where and when" — what an `AnchorSource` produces, what `GeoPipeline.ingest()` takes in. |
| `GeoTile` | Yes, always | A `GeoAnchor` plus pixels. What `ingest()` hands back, one per layer. |
| `GeoStack` | Yes (multiple layers) | Several `GeoTile`s for the same anchor, aligned to one grid and saved/loaded together. |

## GeoAnchor

A frozen, keyword-only dataclass (`geosave_engine.geodata.tile.GeoAnchor`):

```python
geobox: GeoBox                          # extent + resolution + CRS — the "where"
datetime: AnchorDatetime                 # normalizes to a (start, end) range — the "when"
metadata: dict[str, Any]                 # arbitrary key-value bag
polygon: Geometry | None                 # exact AOI footprint, if built from one
```

`AnchorDatetime` accepts a single `datetime`, an ISO string, or a
`(start, end)` pair — `__post_init__` normalizes whatever you pass into a
proper range.

Frozen means every mutating-looking method (`with_metadata`, `with_geobox`,
...) returns a **new** instance via `dataclasses.replace` — the original is
untouched.

Two fields do all the work of being an anchor:

- **`geobox`** carries where and at what resolution — every `Source.load()`
  or STAC search uses `anchor.geobox`/`anchor.wgs84_bbox` directly, no
  separate bounds parameter to keep in sync.
- **`polygon`**, when present, carries the *exact* AOI shape (not just its
  bounding box) — set by `from_polygon`/`from_geojson` when your ingest
  source is a real geometry rather than a bbox or a point-and-radius. An
  anchor built via `from_bbox`/`from_coordinate` has no polygon; its spatial
  identity is just the geobox rectangle.

### Constructing an anchor

| Constructor | From | Notes |
| --- | --- | --- |
| `GeoAnchor.from_coordinate(latitude, longitude, *, datetime, size_m, resolution=10.0, crs=None)` | a center point + size | Projects to local UTM/UPS unless `crs=` is given; raises if `crs` is geographic (not projected). |
| `GeoAnchor.from_polygon(polygon, datetime, resolution=10.0, crs=None)` | a GeoJSON geometry dict | Stores the exact polygon, not just its bbox. |
| `GeoAnchor.from_geojson(path, datetime, resolution=10.0, crs=None)` | a GeoJSON file | Yields one `GeoAnchor` per feature, lazily — each goes through `from_polygon`. Reading the file is eager (one `json.load`), but each feature's own reprojection/GeoBox build is deferred until consumed. |
| `GeoAnchor.from_bbox(bbox, datetime, crs="EPSG:4326", resolution=10.0)` | a raw bounding box | No polygon stored. |

These four live only on `GeoAnchor` — reach for them directly when writing a
new `AnchorSource` or building an anchor by hand in a notebook.

### Anchor properties

| Property | Returns |
| --- | --- |
| `resolution` | Pixel size, from `geobox.affine.a`. |
| `affine` | `Affine` transform. |
| `crs` | `"EPSG:{n}"` or `None`. |
| `width` / `height` | Pixel dimensions. |
| `bbox` | Bounding box in the anchor's own CRS. |
| `wgs84_bbox` | Bounding box reprojected to WGS84 — what STAC searches use. |
| `centroid` | `(lon, lat)` in WGS84. |
| `area_m2` | Polygon area if stored, else `width * height * resolution**2`. |
| `start` / `end` | Datetime range endpoints. |
| `stem` | Deterministic filename identity: `f"{lon:.6f}_{lat:.6f}_{start}_{end}_{res}m"`, e.g. `13.000000_52.000000_20240101T000000_20240101T235959.999999_10m`. This is what makes ingestion resumable and lets `GeoDataset` match layers by dict key instead of a spatial join. |
| `bbox_polygon` / `geojson` | Shapely geometry / WGS84 GeoJSON Polygon. |
| `location` | Reverse-geocoded place info (`{}` on failure). |

### Building or transforming an anchor

- `with_geobox(geobox)` — pure geometry rebase.
- `with_metadata(extra, replace=False)` — attach key-value pairs; raises
  `ValueError` on a key collision unless `replace=True` — a guard against
  silently overwriting something another step already attached.
- `with_data(data: xr.DataArray) -> GeoTile` — attach pixel data directly,
  turning the anchor into a `GeoTile`.
- `with_np(array, names, times=None) -> GeoTile` — build the `DataArray` for
  you from a plain 2D/3D/4D numpy array. The usual path inside an `ingest()`
  body that computes pixels rather than loading them (a cloud mask or NDVI
  derived from an already-fetched tile, for instance).

## GeoTile

A frozen, keyword-only dataclass, subclassing `GeoAnchor`
(`geosave_engine.geodata.tile.GeoTile`):

```python
data: xr.DataArray            # (band, y, x) or (time, band, y, x) — always present
stac: list[pystac.Item]       # STAC provenance
```

A `GeoTile` **always has data** — there's no data-less `GeoTile`; a
data-less reference is a `GeoAnchor`. That split (rather than one class with
an optional `data` field) means "does this have pixels?" is answered by the
type itself, not a `None` check scattered through calling code.

Because `GeoTile` is a `GeoAnchor` subclass, every property and builder
above works on it too — `bbox`, `stem`, `with_metadata`, etc. all apply
unchanged, plus the additions below.

### Lazy loading

`from_geotiff`/`from_zarr` default to `load_data=False`: the backing array
is opened lazily (dask-backed for zarr, chunked for geotiff) — geobox and
datetime are read from attrs/metadata without touching pixel data. Nothing
forces a real read until you call `to_tensor()`/`to_numpy()`, which clips to
`self.bbox` and pulls values.

This is why scanning a whole ingested directory
(`GeoDataset`, see [model.md](model.md#from-ingested-data-to-tensors-geodataset))
is cheap even over thousands of tiles — it opens every store, reads geometry
and stem, and stops there.

### Constructing a tile

| Constructor | From | Notes |
| --- | --- | --- |
| `GeoTile.from_geotiff(path, datetime=None, load_data=False, bands=None)` | an existing raster | If `datetime` is omitted, it's derived from the filename's `-YYYYMMDD` or `-YYYYMMDD-YYYYMMDD` suffix — raises `ValueError` if the filename has neither. |
| `GeoTile.from_zarr(path, datetime=None, load_data=False)` | a store written by `to_zarr` | Datetime prefers the store's own `time` coordinate when present, else falls back to a stored `datetime` attr — raises `ValueError` if neither is available. |

These are what `GeotiffSource`/`ZarrSource` call internally (see
[pipeline.md](pipeline.md#pulling-from-local-geotiff)); reach for them
directly in a notebook or a one-off script.

### Tile properties

| Property | Returns |
| --- | --- |
| `bands` | Band names, in order. |
| `num_bands` | Band count. |
| `has_time` | Whether `data` carries a `time` dimension. |
| `times` | Timestamps present, empty tuple if no time dim. |
| `nodata` | GDAL-standard nodata value from `data.rio.nodata`, or `None` if undeclared. |

### Building or transforming a tile

- `with_data(data)` — replace pixel data, same geometry/time.
- `with_nodata(value: float | None)` — set (or clear, with `None`) the
  GDAL-standard nodata value.
- `with_stac(items)` — append STAC provenance items, de-duplicated by id.

### Reading data out

`to_tensor(bands=None, squeeze=False)` / `to_numpy(bands=None)` — clip to
`self.bbox`, stack bands, return `(band, y, x)` or `(time, band, y, x)`.
This is what actually materializes lazy data into memory.

### Writing data out

| Method | Time dimension | Use for |
| --- | --- | --- |
| `to_zarr(path, save_stac=False)` | Supported | Any layer, especially ones with a time axis. |
| `to_geotiff(path, save_stac=False)` | Raises if present | Single-timestamp rasters. |
| `to_cog(path, save_stac=False)` | Raises if present | Cloud-optimized single-timestamp rasters. |

`save_stac=True` writes STAC provenance alongside the raster as a
`<stem>.stac.json` sidecar; `from_zarr`/`from_geotiff` read it back in
automatically if it exists, no flag needed on the read side.

### Working with more than one tile

Module-level functions, once you have two or more `GeoTile`s in hand:

- **`remap(tile, mapping)`** — relabel integer values (`{old: new}`) —
  the tool for compacting a raw label encoding into a training-ready
  schema (see [pipeline.md](pipeline.md#handling-labels) for a worked
  example).
- **`align(*tiles)`** — narrow several tiles to their common spatial
  intersection; pure geometry, requires matching CRS/resolution/pixel grid.
  Requires at least 2 tiles. `GeoStack`'s constructor calls this for you.
- **`mosaic(tiles, crs=None, time_round_to='D')`** — stitch spatially
  non-overlapping tiles sharing the same bands into one larger tile.

There's no dedicated "combine two different layers' bands into one tile"
function — that's a plain `xr.concat([a.data, b.data], dim="band")`
followed by `a.with_data(combined)`, the same kind of transform you'd write
inside any `preprocess()` body.

## Metadata

`metadata: dict[str, Any]` is a plain, arbitrary bag. It round-trips through
`to_zarr`/`from_zarr` as a JSON-serialized store attr, and through
`to_geotiff`/`to_cog` as a GDAL tag.

One thing worth knowing before you rely on it: JSON object keys are always
strings, so a `metadata` value with integer keys (e.g. `{0: "water", 1:
"trees"}`) comes back as `{"0": "water", "1": "trees"}` after a save/load
round trip. Two consequences that matter in practice:

- **`GeoTile.metadata` is not read anywhere in the training path today** —
  `GeoDataset`, `stack_samples`, `SemanticSegmentationTask`, and
  `SemanticSegmentationDataModule` never touch it. It's provenance for
  humans reading a store's attrs, not a config channel a model consumes.
- **Anything a model actually needs at train time — `class_map`,
  `color_map`, `band_map`, `ignore_index` — belongs in the LightningCLI
  config as a task/data-module constructor arg, not baked into a tile's
  `metadata`.** Config values are parsed straight from YAML with no JSON
  round trip, so integer keys stay integers. See
  [model.md](model.md#semanticsegmentationtask) for where those keys live
  today.

## STAC provenance

`stac: list[pystac.Item]` — the actual catalog items a tile's pixels came
from, when it came from a STAC search (`StacSource.load` attaches them via
`with_stac`). Not populated for tiles built from
`from_coordinate`/`from_bbox`/local files.

## GeoStack

`GeoStack` groups several `GeoTile`s for one anchor — imagery, a cloud
mask, a label layer — into one thing that saves and loads as a unit:

```python
from geosave_engine.geodata.tile import GeoStack

stack = GeoStack(sentinel_2_l1c=image_tile, cloud_mask=mask_tile, dynamicworld=label_tile)
path = stack.save("data/train/13.000000_52.000000_..._10m.geostack")
```

Constructing with more than one tile runs them all through `align()`
automatically — every tile in a `GeoStack` shares one spatial grid by
construction, no separate alignment step to remember.

### The `.geostack` folder convention

`save`/`load` both require the path to end in `.geostack`
(`GEOSTACK_SUFFIX`, exported from `geosave_engine.geodata.tile`) — a
folder-name suffix, the same convention as `.zarr`/`.tif`/CDSE's `.SAFE`.
Each layer is written as `<path>/<layer_name>.zarr` inside it.

This is what makes discovery unambiguous when anchor folders are nested —
`GeoDataset` finds every anchor with a plain
`root.rglob(f"*{GEOSTACK_SUFFIX}")`, at any depth, with no risk of mistaking
a stray unrelated `.zarr` store elsewhere in the tree for a real anchor:

```text
data/train/
├── 13.000000_52.000000_..._10m.geostack/
│   ├── sentinel_2_l1c.zarr
│   ├── cloud_mask.zarr
│   └── dynamicworld.zarr
└── Experts/EH/1/13.320000_..._10m.geostack/
    └── ...
```

Both folders above are found regardless of nesting — grouping ingested
output to mirror a raw dataset's own folder structure (by biome, by
expert/non-expert, whatever the source data's own layout is) works with no
extra bookkeeping.

### Reading a stack back

```python
loaded = GeoStack.load(path, required_layers=["sentinel_2_l1c", "dynamicworld"], load_data=True)
```

Raises `KeyError` if any `required_layers` name is missing from what's on
disk.

### Rendering a sample

```python
sample = stack.to_tensor(sel_bands={"sentinel_2_l1c": ["B04", "B03", "B02"]}, context_fn=pipeline.context)
```

Returns a tensor dict keyed by layer name, plus a `"context"` key only if
`context_fn` is given and returns something non-empty. This is the same
method `GeoDataset.__getitem__` calls per sample (see
[model.md](model.md#from-ingested-data-to-tensors-geodataset)).

## What's next

[pipeline.md](pipeline.md) covers how `GeoTile`s get built in the first
place — pulling from a live STAC catalog or local files, and saving anchors
to `.geostack` folders in bulk. [workflow.md](../guide/workflow.md) is the
ordered procedure that ties it all together.

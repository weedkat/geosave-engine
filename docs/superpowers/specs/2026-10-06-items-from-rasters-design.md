# Items from rasters: describe what was written, not what is found

Status: design agreed in conversation on 2026-10-06; plan in
`plans/2026-10-06-items-from-rasters.md`. Supersedes rules 1, 3 and 5 of
`2026-10-06-catalog-loop-design.md` and its `from_paths`, `from_rasters` and
`stac=True` parts.

## Caller view

```python
# An unsaved raster is saved by the driver, then described.
items = scene.gs.to_items("samples/forest")                      # COG, one Item per instant, one asset per band
items = scene.gs.to_items("samples/forest", split_bands=False)   # one multi-band file per instant
items = scene.gs.to_items("samples/forest.zarr", driver="zarr")  # one store, one Item
items = scene["B04"].gs.to_items("samples/b04")                  # a band converts to a raster first
items = sample.gs.to_items("samples/s0")                         # a stack: one asset per group

# A raster read from one file or store describes itself where it sits.
items = read_raster("samples/forest.zarr").gs.to_items()

catalog = GeoVector.from_items(items)
catalog.gs.to_geoparquet("samples/catalog.parquet")
forest = read_vector("samples/catalog.parquet").gs.query(aoi).gs.to_raster()
```

## Rules

1. **A file holds one raster, so an asset is built from one raster.**
   `stac.asset.from_raster(raster, href, driver=...)` is the only builder and
   opens no file. `stac.asset.from_path(path)` reads the file and calls it.
2. **An Item is stack-shaped:** named rasters at one place and time, one asset
   each. `stac.item.from_assets` is the only assembler.

   | Indexed object | Asset keys |
   | --- | --- |
   | raster, bands split | band names |
   | raster, one file | its variable when it has one, else `image` |
   | stack | group names |

3. **Nothing is guessed back from files.** `to_items` describes the pixels it
   hands to the writer, from the layout the writer follows. `item.from_paths`
   and path-derived names are deleted.
4. **`driver` is the one format word:** `"cog"` (default), `"zarr"`,
   `"netcdf"`. STAC's `media_type` exists only on the pystac Asset.
5. **`to_items` owns no writer option.** It forwards every option to `to_cog`,
   `to_zarr` or `to_netcdf`. Its own arguments are `path`, `driver` and
   `collection`, plus `split_bands`, which it defaults to true for COGs
   because catalogs key assets by band.
6. **Rows of one object share a `collection`:** the name passed to `to_items`
   (`"samples/forest"` gives `"forest"`), or `collection=`. The STAC builders
   themselves default to none.
7. **One table type.** Scene rows and sample rows are both Items in one
   GeoParquet table; `row.gs.to_raster()` and `row.gs.to_stack()` read either.

## Item granularity

| Object and driver | Items | Files |
| --- | --- | --- |
| raster, `cog` | one per instant | `forest/forest_<instant>/<band>.tif`, or `forest/forest_<instant>.tif` unsplit |
| raster, `zarr` / `netcdf` | one, spanning its instants | `forest.zarr` / `forest.nc` |
| stack, `cog` | one per instant any group reaches; a timeless group joins every Item | `s0/<group>/<group>_<instant>.tif`, `s0/<group>.tif` when timeless |
| stack, `zarr` / `netcdf` | one, spanning its groups | `s0/<group>.zarr` / `s0/<group>.nc` |

A stack's store drivers write one store per group, as `workflow.tasks.sample`
already lays a sample out, so each group stays an asset `read_raster` opens.

## Known limits, stated in docstrings

- `to_items()` without a path describes a raster read from one file or store.
  A raster read from a folder of COGs raises: the Dataset no longer says which
  instant each file holds.
- A timeless raster or stack raises, since a STAC Item needs a time. Build its
  Item with `stac.item.from_assets(..., datetime=...)`.
- An asset described at write time and one read back from the file agree on
  href, type, grid, time, dtype, nodata and scale. Two fields differ: a COG
  reads a band without `long_name` back with its name as description, and
  drops an `add_offset` of zero.
- A band indexes as COG only, as its writers are `to_cog` and `to_gtiff`.

## Out of scope

Credentials for describing a saved raster in a private bucket; routing
`workflow.tasks.sample` and `workflow.tasks.labels` through `to_items`; the
readability of `io.geotiff._write` and the two href loops in `io.geoparquet`.

## Verification

`uv run pytest`, `uv run ruff check src tests`, and `uvx ty check` over
`geodata/stac`, `geodata/io/cogs.py` and `geodata/attrs/headers/stac.py`
reporting no diagnostic.

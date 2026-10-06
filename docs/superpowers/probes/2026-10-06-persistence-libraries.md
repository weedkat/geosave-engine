# Persistence library evidence and selected contract

The public representation stays native: PySTAC Item, GeoDataFrame, xarray
Dataset/DataTree. STAC GeoParquet is an index of saved assets. File metadata
owns the grid, pixels, dtype, nodata, native band names and timestamps.

## Serializer decision

Keep PySTAC 1.14.3 and stac-geoparquet 0.8.2, with the existing GeoPandas
1.1.3 / PyArrow 23.0.1 path. Rustac 0.9.17 and arro3-core 0.9.0 were
installed only into `/tmp/geosave-rustac-probe`; project dependencies did
not change.

| Probe | stac-geoparquet | Rustac |
| --- | --- | --- |
| Native PySTAC Item input | Supported | Requires Item dictionaries |
| Heterogeneous properties / asset names | Supported | Supported |
| Nested objects | Supported | Supported |
| Empty nested object in Parquet | Rejected by Arrow/Parquet | Rejected by Rustac/Parquet |
| Explicit null property | Nullable column | Null property omitted when returned as an Item |
| Edited/reordered GeoDataFrame | Supported; derive bbox from current geometry | Supported; still needs current bbox |
| Nine-digit datetime string | Parser rejects unsupported precision | Round trip silently truncates to milliseconds |
| Native Arrow datetime64[ns] catalog write | Preserves nanoseconds | No improvement demonstrated |

Native Item support and preserving precise saved coordinates matter more than
changing serialization engines. Rustac is a credible alternative for bulk
search and one-shot I/O, but these probes do not justify replacing the current
serializer. Keep geometry-derived bbox: neither a stale Item bbox nor dropping
bbox makes GeoDataFrame edits interoperable.

Reproduce the main contracts with:

```bash
uv run pytest tests/geodata/stac/test_native_items.py tests/geodata/io/test_geoparquet.py
uv run pytest tests/geodata/io/test_stack_catalog.py tests/geodata/io/test_catalog.py
```

The Rustac precision failure is reproducible using its native write/read API:

```python
import rustac
entry = item.to_dict()  # independently constructed PySTAC Item
entry["properties"]["datetime"] = "2025-06-01T00:00:00.123456789Z"
rustac.write_sync("/tmp/rustac-time.parquet", [entry])
print(rustac.read_sync("/tmp/rustac-time.parquet")["features"][0]["properties"]["datetime"])
# 2025-06-01T00:00:00.123Z
```

## Selected fields and responsibilities

| Field | Owner / reason |
| --- | --- |
| Item/Asset structure, geometry, datetime/interval, links | PySTAC |
| `proj:*` properties and extension declaration | PySTAC ProjectionExtension |
| Media type and data roles | Native STAC Asset conventions |
| `xarray:open_kwargs` | Existing ecosystem convention for group/backend opening options |
| `geosave:raster` | Logical Dataset identity spanning several acquisition Items |
| Asset `geosave:layer` | Several physical COGs compose one named logical raster |
| `raster_metadata` | Ordered time context, variable order, time axis shape/dtype; used by split COG reconstruction and metadata-only model encoders |
| Row window fields | GeoSave pixel-window selection in ordinary GeoParquet references |
| `sources` | Original provider provenance carried from native GeoSave attrs |

Do not add a general xarray reconstruction model. Native Zarr/NetCDF own full
structure. COG support is explicitly limited to spatial pixels, optional time,
and variable bands; other pixel axes are rejected. The small existing
`raster_metadata` contract remains until a demonstrated standard representation
can preserve these ordering/axis invariants and feed model time context.
External Items require none of those GeoSave fields: each data asset key is
its layer, and explicit `to_raster(layer=...)` resolves ambiguity.

The xarray-assets extension is deprecated upstream. It remains supported by
PySTAC and used by xpystac, with no documented successor for NetCDF/Zarr group
selection. Use its established opening convention through PySTAC's extension
API, document the limitation, and avoid adding our own duplicate option keys.
Datacube describes dimensions/variables, but replacing the current metadata with
it would still require a GeoSave reconstruction policy for COG axis shape,
dtype and asset grouping. xstac targets native Dataset/datacube descriptions;
it does not supply that split-COG/independent-layer reconstruction policy.
rio-stac is useful for raster metadata/statistics, but re-reading pixels for
statistics or introducing another COG registration path adds no demonstrated
benefit here. No new dependency is needed.

## Deletions and private interfaces

Delete `item_columns`, `assets.write_catalog`, `geosave:format`, broad row
`to_xarray`, the special routing for a metadata key named `image`, Stack COG
directory assets, and bare hand-owned `group` / `engine` asset fields.

- `GeoVector.from_items(items) -> GeoDataFrame`: clone, resolve hrefs, native library conversion.
- `group_assets(assets)`: data roles plus explicit composition; never guess paths.
- `read_acquisition(assets, metadata=None) -> Dataset`: lazy native assembly.
- `read(assets, layers=None, metadata=None) -> DataTree`: explicit named selection.
- `io.catalog.write(data, destination, format, catalog=None, ...)`: one complete indexed save for every accessor; physical writers and Parquet serialization remain separate.
- `from_assets(...)` and `from_xarray(...) -> pystac.Item`: native metadata snapshots; no implicit Item-to-frame conversions.
- `write_leaves(...) -> list[Path]`: enumerate completed physical COGs.

## Sources

- [Rustac APIs and goals](https://github.com/stac-utils/rustac-py)
- [stac-geoparquet conversion and I/O](https://stac-utils.github.io/stac-geoparquet/latest/usage/)
- [Xarray assets deprecation and opening convention](https://github.com/stac-extensions/xarray-assets)
- [xpystac opening behavior](https://github.com/stac-utils/xpystac)
- [Datacube metadata](https://github.com/stac-extensions/datacube)
- [xstac Dataset descriptions](https://github.com/stac-utils/xstac)

# Persistence cleanup and native Item construction

Status: cleanup and public skeletons; metadata implementation remains pending. The earlier implementation was rejected; its plan
is saved at `/tmp/geosave-persistence-rejected-plan.md`. This document replaces
its execution instructions. Preserve unrelated working-tree changes.

## Scope

Establish one path:

```text
xarray metadata + actual saved assets
    -> complete native pystac.Item
    -> stac-geoparquet Arrow conversion
    -> native GeoDataFrame
    -> stac-geoparquet serialization
```

GeoSave owns the mapping from its native raster metadata to a complete Item.
PySTAC owns Item/Asset objects and supported extension APIs. stac-geoparquet
owns table conversion, schema inference and STAC GeoParquet file metadata.
Raster format modules own pixels. `catalog=` remains a convenience composing
these same steps after the raster write succeeds.

## Item and asset metadata

Each TIFF asset describes an actual file and only the bands stored in that
file. The writer must provide exact saved hrefs and band arrangement; the Item
builder must not infer them from names or apply the entire Dataset band list
to every asset.

| Write arrangement | Assets in one acquisition Item | Band metadata |
| --- | --- | --- |
| Joined TIFF | One image asset | All written bands, in physical order |
| Split TIFFs | One asset per written band file | Only that file's band |

Item fields describe identity, footprint and temporal coverage. Searchable and
caller annotations belong in Item properties before conversion. Asset fields
contain href, media type, roles, ordered band descriptions and applicable
asset-specific projection/opening metadata. Data type, nodata, units and
calibration belong at the appropriate asset/band scope using common STAC fields
and established extensions. Output metadata must describe written values when
native writer options change their representation.

stac-geoparquet flattens Item properties into columns and retains asset fields
under the assets column. No STAC description is completed by patching a frame
after conversion. Do not export all internal xarray attrs automatically.

Use STAC 1.1 common band metadata as the starting convention. Inspect installed
PySTAC and format generators before choosing implementation: installed extension
APIs do not necessarily implement the newest extension schemas. A generator
must demonstrate the required band identity, ordering and metadata; its existence
alone does not establish suitability.

## Cleanup first

1. Remove the rejected generic indexed-write dispatcher and catalog update
   checks from the conversion design. Restore direct native pixel writer
   ownership while keeping catalog convenience as composition.
2. Replace overlapping Item construction paths with one xarray-to-Item builder.
   File discovery, reading, schema comparison and frame conversion are separate
   responsibilities and must not be hidden inside this builder.
3. Remove post-conversion datetime/raster_metadata patches. Item construction
   must supply the complete supported description before table conversion.
4. Delete STAC column/property whitelists and redundant converter logic. A
   GeoVector Item factory, if retained, only delegates the native conversion.
5. Audit custom raster_metadata/geosave:* fields and their actual reader/model
   consumers. Keep only a demonstrated invariant unavailable from native file
   metadata or common STAC conventions. Migrate real consumers together; do not
   keep fields or compatibility paths solely for old tests.
6. Replace tests of obsolete implementation details with tests of complete
   Items, physical asset metadata and native library conversion.

## Decisions still to establish

- Temporal granularity: recommended TIFF representation is one Item per
  acquisition, with split bands as assets in that Item. Multi-time native stores
  and independently dated groups need their own explicit mapping.
- Dataset/DataTree asset identity: preserve variable and group meaning through
  a demonstrated standard mapping, without another general reconstruction codec.
- Which xarray attrs map to common metadata/extensions, and which remain internal.
- Which native generators can supply asset metadata without eager pixel reads.

Settle these before implementing the replacement Item builder. Catalog update
policy and arbitrary xarray reconstruction are not part of this conversion task.

## Smoke evidence

A disposable test wrote actual joined and split COGs, including split uint16 and
float32 files. Native PySTAC Items passed through stac-geoparquet Arrow,
GeoDataFrame, Parquet and back to Item dictionaries. Band names and dtypes
matched the physical files; quality/surveyor Item properties became columns;
asset band metadata remained nested. No GeoSave reconstruction fields or
post-conversion frame patches were used. This proves the library seam, not a
finished xarray-to-Item builder.

Sources:
- [STAC common metadata](https://github.com/radiantearth/stac-spec/blob/v1.1.0/commons/common-metadata.md)
- [STAC assets](https://github.com/radiantearth/stac-spec/blob/v1.1.0/commons/assets.md)
- [stac-geoparquet conversion and I/O](https://stac-utils.github.io/stac-geoparquet/latest/usage/)

## Public skeletons established

- `stac.asset.from_xarray(data, href, *, media_type, roles, properties) -> pystac.Asset`
- `stac.item.from_xarray(data, *, assets: Mapping[str, pystac.Asset], id, datetime, geometry, properties) -> pystac.Item`
- Catalog and row raster/stack readers retain their signatures and raise
  `NotImplementedError` until their native asset mapping is established.
- Pixel writers retain `catalog=` and reject it before side effects.
- `GeoVector.from_items(items)` delegates to stac-geoparquet today.
- `frame.gs.to_geoparquet(path, stac=True)` explicitly selects the native
  STAC serializer; ordinary GeoParquet uses GeoPandas. No STAC column guessing.
- Dense catalog workflows are skeletons. Independent ingestion, pixel storage,
  spatial operations and explicit reference cropping remain implemented.

The old STAC reconstruction fields have no persistence producer. Internal
training window time/band context stays with `transform.vector`, its actual
consumer boundary, rather than the STAC package. Tests of that context remain.

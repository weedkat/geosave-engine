# Catalog and GeoStack naming

## Intent

Use native STAC and xarray terminology consistently. Keep spatial/time queries
on GeoDataFrames, remove the pandas Series GeoRow accessor, and preserve lazy
loading, pixel-window behavior, and GeoStack persistence. Implemented by the
catalog raster indexing plan linked below.

## Vocabulary

- Collection: STAC dataset/product identity shared by related Items.
- Item: one STAC record represented by one GeoDataFrame row.
- Asset: a file/store reference under an Item's assets mapping.
- Variable: a named xarray Dataset data variable.
- Band: a raster band, including an explicitly named DataArray band axis.
- Group: a named Dataset child of a GeoStack/DataTree.

A group name is a local binding, not a Collection ID. Multiple groups may come
from the same Collection, for example Sentinel-2 data at different resolutions.
Derived rasters can also form groups without belonging to a STAC Collection.
Retain Collection identity in source metadata rather than infer it from names.

## Caller contract

A GeoDataFrame may contain Items from several Collections. Spatial/time queries
retain matching rows; they do not pair products, select bands, or align grids.
Callers select compatible records before loading one Dataset and explicitly
compose named Datasets into a stack:

```python
matches = catalog.gs.query(anchor)
optical = matches.loc[matches['collection'] == 'sentinel-2'].gs.to_raster()
labels = matches.loc[matches['collection'] == 'landcover-labels'].gs.to_raster()
scene = stack({'s2': optical, 'label': labels})
```

Catalog reader selections are named `assets` and refer to asset keys. Dataset
variable selection remains native xarray selection. GeoPackage's `layer` remains
unchanged because it names a real layer in that format.

## Ownership and migration

1. Introduce catalog reading functions in io.catalog. GeoVector.to_raster
   delegates, retaining existing multi-scene behavior and adding asset selection.
2. Move record-specific window interpretation to the chip/reference operation
   that owns it. Apply windows before combining records, preserve padding and
   laziness, and retain file-close ownership.
3. Remove GeoRow after migrating all Series accessor calls. Single-record table
   selections use iloc[[position]], preserving the GeoDataFrame.
4. Explicitly compose mixed-product catalogs with stack(mapping). Do not infer
   groups from Collection IDs or arbitrary STAC asset keys.
5. Preserve native hierarchical Zarr/NetCDF persistence and current stack STAC
   export. Preserve an explicit reconstruction route for GeoSave stack exports,
   whose asset keys intentionally encode group names.
6. Replace ambiguous layer names in raster catalog readers, directory readers,
   training internals, docs, and tests with asset or group according to meaning.
   Keep ModelSpec.rasters binding names; do not rename keys to Collection IDs.

## Refined indexing and stack reconstruction

Dataset identity is stored as geosave:raster, independently of Collection IDs.
GeoVector index properties and readers delegate to io.catalog. Saved stacks
carry ordered group bindings and a geosave:stack identity. GeoVector.to_stack
uses those bindings or an explicit group-to-asset mapping for external tables;
it never infers groups from arbitrary assets. Implementation details and checks
are in ../plans/2026-10-07-catalog-raster-indexing.md.

## Verification scope

- Multi-scene COG and whole-store Zarr/NetCDF catalog round trips.
- Saved GeoStack restoration with group identity and timeless assets preserved.
- Mixed Collections selected into explicit groups; asset keys distinct from
  variable names and group bindings.
- Pixel-window cropping/padding before combining, with lazy pixels and exact grids.
- Prepared and virtual ML samples after GeoRow removal.
- No renaming of native GeoPackage layers or existing persisted group keys.

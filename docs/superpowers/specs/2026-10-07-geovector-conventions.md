# GeoVector conventions and catalog simplification

GeoVector is a native GeoDataFrame of spatial records normalized to STAC field
conventions. Geometry/CRS provide coverage; optional datetime or start/end fields
provide time; optional assets use STAC nested mappings; other columns stay user
properties. A row does not require an invented acquisition date or raster file.

Factories and readers share GeoVector.normalize(frame, datetime_column=None).
Canonical temporal columns are UTC timestamps. Missing datetime is represented by
pandas NaT. Explicit datetime_column maps an external name, without guessing or
overwriting an existing canonical field. Preserve geometry, CRS, index and other
properties. Missing assets do not require an empty struct column in ordinary
GeoParquet. Index discovery on ordinary vectors returns empty tuples.

GeoVector.timespan gives selected records' temporal bounds, or None. Undated
records remain spatially selectable during time queries. to_anchor accepts the
existing GeoAnchor.from_geometry grid parameters; callers supply resolution or
shape, and its default time coverage comes from selected records.

GeoVector.to_items converts records to native PySTAC Items using stac-geoparquet.
It requires explicit string IDs and datetime or complete start/end timestamps;
missing assets become empty mappings at conversion. Use WGS84 geometries and
current bounds, preserving other properties and nested assets. Never invent time.
GeoJSON readers identify actual STAC Item/ItemCollection JSON and use PySTAC,
preserving relative href resolution. Ordinary GeoJSON remains native vector I/O.
STAC table GeoJSON writing serializes PySTAC ItemCollections so core fields stay
at the top level. Direct GeoJSON/GeoParquet/GeoPackage readers normalize as well.

No new table module or catalog wrapper. Existing stac/item.py owns construction
and GeoSave Item/asset identity bindings. io/catalog.py owns physical persistence,
selection, lazy opening, per-record cropping and resource ownership. Public
catalog.write writes pixels then returns Items; catalog.describe describes saved
physical inventories. Native writer outputs supply physical inventories instead
of reconstructing layouts after writing. Preserve Dataset scene grouping, stack
group order, split bands, native store time axes and repeated timeless assets.
Raster accessor to_items remains a convenience delegate to write or describe.
Remove io.catalog.to_items rather than add a compatibility alias.

Work in the current dirty workspace and preserve earlier changes. No new
dependencies, compatibility shims, table.py, automatic dates, commits or publishing.

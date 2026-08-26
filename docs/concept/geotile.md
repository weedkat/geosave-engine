# Spatial types

> **Status:** `GeoVector`, `GeoAnchor`, `GeoRaster`, `GeoTile`, `GeoMosaic`, and `GeoStitcher` are the current settled foundation. `GeoStack` and `GeoSample` are still being redesigned.

## Types

- `GeoVector`: vector features with an explicit CRS.
- `GeoAnchor`: pixel grid, declared time span, vector, and GeoSave metadata without raster pixels.
- `GeoRaster`: potentially large, lazy raster surface.
- `GeoTile`: small raster window intended for whole-array consumption.
- `GeoMosaic`: strict union of compatible raster footprints.
- `GeoStitcher`: incremental reconstruction of tiled raster or prediction outputs.
- `GeoStack`: named raster layers; current interface is provisional.
- `GeoSample`: named tile layers and model context; current interface is provisional.

## Current rules

Pixels use the [canonical raster array contract](array-format.md). Composition is strict: callers explicitly reproject, resample, select, rename, or cast incompatible data before composing it.

Full lifecycle examples will be added after `GeoStack` and `GeoSample` settle.

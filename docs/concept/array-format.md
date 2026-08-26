# Raster array contract

> **Status:** The in-memory contract is settled. Persistence details remain under review.

## In memory

Spatial raster data is an `xr.DataArray` with exactly one of these layouts:

- `(band, y, x)`
- `(time, band, y, x)`

`band` is mandatory, ordered, uniquely named, and non-empty. A time coordinate, when present, is ordered, unique, and uses `datetime64`. Every array carries a CRS and preserves its pixel dtype and nodata declaration.

Raw NumPy pixels require explicit band names. External Xarray layouts are normalized only by format adapters before entering Spatial core.

## Persistence

- GeoTIFF uses one file band per canonical band.
- NetCDF and Zarr use a CF Dataset at the storage seam.
- CF conversion is handled by `da_to_cf` and `cf_to_da`.

Detailed attribute encoding, chunking, and compatibility guidance will be documented after the persistence interface settles.

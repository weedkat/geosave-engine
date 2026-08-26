# 2. Unregistered attrs are dropped with a warning, not rejected

Date: 2026-08-26

## Status

Accepted

## Context

`GeoHeader` resolves each attr key to a registered `GeoExtension`. A key no
extension claims has no schema to validate against and no class to become.

This library otherwise fails fast: mismatched grids, CRSs, bands and dtypes
are errors rather than silent repairs. Applying that rule here would mean
`GeoRaster.open` raising on any file carrying an attr we do not know.

Raster attrs are a loose, open format. A GeoTIFF or Zarr store picks up keys
from whatever wrote it — GDAL, another toolchain, a colleague's script, or a
newer version of this library that registered a namespace this one has not
imported. None of those make the pixels unreadable.

## Decision

`GeoHeader.__post_init__` drops an unregistered namespace with a warning and
keeps going. It does not raise.

## Consequences

- `GeoRaster.open` reads any file, whatever else wrote it. A user never has
  to strip attrs by hand before this library will look at their data.
- A store written by a newer version reopens in an older one, losing only the
  namespaces that version cannot interpret.
- The warning names the namespace and says to import its module, which is the
  fix when the namespace is one of ours and simply was not imported.
- Metadata can be lost silently-ish on a read/write round trip: an unknown
  namespace is dropped on read and therefore absent on write. The warning is
  the only signal.
- This is a deliberate exception to the fail-fast rule, and it applies to attr
  decoding only. Grid, CRS, band, time, dtype and nodata mismatches stay errors.

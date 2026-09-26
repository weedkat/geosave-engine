# Ingest and Training Preparation Split

## Purpose

Keep ordinary raster acquisition separate from building a training dataset.
The public workflows are:

```python
ingest(sources, anchor, output=..., spec=...) -> zarr_path
prepare_training(labels, sources, output=..., spec=..., pattern="**/*.tif")
    -> manifest_path
```

## Ownership

`ingest` accepts one explicit `AnchorConfig`, loads the model sources on that
anchor, and writes one Zarr stack. It does not know about labels, globs,
training samples, or catalogs.

`prepare_training` expands a local label glob, submits one `prepare_sample`
task per label, and publishes a GeoVector catalog after every sample succeeds.
Each sample contains the original `label` raster and matching model sources on
the label's exact grid and time.

Both workflows compose ordinary `RasterLoader` and `write_stack` operations.
One flow never calls the other.

## Validation

Pydantic owns primitive configuration validation and native filesystem/I/O
operations own their errors. Workflow code checks only domain invariants:

- runtime source names equal model source names;
- `label` is reserved in training preparation;
- label rasters have time;
- discovery finds at least one file;
- distinct labels do not map to the same sample Zarr.

Resume validation checks that an existing sample still has the label and
current model-source groups, one shared grid, label time, and required source
variables. Catalog footprints use EPSG:4326 while retaining each sample's
native grid metadata, and sample paths remain portable relative to the
GeoParquet file.

## Generated workspace

The ingestion script calls `prepare_training`. It owns only project defaults,
environment loading, GDAL configuration, and CLI output.


# Bulk Label Ingestion and GeoVector Catalog Design

## Purpose

Prepare a local training dataset from a tree of label rasters. Each label
defines the exact grid and time used to find model-required STAC imagery. The
flow writes one self-contained Zarr sample per label and publishes one
GeoParquet catalog through `GeoVector`.

The public workflow is dataset-oriented. Callers identify a label tree rather
than constructing anchors or invoking ingestion once per sample.

## Inputs and output

The deployable Prefect flow accepts only serializable job inputs:

```python
manifest = ingest(
    labels="data/labels",
    pattern="**/*.tif",
    sources={
        "optical": {
            "query": {},
            "load": {"chunks": {"x": 1024, "y": 1024}},
        }
    },
    output="data/prepared",
    spec="model_spec.yaml",
)
```

- `labels` is a local directory containing label rasters.
- `pattern` is a recursive glob relative to `labels` and defaults to
  `"**/*.tif"`.
- `sources` contains caller-owned runtime STAC query and loading settings.
- `output` is a new or existing local dataset directory.
- `spec` is the model-owned YAML document defining source requirements.
- The returned string is `<output>/manifest.parquet`.

Label discovery and output persistence are local-only in this version. STAC
catalogs and their raster assets may remain remote. Globs, label locations,
output locations, and runtime source settings do not belong in
`model_spec.yaml`.

## Dataset layout

The label path relative to `labels` is the stable sample identity. Its suffix
is retained in `sample_id` and replaced with `.zarr` only for the output path:

```text
data/labels/
└── train/
    └── area-01.tif

data/prepared/
├── samples/
│   └── train/
│       └── area-01.zarr
└── manifest.parquet
```

Discovery is sorted for deterministic task submission and catalog row order.
Before submitting tasks, the flow rejects a missing/non-directory label root,
an absolute pattern, no matches, non-file matches, and distinct labels that
would map to the same output.

## Prepared sample

One preparation task owns one label and produces one completed Zarr. It opens
the label once, derives its native `GeoAnchor`, and requires a timespan so STAC
matching is temporally defined. It loads every model-required source on that
anchor and validates source variables through the corresponding
`RasterRequirement`.

The written `DataTree` contains:

```text
/
├── label/       original label Dataset
├── optical/     model source Dataset
└── ...          any other model source groups
```

`label` is a reserved group name and model source requirements may not use it.
All groups must share the label's exact grid. The existing atomic local Zarr
publication behavior remains: staging is cleaned after failure and a sample
appears at its final path only after every chunk has been written.

Parallelism is across labels. Loading multiple sources for one sample remains
inside the same task so its anchor, failure, retry, and publication have one
owner. Native anchors and lazy xarray objects do not cross the Prefect seam;
the task returns only the completed sample path.

## GeoVector catalog

After every requested sample is complete, one catalog task opens each Zarr
lazily and constructs a record with:

- `sample_id`: the label's relative POSIX path, including its original suffix;
- `path`: the prepared Zarr path;
- geometry, start/end datetime, and exact grid metadata derived by
  `GeoVector.from_xarray`;
- ordered grouped variable identities from the prepared `DataTree`.

The task combines records with `GeoVector.concat` and writes
`manifest.parquet`. GeoParquet makes sample paths relative to the catalog when
possible, so moving the prepared dataset directory keeps its pointers usable.
The manifest is atomically replaced only after all requested samples are
available.

GeoVector describes usable training data only. It does not contain failed
rows, error messages, retry counts, or Prefect state.

## Resume and failure behavior

Sample paths are deterministic. A rerun handles an existing sample as follows:

1. Open it as a raster stack.
2. Require the `label` group and every current model source group.
3. Require their shared grid to match the stored label grid.
4. Reuse the sample when valid.
5. Fail without overwriting when invalid.

Successful samples from a partially failed run remain published. The catalog
is not replaced unless the complete requested label set succeeds. On rerun,
valid samples are reused and only missing samples perform STAC acquisition.
Prefect owns task failures, retries, and logs.

Adding labels beneath the same root and rerunning prepares the new samples and
atomically replaces the catalog with the complete discovered set. Removing a
label removes its row from the next catalog but does not automatically delete
its old Zarr; destructive cleanup remains explicit.

## Module changes

- `workflow.flows.ingest` becomes the bulk orchestration module described
  above.
- A single deep ingestion task owns label opening, anchor derivation, source
  loading, alignment, and sample publication.
- A catalog task owns lazy sample registration and GeoParquet publication.
- Source parsing continues to use `SourceConfig.model_validate`.
- `RasterLoader` consumes a native `GeoAnchor` inside the ingestion task.
- The public workflow `AnchorConfig` hierarchy and its tests are removed.
- The one-anchor ingestion flow is removed rather than retained as another
  execution path.
- The generated workspace `IngestManifest` CSV/Excel ledger is removed. Its
  script becomes a thin caller of the library flow with project-specific
  source settings.
- No ingestion YAML, manifest wrapper, storage registry, or compatibility
  alias is introduced.

## Validation and tests

Tests cover:

- deterministic recursive discovery and relative sample identity;
- empty, invalid, and colliding discoveries before task submission;
- a real label GeoTIFF supplying exact grid and time;
- matched STAC imagery written with the label in one Zarr `DataTree`;
- multiple labels producing multiple samples and one GeoVector catalog;
- manifest geometry, time, grid, variables, and portable sample paths;
- reruns reusing valid samples without repeating STAC acquisition;
- invalid existing samples failing without overwrite;
- partial failure preserving completed samples without replacing the catalog;
- source-name and source-setting validation before pixel work;
- scoped Ruff and type checks plus the complete workflow test suite.

Operations that promise lazy registration assert that constructing the
GeoVector records does not compute sample pixels.

## Breaking changes

The `ingest` flow no longer accepts `anchor` or a single `.zarr` output path.
It accepts a label root, pattern, and dataset output directory. Public
`AnchorConfig`, `CoordinateAnchorConfig`, `GeoJSONAnchorConfig`, and
`RasterAnchorConfig` workflow exports are removed because label rasters now
own preparation anchors. Callers needing ordinary spatial acquisition use the
native `GeoAnchor` and `StacSource` interfaces outside this training-data flow.

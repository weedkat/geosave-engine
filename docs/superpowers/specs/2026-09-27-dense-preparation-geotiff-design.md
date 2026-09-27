# Dense Preparation GeoTIFF Design

Status: awaiting written-spec review.

## Goal

Make `prepare_dense_data` produce discoverable Cloud Optimized GeoTIFF samples
by default while retaining Zarr as an explicit alternative. Preserve the label
directory tree so caller-owned `train`, `validation`, and `test` splits remain
ordinary directories, and keep each completed sample an atomic, resumable unit.

The change also removes the cross-module import of a private Prefect task,
simplifies label discovery and sample identity, and replaces the generator-based
raster-opening context manager in prediction.

## Logical sample

The label and acquired model rasters form one logical sample. They share an
exact grid and sample identity, but GeoTIFF does not combine them into one
physical file. A class label and Sentinel-2 scene may have different dtypes,
nodata values, band semantics, and metadata, so each named raster remains an
independent asset.

Zarr stores the same logical sample as named groups in one store:

```text
tile-001.zarr
├── label/
└── sentinel_2/
```

GeoTIFF stores the named rasters as sibling COGs in one sample directory:

```text
tile-001/
├── label.tif
└── sentinel_2.tif
```

The existing general-purpose `GeoStack.to_cog()` remains unchanged. It supports
arbitrary groups and time cubes whose tree depth can vary with the data. Dense
preparation instead owns a stricter single-scene layout so file discovery never
depends on whether one input happened to retain a singleton time dimension.

## Dataset layout

The label path relative to `labels` defines the sample path. Its suffix is
removed from the sample identity and output path. Every parent directory is
preserved:

```text
data/labels/
├── train/
│   └── region-a/
│       └── tile-001.tif
└── validation/
    └── region-b/
        └── tile-002.tif
```

The default GeoTIFF output is:

```text
data/prepared/
├── manifest.parquet
├── train/
│   └── region-a/
│       └── tile-001/
│           ├── label.tif
│           └── sentinel_2.tif
└── validation/
    └── region-b/
        └── tile-002/
            ├── label.tif
            └── sentinel_2.tif
```

The equivalent Zarr output is:

```text
data/prepared/
├── manifest.parquet
├── train/
│   └── region-a/
│       └── tile-001.zarr
└── validation/
    └── region-b/
        └── tile-002.zarr
```

There is no intermediate `samples/` directory. For GeoTIFF, the reserved label
asset is always `label.tif`; each model raster is `<raster-name>.tif`. Raster
names already satisfy the model specification's safe `Name` grammar.

Distinct labels that map to the same suffix-free identity, such as `a.tif` and
`a.tiff`, are rejected before task submission. Discovery order, task submission,
and manifest row order remain deterministic.

## Flow API

The flow accepts a literal format and serializable native writer options:

```python
prepare_dense_data(
    labels: str,
    *,
    output: str,
    spec: str,
    format: Literal["geotiff", "zarr"] = "geotiff",
    write_options: dict[str, JsonValue] | None = None,
    pattern: str = "**/*.tif",
    max_concurrency: PositiveInt = 1,
) -> str
```

`geotiff` names the user-facing format and writes COG files. `write_options`
passes serializable encoding settings to the selected native writer rather than
introducing a parallel output-configuration model. GeoSave retains ownership of
structural and publication settings: callers cannot override layout,
`split_bands`, `overwrite`, or deferred computation. Unknown or incompatible
options fail through the selected native writer.

The CLI adds `--format [geotiff|zarr]` and `--write-options JSON`, with the same
GeoTIFF default. The generated ingestion script exposes the same ordinary
parameters. Output format and encoding remain job-owned settings and do not
enter `model_spec.yaml`.

## Single-scene GeoTIFF contract

Every GeoTIFF asset represents one scene. A raster with no time dimension or a
one-element time dimension is accepted; the latter is reduced to a scalar time
coordinate before writing so its timestamp remains in TIFF metadata without
creating another directory level. A raster with more than one time step is
rejected with an error directing the caller to `format="zarr"`.

All variables within one named raster remain bands in one COG. This preserves a
flat sample directory while keeping label and input rasters in their native
dtypes and metadata domains.

## Persistence ownership

`workflow.tasks.save` owns the shared prepared-sample persistence seam:

- Zarr delegates to the existing atomic stack writer.
- GeoTIFF creates a temporary sibling sample directory, writes every named COG,
  reopens and validates the completed directory, then atomically renames it to
  the final path.
- Both formats refuse an existing invalid destination rather than overwriting
  it.
- A destination becomes visible only after all lazy pixels have been computed
  and all assets are complete.
- A failed write removes its staging directory and leaves no final sample.

The persistence module also owns opening a prepared sample as a native
`DataTree`: Zarr opens its groups directly, while GeoTIFF opens each flat asset
and constructs the same logical `label` plus model-raster groups. Dense sample
validation and manifest publication use this public cross-module seam rather
than importing private helpers.

## Validation and resume

Before loading STAC data, an existing destination is opened and validated. A
GeoTIFF sample directory is reusable only when:

- it contains exactly `label.tif` and one `<raster-name>.tif` for every current
  model requirement;
- the label carries a time;
- all rasters share the label's exact grid;
- every model raster satisfies its `RasterRequirement`; and
- no asset spans more than one time step.

Zarr retains the corresponding existing group, time, grid, and requirement
checks. A valid sample returns immediately without another STAC request. An
invalid existing sample raises without modifying it.

Completed samples from a partially failed run remain reusable. The manifest is
published only after every requested sample succeeds, so a failed run does not
replace an earlier complete manifest.

## Manifest

`manifest.parquet` retains one row per logical sample in stable discovery order.
Each row contains:

- suffix-free `sample_id`, such as `train/region-a/tile-001`;
- `path` to the GeoTIFF sample directory or Zarr store;
- `format`, either `geotiff` or `zarr`;
- spatial geometry, time coverage, grid metadata, and grouped variable names
  derived by reopening the persisted logical stack.

GeoParquet keeps paths relative to the manifest when possible. Individual COG
paths need no separate manifest columns because the sample layout is fixed and
each filename is its raster name.

## Module boundaries and cleanup

The Prefect task becomes public `prepare_dense_sample` and is exported from
`workflow.tasks`; the flow no longer imports `_prepare_dense_sample` from its
implementation module. Sample validation remains private to the dense task
module because no other module calls it directly.

The flow-local label helper becomes `_find_labels`. It returns stable,
suffix-free sample identities and paths without embedding either output format
in discovery.

Prediction replaces its generator decorated with `contextlib.contextmanager`
with direct resource ownership through `ExitStack`. `contextmanager` is not
deprecated, but the decorator and generator add no useful abstraction here.

## Verification

Tests lead the implementation and cover:

- the GeoTIFF default and literal format validation;
- mirrored nested `train`, `validation`, and `test` paths;
- flat `label.tif` and model-raster COG assets with real round trips;
- suffix-free sample IDs and `.tif`/`.tiff` collisions;
- writer-option forwarding and rejection of workflow-owned structural options;
- refusal of multi-time GeoTIFF rasters with a Zarr alternative;
- exact group, grid, time, and raster-requirement validation;
- valid-sample reuse without STAC access;
- atomic cleanup after a failed COG write;
- manifest-last behavior after a partial failure;
- explicit Zarr output and its existing lazy round trip;
- CLI and generated-script forwarding; and
- prediction raster lifetime without the generator context manager.

A focused real-COG smoke test runs before the implementation design is treated
as settled. The affected workflow, CLI, template, GeoTIFF, and Zarr tests then
run with scoped Ruff and BasedPyright checks plus `git diff --check`.

## Breaking changes and limits

- GeoTIFF becomes the default prepared-data format.
- Prepared samples move out of the intermediate `samples/` directory.
- Manifest sample IDs lose their source filename suffix.
- GeoTIFF preparation supports one scene per named raster; multi-temporal
  samples use Zarr.
- No compatibility aliases or duplicate legacy layout are retained during the
  alpha phase.
- Remote prepared-data publication remains outside this change.

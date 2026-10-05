# Dense manifest metadata

## Intent

`prepare-dense-data` publishes a portable GeoParquet index of completed dense
samples. The manifest describes where each sample is and enough of its spatial
and temporal shape to filter it without opening raster pixels. Raster variable
names remain in the raster files, while the model specification remains the
authority on variables required by the model.

Users may attach their own per-sample columns through one ordinary tabular
sidecar. GeoSave joins those values by a label path relative to the table file,
while each prepared sample's relative output path remains its only catalog
identity.

The change stays deliberately small. It does not introduce a manifest model,
metadata registry, remote enrichment service, or automatic country and biome
lookups.

## Command and flow contract

The CLI gains one optional input:

```bash
geosave workflow prepare-dense-data \
  --labels labels/ \
  --metadata samples.parquet \
  --output prepared/ \
  --spec model_spec.yaml
```

The `prepare_dense_data` flow gains the matching optional `metadata` path.
Omitting it preserves the simple directory-only workflow.

CSV, TSV, Parquet, and XLSX are accepted. Each contains one `label_path` column
plus any number of caller-owned columns:

```text
label_path          split   source       quality
labels/train/a.tif  train   survey-2025  0.97
labels/train/b.tif  val     survey-2025  0.91
```

`label_path` is resolved from the table file's parent directory. It must be
relative, may traverse to a sibling directory with `..`, and must resolve to a
label discovered under `--labels`. The original spelling is not copied into
the output manifest. Parquet is preferred when exact column types matter. CSV
and TSV use pandas' normal type inference; XLSX reads its first worksheet with
the project's existing OpenPyXL dependency.

## Metadata validation

Metadata is loaded and validated after label discovery but before any sample
task is submitted. When a sidecar is supplied:

- `label_path` is required, non-null, relative, string-valued, and unique after
  resolution from the table's parent directory.
- Resolved label paths exactly match the discovered label paths. Missing and
  extra paths are both errors.
- Every column name is a string.
- Caller columns may contain null values.
- `label_path` is a join key, not a caller column. The remaining caller columns
  may not collide with manifest-owned columns: `path`, `format`,
  `start_datetime`, `end_datetime`, `grid_crs`, `grid_height`, `grid_width`,
  or `geometry`.

Validation errors name the offending columns or IDs. Validating before STAC
access avoids downloading imagery for a manifest that cannot be published.
GeoSave does not add a separate schema language for custom values; pandas and
its existing PyArrow and OpenPyXL engines remain responsible for representing
the supported table values.

## Output manifest

The manifest contains these owned columns:

| Column | Meaning |
| --- | --- |
| `path` | Prepared sample path stored relative to the manifest |
| `format` | `geotiff` or `zarr` |
| `start_datetime`, `end_datetime` | Inclusive UTC temporal coverage |
| `grid_crs` | Exact native raster CRS |
| `grid_height`, `grid_width` | Native raster dimensions |
| `geometry` | WGS84 sample footprint |

Caller columns are placed after the owned non-geometry columns and before
`geometry`, in their input order. Core columns also have deterministic order.

`variables` is removed from dense manifests. The sample files already preserve
their group and variable identities, and `ModelSpec.rasters` declares and
validates model inputs. The generic `GeoVector.from_xarray(...,
fields=("variables",))` capability remains unchanged for callers that actually
need a self-describing xarray record.

`grid_transform` is also omitted from dense manifests. The raster file owns the
exact transform, and manifest consumers open that file rather than reconstruct
a raster from catalog fields. The generic `GeoVector` grid field remains
unchanged; only the dense manifest selects the smaller set of useful columns.

`sample_id` is omitted as well. It would repeat the prepared relative `path`
without adding identity. Preparation may still derive a suffix-free relative
label path internally to preserve the input directory tree, but that value is
not a manifest column. Incrementing IDs are not introduced because inserting a
newly sorted label would renumber later samples across reruns.

GeoParquet persistence already stores local asset paths relative to the
manifest and `read_vector` resolves them for use. Dense preparation keeps this
existing portable-path contract and adds a direct regression assertion on the
raw stored value.

## Data flow

1. Load the model specification and discover labels.
2. Load the optional metadata sidecar, resolve `label_path` from the table's
   parent directory, and validate its exact match with discovered labels.
3. Prepare or reuse each self-contained sample as today.
4. Open completed sample metadata without computing pixels.
5. Build one record with the owned catalog fields and that sample's caller
   columns.
6. Concatenate records, order columns deterministically, and atomically replace
   `manifest.parquet`.

Custom metadata affects only final manifest publication. Changing a split,
quality score, or other caller value does not rewrite valid raster samples.

## Failure and resume behavior

The existing atomicity contract remains:

- Invalid sidecar metadata fails before sample submission and leaves any old
  manifest untouched.
- A sample failure stops later submissions according to the existing bounded
  workflow and leaves any old manifest untouched.
- A manifest serialization failure leaves the old manifest untouched through
  the existing staged GeoParquet write.
- A rerun reuses valid prepared samples and republishes caller metadata from the
  current sidecar.

No CSV status ledger or additional error handler is introduced. Prefect owns
execution state; the GeoParquet manifest contains completed dataset records.

## Country and biome enrichment

Automatic enrichment is outside this implementation. Country boundaries,
disputed-border policy, biome dataset versions, licenses, and assignment of a
sample crossing multiple polygons are separate concerns from data preparation.

Users can produce `country_code`, `biome`, `ecoregion`, or similar columns in
the metadata sidecar with a local, versioned spatial join. A later generic
catalog-enrichment design may standardize this without coupling network APIs
or one boundary dataset to `prepare-dense-data`.

## Verification

Focused tests cover:

- manifests omit `variables` and `grid_transform`;
- raw GeoParquet stores relative paths and `read_vector` resolves them;
- CSV, TSV, Parquet, and XLSX caller columns round-trip into the manifest;
- changed metadata republishes without rewriting completed samples;
- nested labels with duplicate filename stems join unambiguously by path;
- label paths resolve relative to the sidecar, including sibling paths using
  `..`, and `label_path` is absent from the output manifest;
- missing, extra, absolute, null, duplicate, or non-string label paths,
  non-string column names, and reserved-name collisions fail before task
  submission;
- both GeoTIFF and Zarr sample formats retain time, CRS, dimensions, geometry,
  paths, format, and custom columns;
- manifest construction remains lazy with respect to sample pixels;
- existing failure and resume behavior remains intact.

The CLI test verifies `--metadata` forwarding and help text. Scoped Ruff,
focused workflow tests, the full test suite, and `git diff --check` complete the
implementation verification.

# Workflows on STAC tables

## Intent

Dense preparation and ingestion read and write samples through the library's
own readers, writers, and STAC tables. A label set is a table, a sample is a
row, and the manifest is the record of what has been written. The workflow
layer keeps only what is its own: bounded concurrency, atomic publication, and
resume.

```bash
geosave workflow prepare-dense-data --labels labels.parquet --output prepared/ --spec model_spec.yaml
geosave workflow prepare-dense-data --labels labels/        --output prepared/ --spec model_spec.yaml
geosave workflow ingest --anchor '{...}' --output scenes/s1 --spec model_spec.yaml --catalog scenes/catalog.parquet
```

```python
sample = read_stack("prepared/s1")                       # one group per layer
row = GeoVector.from_xarray(sample, id="s1")             # each group knows its file
manifest = manifest.gs.upsert(row, on="id")
```

This supersedes `2026-09-29-dense-manifest-metadata-design.md`.

## `read_stack` opens a directory

`stack.gs.to_cog(directory)` already writes one raster per group, named after
the group: `label.tif` for a flat group, `optical/<instant>.tif` for a timed
one. `read_stack` gains the matching read.

```python
read_stack("prepared/s1").gs.groups        # ('label', 'sentinel_2_l2a')
```

- Each entry of the directory is one group, named by its stem: a raster file, a
  `.zarr` store, or a sub-directory holding a tree of leaves. Each is opened
  with `read_raster`, so every group records the path it was read from.
- Entries whose name starts with `.`, and files that are not rasters, are
  skipped. Groups are ordered by name.
- A directory holding no raster raises `ValueError`.
- A `.zarr`, NetCDF, or `.safe` path keeps reading as a multi-group store.

`read_raster` on the same directory still merges every leaf into one cube, as
it does for any tree. `read_stack` is the reader that keeps layers apart.

## One sample layout and one sample task

A sample is a directory holding one raster per layer. `write_sample` is the
only writer and keeps its one workflow concern, atomic publication: it writes
into a staging directory beside the destination and renames it into place.

| Format | Layer on disk | Written with |
| --- | --- | --- |
| `geotiff` | `<layer>.tif`, one instant | one `write_cog` per layer, a singleton time kept as a scalar |
| `zarr` | `<layer>.zarr`, a time series | one `zarr.write` per layer |

Zarr samples keep one store per layer, since a row names one store per layer.
A destination named like a store (`.zarr`, `.safe`) is refused, because
`read_stack` would read it as one.

Registering a sample needs no workflow helper:

```python
folder = write_sample(rasters, output, format=format, write_options=write_options)
row = GeoVector.from_xarray(read_stack(folder), id=sample_id, properties=columns)
```

`open_sample`, `sample_assets`, `write_stack`, `write_manifest`,
`read_sample_metadata`, and the CSV, TSV, and XLSX readers are deleted.

## Labels are a table

`--labels` takes a GeoParquet STAC table, or a directory.

- **A table** has one row per label: a unique text `id` naming a folder inside
  the output, an asset named `label`, and any caller columns. It is read with
  `read_vector`. A sample's time is its label raster's own.
- **A directory** is indexed into such a table: every file matching `pattern`
  becomes `GeoVector.from_assets({"label": path}, id=<relative path without
  suffix>)`. A label carrying no time raises `ValueError("Label raster has no
  time: <path>")`.

`pattern` applies only to a directory. `--metadata` is removed: a caller column
is a column of the label table, and every column of a label row that is not an
item column is carried onto its sample row. The finished manifest takes those
columns from the label table as it is at the end of the run, so a rerun picks
up a column that was added or changed.

Splits are not a column. A caller cuts the finished manifest into one file per
split with the accessor's writer:

```python
manifest[manifest.region == "north"].gs.to_geoparquet("prepared/train.parquet")
```

## `prepare-dense-data`

```python
labels = read_labels(labels, pattern)
manifest = read_vector(path) if path.exists() else None

for label in labels:                       # bounded by max_concurrency
    reuse the row, register the folder, or submit prepare_dense_sample

as each task finishes:
    manifest = row if manifest is None else manifest.gs.upsert(row, on="id")
    manifest.gs.to_geoparquet(path, overwrite=True)
```

`prepare_dense_sample` reads the label, loads the spec's rasters on the
label's anchor, writes the sample, and returns its one-row frame. Its `sources`
come from the loaded rasters' `StacMetadata`.

**The manifest is written as samples finish.** Each write is staged and
atomic. After the last sample, the manifest is written once more holding
exactly the label ids, in label order.

**Resume reads the manifest, not the samples.**

| State of a label | Action |
| --- | --- |
| Row whose assets are exactly `label` plus the spec's rasters, each holding the spec's variables | reused, nothing is opened |
| Row whose assets or bands differ | `ValueError("Existing sample does not match model rasters")` |
| No row, sample directory exists | registered with `from_xarray(read_stack(...))`, then checked as a row |
| No row, no directory | prepared |

A file at the manifest path that is not a STAC table with `id` and `assets`
raises `ValueError` before any sample is submitted.

**Failure.** A failed sample stops further submissions, as today. Tasks
already running are waited for, and every one that completes is recorded before
the failure is raised. The manifest then holds the samples that finished,
where it used to be left untouched. A rerun continues from it.

## `ingest`

`ingest` writes a sample directory through the same task, without a label:

```python
ingest(anchor, output="scenes/s1", spec="model_spec.yaml", format="zarr", catalog=None)
```

- `output` is a directory, no longer a single `.zarr` store. `format` defaults
  to `zarr`.
- `catalog` names a GeoParquet table. When given, the scene's row is upserted
  into it on `id`, the id being the output directory's name, and the table is
  written back atomically. The catalog is validated before anything is loaded:
  it must be GeoParquet, and an existing one must have `id` and be in
  longitude/latitude.
- The flow returns the output path.

## CLI

| Command | Change |
| --- | --- |
| `prepare-dense-data` | `--labels` takes a table or a directory; `--metadata` removed |
| `ingest` | `--output` is a directory; `--format` and `--catalog` added |

## Breaking changes

- `ingest` output is a directory of per-layer rasters. Code that opened it with
  `read_stack("scene.zarr")` opens the directory instead, unchanged otherwise.
- `--metadata` and its `label_path` tables are gone.
- A failed dense run leaves a partial manifest.
- During a run the manifest's rows are in completion order; they are in label
  order once the run completes.

## Out of scope

- A prediction catalog for mapping.
- Remote sample output.
- Batching manifest writes. One rewrite per finished sample costs milliseconds
  against a download measured in seconds.

## Verification

- `read_stack(directory)` opens a `to_cog` directory, a directory of `.zarr`
  layers, and a mixed one; skips hidden and non-raster entries; refuses a
  directory holding no raster; stays lazy.
- `GeoVector.from_xarray(read_stack(directory))` keys its assets by layer.
- A label directory and the equivalent label table produce the same manifest.
- Caller columns of a label row reach its sample row, and a label without time
  is named in the error.
- The manifest holds each finished sample when a later sample fails, and a
  rerun prepares only the rest.
- A recorded row is reused without opening its files; a row whose assets do not
  match the spec is refused; a sample directory without a row is registered,
  not rewritten.
- The finished manifest follows label order and validates as STAC.
- `ingest` writes a directory `read_stack` reopens; with `catalog` it adds one
  row to a new or existing table, replacing a row that already carries its id;
  an existing output directory is refused, as today.
- Both CLI commands forward their new options; `--metadata` is rejected.

Scoped Ruff, `basedpyright`, the full test suite, and `git diff --check`
complete the implementation verification.

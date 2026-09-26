# Ingest and Dense Preparation Split

## Purpose

Keep ordinary raster acquisition separate from dense training preparation.
Task identity lives in a module namespace so public function names stay short:

```python
from geosave_engine.workflow import dense
from geosave_engine.workflow.flows import ingest

raw = ingest(sources, anchor, output=..., spec=...)
manifest = dense.prepare(labels, sources, output=..., spec=...)
dense.validate_sample(path, requirements)
```

No generic training-preparation interface, registry, or compatibility alias is
introduced. A future non-dense task can own a sibling workflow module when its
actual input and output contracts are known.

## Ownership

`ingest` accepts one explicit `AnchorConfig`, loads the model sources on that
anchor, and writes one Zarr stack. It does not know about labels, globs,
training samples, or manifests.

`workflow.dense` owns the dense contract:

- discover local label rasters;
- derive each sample anchor from its label grid and time;
- load matching model sources;
- write the label and sources as one Zarr stack;
- validate reusable samples; and
- publish the completed sample paths as a GeoParquet manifest.

The module exposes only `prepare` and `validate_sample`. Its per-label Prefect
task remains private. Generic operations remain in `workflow.tasks`:
`load_raster`, `write_stack`, and `write_manifest`.

## Data flow

```text
label tree -> dense.prepare -> private per-label tasks -> sample Zarrs
                                                   \-> manifest.parquet
```

Each sample contains the original `label` raster and matching model sources on
the label's exact grid and time. Source loading and lazy pixel persistence hold
the configured Prefect global concurrency limits. Prefect boundaries carry
serializable paths and settings, not open rasters or clients.

## Validation

Pydantic owns primitive configuration validation and native filesystem/I/O
operations own their errors. Dense preparation checks only its domain
invariants:

- runtime source names equal model source names;
- `label` is reserved in dense samples;
- label rasters have time;
- discovery finds at least one file;
- distinct labels do not map to the same sample Zarr; and
- an existing sample has the label, current source groups, one shared grid,
  label time, and required source variables.

The manifest is replaced only after every requested sample succeeds. Completed
sample stores remain reusable after a partially failed run.

## Modules and migration

- Add `workflow/dense.py` as the cohesive dense preparation namespace.
- Remove `workflow/flows/prepare_training.py` and
  `workflow/tasks/sample.py`.
- Remove the public `prepare_training` and sample-task exports without aliases.
- Update the semantic-segmentation workspace script to call `dense.prepare`.
- Mirror the new source ownership in `tests/workflow/test_dense.py`.

Tests cover the public namespace, dense sample preparation and validation,
resume behavior, manifest publication, source concurrency, and the ordinary
anchor-based `ingest` flow.

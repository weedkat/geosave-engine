# Training-first cleanup design

## Intent

GeoSave will stabilize data preparation and the first official training path
before designing production prediction. The cleanup removes unfinished
prediction interfaces, completes in-progress package moves, and exposes the
ordinary Python functions needed to acquire and preprocess model inputs.

The durable modeling direction is pure PyTorch computation. A task defines its
own tensor contract; Prefect coordinates external work but does not implement a
model's inference algorithm.

## Scope

This change will:

- complete the move from `geosave_engine.ml.cli` to
  `geosave_engine.ml.lightning.cli`;
- make semantic-segmentation `forward()` return one logits tensor and remove the
  `_logits` normalization helper while retaining `predict_step`;
- remove the unfinished dense-prediction workflow, tasks, declarations, tests,
  exports, and documentation;
- reduce `ModelSpec` to raster requirements and preprocessing declarations;
- expose model-spec-driven raster loading and preprocessing as ordinary public
  Python functions;
- reserve Prefect flows for independently runnable jobs and Prefect tasks for
  work submitted concurrently;
- finish the caller-metadata path for dense sample preparation;
- move generic table reading into public utilities;
- convert the release module into a package and colocate its Hugging Face
  persistence adapter there.

This change will not:

- design or implement production prediction;
- add few-shot, autoregressive, hosted-serving, or generic runner interfaces;
- change geodata behavior or add new ML capabilities;
- add an MLflow adapter before an MLflow publication use case exists;
- implement the generated workspace's training dataset. That is the next
  milestone after cleanup.

## Tensor model contracts

A native model is an ordinary `torch.nn.Module`. Its inputs and outputs are
tensors or an explicit structure of tensors. It does not acquire rasters,
interpret xarray objects, tile scenes, persist outputs, or depend on Prefect.

The exact forward contract belongs to the task rather than to every
`ModelChain`. Semantic segmentation has the narrow contract:

```python
def forward(self, **model_inputs: Any) -> torch.Tensor:
    """Return raw per-pixel logits."""
```

The configured chain must therefore end in one terminal tensor head. A mapping
result is an invalid semantic-segmentation model, not an alternate result shape
for the task to normalize. Training, validation, test, and prediction steps all
call `self(**model_inputs)` directly. `predict_step` remains part of model
development and returns raw tile logits with their sample indices.

Other tasks may define different pure-tensor contracts when they exist. A
few-shot model may accept support tensors, support labels, and query tensors in
one call. An autoregressive model may return next-step logits plus tensor cache
state. GeoSave will not force those tasks through the semantic-segmentation
single-logits interface.

Repeated model invocation is an inference-program concern. Test-time
adaptation, support-embedding reuse, sampling, stopping criteria, and
autoregressive loops are ordinary Torch code around the model. Their first
implementation belongs in the generated workspace that needs them. Shared
library code is introduced only after a second concrete consumer establishes a
real seam.

Prefect flows may call an inference program in the future, but they own only
external orchestration: acquiring inputs, scheduling coarse work, retries,
concurrency, and durable publication. Per-token loops and per-episode adaptation
are not Prefect tasks or subflows. A supporting operation becomes a subflow only
when it is independently runnable and produces a durable artifact, such as a
persisted support-embedding index.

## Model specification

During the training-first phase, `ModelSpec` contains:

```text
schema_version
rasters
preprocessing
```

The following premature prediction fields are removed:

```text
tiling
model_inputs
aggregation
postprocessing
exports
```

The semantic-segmentation template removes the corresponding YAML sections.
Training remains configured independently by `train.yaml`.

Model releases continue to bundle `model_spec.yaml`, but release validation no
longer compares model inputs with a nonexistent prediction contract. Release
validation checks only the construction recipe and artifact completeness until
an inference interface is designed from a real consumer.

## Public data-preparation modules

Reusable lazy work is exposed as ordinary Python functions rather than Prefect
objects.

### Raster acquisition

`geosave_engine.workflow.ingest` exposes:

```python
def load_rasters(
    anchor: GeoAnchor,
    spec: ModelSpec,
) -> dict[str, xr.Dataset]:
    """Load every required raster lazily from its model-owned STAC recipe."""
```

The function validates that every raster requirement has a STAC recipe and
delegates each individual acquisition to the existing STAC-loading
implementation. It accepts native objects so library callers do not have to
serialize an anchor or reload a specification.

`geosave_engine.workflow.flows.ingest` remains the independently runnable job.
It accepts deployment-safe values, opens the anchor, loads the model
specification, calls `load_rasters()`, and atomically writes the raster stack.
Dense sample preparation reuses `load_rasters()` directly; it does not invoke
the ingest flow as a subflow.

### Processing

`geosave_engine.workflow.processing` exposes:

```python
def run_stage(stage: StageSpec, inputs: Mapping[str, Any]) -> dict[str, Any]: ...

def preprocess(
    inputs: Mapping[str, Any],
    spec: ModelSpec,
) -> dict[str, Any]: ...
```

Both functions remain synchronous and preserve lazy xarray/Dask values. They
are not Prefect tasks or flows. `run_stage()` executes ordered model-owned calls;
`preprocess()` first selects and validates only referenced raster inputs, then
executes the preprocessing stage.

There is no public `postprocess()` during this phase because prediction is not
defined. It will return only with a concrete prediction consumer.

### Tables and sample identity

`geosave_engine.utils.read_table(path)` reads CSV, TSV, Parquet, or the first
XLSX worksheet. This is a generic public utility and has source-mirrored tests
under `tests/utils/`.

`geosave_engine.workflow.metadata.read_sample_metadata()` retains the
workflow-specific `label_path` join, exact label-set validation, and
manifest-owned-column checks.

Stable label discovery and output naming move to
`geosave_engine.workflow.samples` as public functions:

```python
def discover_labels(root: Path, pattern: str) -> dict[str, Path]: ...

def dense_sample_path(
    root: Path,
    sample_id: str,
    format: SampleFormat,
) -> Path: ...
```

Endpoint retry classification, cached STAC clients, YAML loaders, GeoTIFF scene
normalization, and completed-sample validation remain private implementation.
They do not gain leverage from a public interface.

## Prefect ownership

Only independently runnable jobs are flows:

- `ingest` creates one durable raster stack from an explicit anchor;
- `prepare_dense_data` creates or reuses dense samples and publishes a
  manifest.

Only work requiring bounded concurrency is submitted as a task:

- `prepare_dense_sample` acquires and writes one label-aligned sample.

Manifest construction and publication are called synchronously after all
sample tasks complete. Raster loading, preprocessing, sample opening and
writing, and other reusable operations are ordinary functions. This prevents
Prefect task objects from becoming the Python library interface and avoids
passing lazy in-memory xarray values through unnecessary orchestration seams.

## Dense metadata completion

The existing metadata reader is wired into the runnable preparation path:

1. Discover label rasters.
2. Read and validate the optional metadata table before submitting work.
3. Submit at most `max_concurrency` sample preparations.
4. Call manifest publication synchronously with completed sample paths and the
   metadata mapping.

The CLI and generated ingestion script accept the optional metadata table.
Invalid metadata therefore fails before STAC access and cannot leave a partially
updated manifest.

## Release package

The release implementation becomes:

```text
geosave_engine/release/
├── __init__.py
└── huggingface.py
```

The existing public imports remain:

```python
from geosave_engine.release import load_model, load_spec, publish_model, save_model
```

`release.huggingface` owns `GeoSaveConfig`, `GeoSaveModel`, and Transformers
registration. Tests move from the ML test tree to the source-mirrored release
test tree. Imports of the optional Transformers adapter remain lazy so
`load_spec()` does not import Transformers.

No publisher protocol or registry is added. Hugging Face is the only current
adapter; a future `release.mlflow` module should be designed from actual MLflow
artifact behavior.

## Deletions and stale references

The cleanup removes:

- `workflow/prediction.py`;
- `workflow/flows/predict.py`;
- `workflow/tasks/predict.py`;
- `workflow/specs/prediction.py`;
- their source-mirrored tests;
- prediction-only model-spec and template declarations;
- postprocessing execution and tests;
- stale prediction design and implementation-plan documents;
- all imports of the deleted `geosave_engine.ml.cli` path;
- any compatibility aliases for removed Alpha interfaces.

Current geodata implementations and behavior remain unchanged.

## Failure behavior

- A semantic-segmentation chain returning anything other than a tensor raises a
  direct `TypeError` at `SemanticSegmentationTask.forward()`.
- Missing STAC recipes fail in `load_rasters()` before any raster is loaded.
- Preprocessing validates all references before invoking the first declared
  call and preserves contextual error notes from `CallSpec.invoke()`.
- Invalid metadata fails before sample tasks are submitted.
- Existing atomic write and resume behavior remains unchanged.
- Removed prediction configuration is rejected by strict `ModelSpec`
  validation rather than ignored.

## Verification

Implementation follows focused test-first slices:

1. CLI import and semantic tensor-contract tests.
2. Absence/rejection tests for removed prediction interfaces and YAML fields.
3. Public `load_rasters()` tests using native anchors and model specifications.
4. Plain `preprocess()` laziness, validation, and ordered-execution tests.
5. Public table and sample-identity utility tests.
6. Dense metadata CLI-to-manifest tests, including failure before submission.
7. Release package round trips and a fresh-process adapter load.

The final verification runs focused workflow/template/ML/release tests, the
complete test suite, scoped Ruff, BasedPyright, and `git diff --check`.

The next milestone uses the cleaned interfaces to build a template-owned
manifest Dataset/DataModule and proves one real preparation-to-training run.

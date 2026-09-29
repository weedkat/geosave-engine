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
- expose raster loading and preprocessing as behavior on `ModelSpec` rather
  than external functions that unpack it;
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

The task implementation is intentionally direct:

```python
def forward(self, **model_inputs: Any) -> torch.Tensor:
    """Return raw per-pixel logits for prepared model inputs."""
    logits = self.model(**model_inputs)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Semantic segmentation models must return logits as a tensor")
    return logits

def predict_step(
    self,
    batch: tuple[dict[str, Any], Any],
    batch_idx: int,
    dataloader_idx: int = 0,
) -> tuple[torch.Tensor, Any]:
    model_inputs, index = batch
    return self(**model_inputs), index
```

The same direct call replaces `_logits(self(...))` in the train, validation,
and test steps. The current mapped-result fixture becomes a rejection test;
the existing Lightning tile prediction and explicit `predict_step` tests stay.

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

Removing `model_inputs` deliberately leaves tensor-input selection with the
first generated training Dataset. For semantic segmentation that Dataset will
select the preprocessing output named `image`, convert the cropped value to a
tensor, and return it under the model input named `image`. GeoSave will not
infer that the last preprocessing result is a model input or add an output
declaration until the real training path demonstrates a reusable requirement.

Model releases continue to bundle `model_spec.yaml`, but release validation no
longer compares model inputs with a nonexistent prediction contract. Release
validation checks only the construction recipe and artifact completeness until
an inference interface is designed from a real consumer.

## Behavior-rich model specification

`ModelSpec` is the primary public module for behavior completely described by
`model_spec.yaml`. Callers should not extract its declarations and pass them to
unrelated public functions.

```python
model = ModelSpec.load("configs/model_spec.yaml")
rasters = model.load_rasters(anchor)
prepared = model.preprocess(rasters)
```

`ModelSpec.load()` remains inert: it parses and validates declarations without
opening endpoints, importing declared calls, or invoking code. Network and
call execution begin only through the two explicit runtime methods.

### Raster acquisition

`ModelSpec` exposes:

```python
def load_rasters(
    self,
    anchor: GeoAnchor,
) -> dict[str, xr.Dataset]:
    """Load every required raster from its model-owned STAC recipe."""
```

The method revalidates the specification, checks that every raster has a STAC
recipe before opening the first endpoint, and loads rasters in declaration
order. Each result is selected and validated against its
`RasterRequirement`. Missing-recipe errors list every affected raster;
acquisition and validation failures retain their native type with the raster
name added as context.

STAC catalog and search metadata access is necessarily eager. GeoSave does not
call `compute()` on raster pixels, and the default recipe remains Dask-backed;
an explicit `chunks: null` or a user-declared eager operation may still
materialize pixels. The contract therefore promises that GeoSave adds no
implicit eager pixel computation, not that arbitrary recipes are always lazy.

Cached clients, endpoint retry classification, and fresh mutable sources stay
inside a private acquisition implementation. There is one concrete STAC
adapter, so no loader protocol, registry, or injected callable is introduced.
Individual raster acquisition is not a public interface in this change; it
would bypass the model-level all-recipes-present preflight without a current
caller that needs that behavior.

`geosave_engine.workflow.flows.ingest` remains the independently runnable job.
It accepts deployment-safe values, loads the model specification, performs the
missing-recipe preflight, opens the anchor, calls `model.load_rasters()`, and
atomically writes the raster stack. Dense sample preparation receives the
complete `ModelSpec` and calls the same method directly; it does not receive an
extracted requirements mapping or invoke the ingest flow as a subflow.

The ingest flow retains one deliberate declaration inspection: it checks for
missing STAC recipes before opening an anchor that may itself require I/O.
`load_rasters()` repeats the preflight before endpoint access. This preserves
fail-fast behavior without adding a shallow third method such as
`validate_for_ingest()`.

### Processing

`StageSpec` owns ordered execution:

```python
def run(self, inputs: Mapping[str, Any], /) -> dict[str, Any]: ...
```

It validates missing and forward references before invoking the first call,
threads earlier outputs into later calls, permits explicit rebinding, imports
targets only during execution, and returns only declared results in declaration
order. `CallSpec.invoke()` continues to own target resolution, argument binding,
and per-call error context.

`ModelSpec` exposes the normal model-level entry point:

```python
def preprocess(
    self,
    inputs: Mapping[str, xr.Dataset],
) -> dict[str, Any]: ...
```

The method revalidates the specification, selects and validates only referenced
declared rasters, then delegates ordered execution to
`self.preprocessing.run()`. Preprocessing declarations may consume only model
rasters or results assigned earlier in the stage; runtime-only external roots
are rejected by `ModelSpec` validation so hidden task state cannot become part
of a model recipe. This validation examines references but never resolves or
imports their call targets.

Both methods are synchronous and are not Prefect tasks or flows. They do not
mutate the caller's mapping, although an explicitly declared callable may
mutate an object it receives. Loading YAML from an untrusted source remains
safe and inert; invoking `preprocess()` executes declared Python calls and is a
trusted-spec operation.

There is no public `postprocess()` during this phase because prediction is not
defined. It will return only with a concrete prediction consumer.

Deterministic model-required preparation belongs in `ModelSpec.preprocessing`.
Random augmentation, cropping, episode construction, label conversion, and
selection of model keyword arguments remain Dataset/DataModule policy.

### Concrete implementation

`StageSpec.run()` absorbs the current external `run_stage()` implementation:

```python
def run(self, inputs: Mapping[str, Any], /) -> dict[str, Any]:
    """Execute declarations in order after validating their references."""
    self.validate_inputs(inputs.keys())
    state = dict(inputs)
    results = {}
    for output, declaration in self.items():
        result = declaration.invoke(declaration.select_inputs(state))
        state[output] = result
        results[output] = result
    return results
```

`ModelSpec` owns structural validation and the two normal runtime operations:

```python
@model_validator(mode="after")
def _validate_preprocessing_roots(self) -> Self:
    unknown = set(self.preprocessing.external_inputs) - self.rasters.keys()
    if unknown:
        raise ValueError(
            f"Preprocessing inputs must be declared rasters: {sorted(unknown)}"
        )
    return self

def _validated(self) -> Self:
    return type(self).model_validate(self.model_dump())

def load_rasters(self, anchor: GeoAnchor, /) -> dict[str, xr.Dataset]:
    from geosave_engine.workflow import _acquisition

    model = self._validated()
    missing = [
        name for name, requirement in model.rasters.items()
        if requirement.stac is None
    ]
    if missing:
        raise ValueError(f"STAC recipes required for rasters: {missing}")

    rasters = {}
    for name, requirement in model.rasters.items():
        try:
            rasters[name] = _acquisition.load_raster(anchor, requirement)
        except Exception as error:
            error.add_note(f"While loading raster {name!r}")
            raise
    return rasters

def preprocess(
    self,
    inputs: Mapping[str, xr.Dataset],
    /,
) -> dict[str, Any]:
    model = self._validated()
    state = dict(inputs)
    for name in model.preprocessing.external_inputs:
        if name not in state:
            continue  # StageSpec.run() reports missing references.
        try:
            state[name] = model.rasters[name].select_raster(state[name])
        except (TypeError, ValueError) as error:
            error.add_note(f"While preprocessing raster {name!r}")
            raise
    return model.preprocessing.run(state)
```

The private acquisition implementation contains the existing STAC mechanics:

```python
def load_raster(
    anchor: GeoAnchor,
    requirement: RasterRequirement,
) -> xr.Dataset:
    recipe = requirement.stac
    if recipe is None:
        raise ValueError("Raster requirement needs a STAC recipe")
    client = _open_client(
        recipe.collection,
        tuple(str(endpoint) for endpoint in recipe.endpoints),
    )
    source = client.source(recipe.collection)
    source.query = recipe.query.to_query(recipe.collection)
    source.config = recipe.load
    return requirement.select_raster(source.load(anchor))
```

`_open_client()` retains cached `StacClient` instances and constructs a fresh
mutable source in `load_raster()`. A `ValueError` raised specifically by
`StacClient.collection()` means that endpoint lacks the collection and permits
fallback. Errors from opening the endpoint, parsing malformed collection
documents, authentication, and invalid queries continue to propagate.

Because `StacClient.collection()` currently uses `ValueError` for both its
documented missing-collection result and errors raised while parsing a returned
document, `_open_client()` matches only the former's exact message:

```python
try:
    metadata = client.collection(collection)
except ValueError as error:
    expected = (
        f"collection {collection!r} not found on this STAC endpoint; "
        "call collections() to see what is available"
    )
    if str(error) != expected:
        raise
    metadata = None
```

This is a local correction to the adapter's existing fallback policy, not a
change to the frozen geodata STAC API. The local import in `load_rasters()`
also avoids making the private acquisition module part of spec import-time
validation.

The runnable ingest flow becomes a small deployment adapter:

```python
@flow(name="ingest", persist_result=False)
def ingest(anchor: dict[str, JsonValue], *, output: str, spec: str) -> str:
    model = ModelSpec.load(spec)
    missing = [
        name for name, requirement in model.rasters.items()
        if requirement.stac is None
    ]
    if missing:
        raise ValueError(f"STAC recipes required for rasters: {missing}")
    target = TypeAdapter(AnchorConfig).validate_python(anchor).open()
    return write_stack(model.load_rasters(target), output)
```

The repeated missing-recipe comprehension is intentional and tested: the flow
must fail before opening a potentially remote anchor, while
`ModelSpec.load_rasters()` must enforce its own interface for direct callers.
It is not extracted into another public validation method.

### Tables and sample identity

`geosave_engine.utils.read_table(path)` reads CSV, TSV, Parquet, or the first
XLSX worksheet. This is a generic public utility and has source-mirrored tests
under `tests/utils/`.

`geosave_engine.workflow.metadata.read_sample_metadata()` retains the
workflow-specific `label_path` join, exact label-set validation, and
manifest-owned-column checks.

`geosave_engine.workflow.catalog` owns the manifest schema and exposes its
ordered owned-column names once. Metadata validation and manifest publication
both use that definition rather than maintaining duplicate reserved-name sets.
`write_manifest()` is an ordinary synchronous function in that module.

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

### Physical locality

The source layout reflects runtime responsibility rather than placing every
workflow helper under `tasks`:

```text
workflow/
├── _acquisition.py       # private STAC endpoint and source implementation
├── catalog.py            # manifest schema and synchronous publication
├── metadata.py           # caller metadata join and validation
├── samples.py            # label discovery and output identity
├── storage.py            # atomic stack/sample open and write functions
├── flows/                # independently runnable jobs
├── specs/                # behavior-rich declarative model modules
└── tasks/                # only concurrently submitted work
```

The normal public seam stays on `ModelSpec`; the private acquisition module
keeps network mechanics out of `specs/model.py` without creating another public
loader. `StageSpec.run()` lives with `StageSpec`. Source-mirrored tests move with
these modules instead of preserving their old `tasks` locations.

The physical moves are explicit:

| Current code | Target owner |
| --- | --- |
| `workflow/tasks/load.py` | `workflow/_acquisition.py` |
| `workflow/tasks/process.py::run_stage` | `workflow/specs/stage.py::StageSpec.run` |
| `workflow/tasks/process.py::preprocess` | `workflow/specs/model.py::ModelSpec.preprocess` |
| `workflow/tasks/process.py::postprocess` | deleted |
| `workflow/tasks/save.py` | `workflow/storage.py` |
| `workflow/tasks/catalog.py` | `workflow/catalog.py` |
| private label/path helpers in `flows/prepare_dense_data.py` | `workflow/samples.py` |
| `ml/huggingface.py` | `release/huggingface.py` |
| top-level `release.py` | `release/__init__.py` |

After the move, `geosave_engine.workflow.tasks` exports only
`prepare_dense_sample`. Existing tests move to the source-mirrored target
locations; no forwarding modules or import aliases remain for these Alpha
interfaces.

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

Dense preparation continues to persist selected source rasters plus labels; it
does not silently materialize preprocessing outputs. The generated Dataset
loads one `ModelSpec`, keeps each sample open while preprocessing and tensor
conversion read it, and applies `model.preprocess()` to cropped sample values.
Expensive deterministic preprocessing may be cached later only after the first
training run provides evidence that this is useful.

The submitted task receives the whole model specification, which makes the
public behavior the only acquisition path:

```python
@task(cache_policy=NO_CACHE, persist_result=False)
def prepare_dense_sample(
    label: str | Path,
    model: ModelSpec,
    output: str | Path,
    *,
    format: SampleFormat = "geotiff",
    write_options: Mapping[str, JsonValue] | None = None,
) -> str:
    destination = Path(output)
    if destination.exists():
        _validate_dense_sample(destination, model.rasters, format=format)
        return str(destination)

    with io.read_raster(label) as label_raster:
        anchor = label_raster.gs.anchor
        if anchor.timespan is None:
            raise ValueError(f"Label raster has no time: {label}")
        return write_sample(
            {"label": label_raster, **model.load_rasters(anchor)},
            destination,
            format=format,
            write_options=write_options,
        )
```

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

The flow's setup and finalization become:

```python
model = ModelSpec.load(spec)
if "label" in model.rasters:
    raise ValueError("Model raster name 'label' is reserved")

discovered = discover_labels(Path(labels), pattern)
properties = read_sample_metadata(metadata, discovered)

# The existing bounded loop submits:
prepare_dense_sample.submit(
    label,
    model,
    dense_sample_path(destination, sample_id, format),
    format=format,
    write_options=write_options,
)

# Once every future is resolved, finalize synchronously:
return write_manifest(
    ordered,
    destination / "manifest.parquet",
    format=format,
    metadata=properties,
)
```

`metadata` is added to both the flow and CLI command as
`str | None = None`. Recipe validation happens through the `ModelSpec` passed
to each submitted sample. Metadata and label validation happen before any
submission. A separate flow-level missing-recipe check remains useful here too:
it guarantees every recipe error is reported before a worker starts STAC I/O.

The CLI forwards `metadata=str(metadata) if metadata is not None else None`;
the generated preparation script exposes the same optional argument. Neither
layer reads the table itself.

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

`release.__init__` keeps the existing save/load/publish functions and changes
only their local lazy imports:

```python
def save_model(...):
    from .huggingface import GeoSaveModel
    ...

def load_model(...):
    from .huggingface import GeoSaveModel
    ...
```

Release validation retains `json.dumps(model.stage_specs)` and complete-file
checks, but removes the comparison against `spec.model_inputs` because that
prediction-only declaration is removed. `_validate_release` consequently takes
only `model`; `save_model` still validates and bundles the supplied `ModelSpec`.

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

Every remaining `GeosaveCLI` import changes directly to
`geosave_engine.ml.lightning.cli`. The new package path is the only path; the
deleted `geosave_engine.ml.cli` module is not restored as a forwarding alias.

The existing `values.yaml` test fixture mixes generic stage execution with a
model recipe by supplying scalar runtime roots such as `scale`, `value`, and
`audit`. Those cases move to direct `StageSpec` tests. Model-spec fixtures use
only declared raster roots, proving that model preprocessing is self-contained.

Current geodata implementations and behavior remain unchanged.

The existing public `validated_copy()` helper becomes private implementation.
Pydantic's frozen model does not deeply freeze nested dictionaries, so runtime
methods and saving still revalidate an independent copy before acting; callers
do not need that mechanism as a separate interface.

## Failure behavior

- A semantic-segmentation chain returning anything other than a tensor raises a
  direct `TypeError` at `SemanticSegmentationTask.forward()`.
- Missing STAC recipes fail in `load_rasters()` before any raster is loaded.
- A raster without a STAC recipe remains valid when callers supply that raster
  directly to `preprocess()`; only acquisition requires every recipe.
- Endpoint fallback treats the current `StacClient.collection()` missing-
  collection `ValueError` as an unavailable collection and tries the next
  configured endpoint. Authentication, invalid query, and other non-retryable
  failures retain their native errors.
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
3. `ModelSpec.load_rasters()` tests using native anchors and specifications,
   including all-recipes preflight, endpoint fallback, and error context.
4. `StageSpec.run()` and `ModelSpec.preprocess()` tests for inert loading,
   ordered execution, rebinding, forward references, runtime-root rejection,
   raster selection, no caller-mapping mutation, and laziness.
5. Public table and sample-identity utility tests.
6. Dense metadata CLI-to-manifest tests, including failure before submission.
7. Release package round trips and a fresh-process adapter load.

The final verification runs focused workflow/template/ML/release tests, the
complete test suite, scoped Ruff, BasedPyright, and `git diff --check`.

The next milestone uses the cleaned interfaces to build a template-owned
manifest Dataset/DataModule and proves one real preparation-to-training run.
That test will explicitly lock down preprocessing output selection, tensor
conversion while sample resources remain open, target construction, and random
augmentation ownership before any of those concerns move into the library.

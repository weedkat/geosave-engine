# Workflow Processing Design

Status: implemented.

## Goal

Make every model-processing stage use one explicit call grammar. The model YAML
retains the ordered, model-specific recipe, while reusable Python operations live
in their domain libraries rather than in separate preprocessing and
postprocessing utility layers.

## Model document

`model_spec.yaml` remains the model-owned contract. Its three processing fields
all use the same ordered `StageSpec` declaration:

```python
class ModelSpec(SpecModel):
    filename: ClassVar[str] = "model_spec.yaml"

    preprocessing: StageSpec = Field(default_factory=StageSpec)
    inference: StageSpec = Field(default_factory=StageSpec)
    postprocessing: StageSpec = Field(default_factory=StageSpec)
```

The separate `PostprocessingSpec` has no distinct invariant and is removed.
Stage names remain separate because they execute at different lifecycle points,
not because they use different declaration or operation types.

`ModelSpec.resolve_path` remains owned by `ModelSpec` but becomes a staticmethod.
Path resolution does not vary by subclass. Directory paths resolve against
`ModelSpec.filename`; its default remains `model_spec.yaml`. Explicit `.yaml` and
`.yml` paths remain valid, while remote URLs and other file suffixes remain
invalid for local save/load.

## One call grammar

Every `CallSpec` names either an importable library function or a referenced
bound method and supplies all remaining arguments through `kwargs`:

```python
class CallSpec(SpecModel):
    call: str | Ref
    kwargs: dict[str, CallValue] = Field(default_factory=dict)
```

An imported `call` must contain a valid dotted Python path. It is inert while the
YAML is loaded and is imported only when its stage executes. A referenced call
resolves a supplied value and follows ordinary Python attributes to a callable.
Python binds `self` when that target is a method, so invocation still passes only
the declared kwargs:

```yaml
preprocessing:
  selected:
    call: !ref optical.__getitem__
    kwargs:
      key: [red, nir]
  unpacked:
    call: !ref selected.gs.unpack
```

`Ref` may provide the call target or a value inside `kwargs`. A reference may
select a supplied root or follow ordinary Python attributes from it. Literal
strings stay literal. Lists and string-keyed mappings may contain references
recursively.

Each stage child binds the call's complete return value to its YAML key. A later
call refers to that result explicitly as its bound receiver or through its kwargs.
There is no separately configured receiver, previous-result injection, expression
language, positional argument grammar, or result unpacking.

```yaml
preprocessing:
  valid_pixels:
    call: geosave_engine.geodata.transform.nodata.to_nan
    kwargs:
      data: !ref sentinel_2_l2a
  image:
    call: geosave_engine.geodata.transform.packing.unpack
    kwargs:
      data: !ref valid_pixels
```

## Operation ownership

Python operations are grouped by the values and invariants they own, not by the
workflow stage that happens to invoke them:

- `geodata.transform` owns native raster transformations such as nodata
  conversion, unpacking, reprojection, tiling, and reconstruction.
- `geodata.features` owns derived raster features.
- `ml.transforms` owns reusable transformations of model tensors or outputs that
  are not already native object capabilities.
- Lightning tasks own training and prediction lifecycle plus learned or
  calibrated state. They may call the same transforms, but are not the only
  public route to them.

There is no Python package of generic preprocessors or postprocessors.
`workflow` resolves declarations and orchestrates calls; it does not duplicate
the operations.

YAML prefers an importable domain function when the library exposes one. It uses
a referenced bound method when the operation is natively method-shaped, such as
xarray variable selection or `.gs.to_tensor()`. GeoSave does not add a thin
wrapper merely to turn such a method into an imported function.

Segmentation tensor interpretation moves from the Lightning task module to
`ml.transforms.semantic_segmentation`. `softmax_argmax` and
`apply_thresholds` remain ordinary importable functions. The transform owns the
final label and probability dtype behavior: labels are `torch.uint8` and
probabilities are `torch.float32`. `SemanticSegmentationTask.postprocess` is
removed because it would only wrap that transform; the task's
`class_thresholds` and `ignore_index` remain the authoritative runtime values.

## Stage execution

Preprocessing and postprocessing use the public `run_stage` function in
`workflow/flows/stage.py`. It composes model-owned calls inside an active Prefect
flow without coupling the inert `StageSpec` model to orchestration. Before
submitting any Prefect task, the runner
validates every external and ordered reference.
Each call receives only the roots it references. Independent calls remain free
to overlap, while calls referring to preceding results receive their futures as
dependencies.

The runner copies the supplied state, binds each declared result in order, and
returns only results declared by the stage. Rebinding a supplied name replaces
that name for later calls without changing other aliases. Calls returning `None`
bind `None`. An empty stage returns an empty result mapping.

`preprocess` retains the behavior specific to source requirements: it validates
and selects only model sources consumed by the preprocessing stage before call
submission. `postprocess` supplies stitched inference results and task state to
the same stage machinery without imposing segmentation, raster, persistence, or
merging semantics.

The normal postprocessing inputs for semantic segmentation are a stitched
`logits` value and the loaded Lightning module under `task`:

```yaml
postprocessing:
  prediction:
    call: geosave_engine.ml.transforms.semantic_segmentation.apply_thresholds
    kwargs:
      logits: !ref logits
      thresholds: !ref task.class_thresholds
      ignore_index: !ref task.ignore_index
```

An optional mask is declared only by models whose runtime supplies one. Model
loading, tile accumulation, batch lifetime, output persistence, and the complete
prediction flow remain outside this change.

## Shipped model example

The semantic-segmentation `model_spec.yaml` uses importable library functions for
all nonempty stages. It demonstrates predecessor values passed through kwargs and
task state passed as explicit references. The YAML remains the source of operation
order and literal parameters; Python functions do not hide a model-specific
recipe.

The example's inference conversion keeps the native `.gs.to_tensor` bound-method
reference. Postprocessing points at the importable segmentation transform and
does not require a separate postprocessing schema.

## Validation and failures

- Loading and saving a model spec never imports declared functions.
- An imported call path without a module component is rejected during model
  validation.
- A referenced call target must use a valid reference path and resolve to a
  callable when its stage executes.
- Duplicate YAML keys, malformed references, unsupported literal values, and
  cyclic configuration containers remain invalid.
- Missing external roots and forward stage references fail before any task is
  submitted.
- Import failures, noncallable targets, invalid kwargs, and operation failures
  retain the declaration's call context on the raised exception.
- Inactive-stage imports never affect loading or execution of another stage.

## Tests

Tests mirror source ownership and cover:

- `ModelSpec.resolve_path` as a staticmethod using `ModelSpec.filename`, including
  directory, YAML filename, invalid suffix, and remote-path behavior.
- All three model processing fields materializing as `StageSpec` and round-tripping
  through YAML.
- Imported and referenced `CallSpec` targets, method binding, and inert loading.
- Literal and nested `Ref` kwargs, imported invocation, argument binding, fresh
  containers, `None` results, and contextual errors.
- Ordered predecessor references, rebinding, missing inputs, and forward
  references.
- Preprocessing source selection, validation-before-submission, named Prefect
  tasks, laziness, and independent-call concurrency after runner extraction.
- Postprocessing with supplied logits and task state, including an empty stage.
- Segmentation threshold, nodata-mask, output dtype, and no-logit-mutation
  behavior in `tests/ml/transforms`.
- The shipped model spec loading, round-tripping, and executing its available
  preprocessing and postprocessing declarations without eager raster computation.

## Breaking changes

- `PostprocessingSpec` and its export are removed.
- Segmentation processing helper imports move from the Lightning task module to
  `ml.transforms.semantic_segmentation`.
- `SemanticSegmentationTask.postprocess` is removed; callers use the same
  transform declared by the model YAML.

No compatibility aliases or alternate execution paths are added.

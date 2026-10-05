# Workflow spec and processing core

Status: implemented core, with inference execution and model loading deferred.
This document replaces the earlier inference sketches and dual-schema transition.
See [field definitions](2026-09-25-model-spec-fields.md) and the
[usage guide](../../../src/geosave_engine/workflow/spec/README.md).

## Model document and job inputs

The YAML accompanies the model. It defines its required data and model-specific
processing, so downstream users can prepare compatible inputs without adopting a
particular workflow or server. GeoSave reads those same declarations to configure
acquisition and processing.

Keep three responsibilities separate:

- Model spec: logical source requirements and model-specific processing calls.
- Job parameters: the requested area, dates, actual inputs and output destination.
- Runtime configuration: catalogue bindings, chunking, batching and orchestration.

Source requirements describe what is needed. Python runtime settings determine
where and how it is acquired. A new job does not rewrite the model document or
implicitly override its processing kwargs. No separate ingestion or merging YAML
is part of this design. Model-specific reconstruction may be declared inside that
model's postprocessing, with accumulators and batch lifetimes owned by its caller.

## Ownership

```text
workflow/
  spec/          ModelSpec, OperationSpec, Ref, raster requirements and YAML
  processing.py  Processor with an explicit preprocessing/postprocessing stage
  ingestion.py   Native STAC acquisition
  io.py          Native raster opening and completed persistence
  runtime.py     Decode acquisition settings into native objects
  flows.py       Explicitly imported Prefect acquisition flow

ml/
  tasks/semantic_segmentation.py  Training task and tensor interpretation
```

ML contains model development and task-specific behavior. Workflow owns the spec
and execution of its declarations. Importing workflow specs/processors does not
import Prefect or Lightning. Import `workflow.flows` explicitly for orchestration.

The previous version-1 recipes, input-binding schema, sampling/inference executor,
postprocessing executor and combined prediction flow are deleted. There are no
compatibility aliases or parallel schemas. Model loading and the replacement
inference mechanism require a separate design decision.

## One call and assignment rule

```yaml
schema_version: 2
sources:
  optical:
    type: raster
    variables: [red, nir]
preprocessing:
  selected:
    call: !ref optical.__getitem__
    kwargs:
      key: [red, nir]
  valid_pixels:
    call: !ref selected.gs.to_nan
  reflectance:
    call: !ref valid_pixels.gs.unpack
```

Each processing-stage child names the complete return value. `call` identifies
an importable module-level callable/class, or references a supplied callable or
bound method. `kwargs` supplies its actual keyword arguments. Python binds `self`
for methods; there is no extra receiver field, implicit input or return conversion.
Calls returning `None` bind `None`.

Resolve each call and its arguments before assigning its result. A supplied name
or a name from a preceding stage can be replaced. Its old value and other aliases
are not mutated by rebinding. Duplicate keys within one YAML mapping are errors;
use distinct names for intermediate results within one stage. Forward references
must already be supplied by the caller; there is no dependency graph scheduler.

## References and native YAML extension

`!ref optical` retrieves the supplied object. `!ref optical.gs.unpack` follows
Python attributes and obtains a bound method. A plain string stays a literal in
kwargs. References resolve recursively in configured lists/mappings, but resolved
Python objects are passed unchanged, without examining their contents again.
There is no expression evaluation, indexing language or automatic dictionary lookup.

The tag is a constructor hook for an inert `Ref`, not execution of the referenced
function. Parsing a spec never imports or runs its configured callables. PyYAML's
loader validates mappings and constructs reference markers; its dumper writes the
markers back as tags. Those hooks are local subclasses with their behavior in
the methods, without changing global `yaml.safe_load` behavior.

## Independent processing

```python
from geosave_engine.workflow.processing import Processor
from geosave_engine.workflow.spec import ModelSpec

spec = ModelSpec.load("model_spec.yaml")
prepared = Processor(spec, stage="preprocessing")({"optical": optical})
finish = Processor.load("model_spec.yaml", stage="postprocessing")
results = finish({"logits": logits})
```

Each invocation creates a fresh value dictionary and fresh literal kwargs
containers. Referenced native objects retain their identity and laziness. Explicit
calls can mutate their receivers; a processor does not hide that behavior or store
per-request accumulators on itself.

Only the selected stage's imports are resolved. Missing external reference roots
and relevant source requirements fail before executing operations. Results of
preceding calls become available in order. Attribute and output-type validation
remains runtime work where it depends on produced objects. An empty stage is a
no-op returning a fresh copy of the supplied mapping.

`ModelSpec.save/load` and Python construction/model_dump round-trip declarations,
references and literals. This is not conversion of arbitrary Python source code.
Formatting, comments and YAML alias identity are not part of the round-trip contract.

## Sampling and interpretation

Reuse native `window_stack`, `Tiles`, `TileDataset` and `TileMerger`. Select/order
bands with xarray. Tensor conversion uses the existing `.gs.to_tensor()` API on
bounded samples; model kwargs must not double as conversion instructions.
`Tiles` normalizes sequence tile shapes to tuples; its lazy numbering and the
merger's accumulation/draining lifecycle remain unchanged.

Segmentation's `softmax_argmax` and `apply_thresholds` live with its ML task.
Generic workflow postprocessing may invoke any configured callable and is not
restricted to segmentation or raster outputs. Spatial merging is postprocessing
when required by a model; it is not a universal inference step.

## Orchestration

Keep ordinary processing usable from a script, a Prefect job or a LitServe adapter.
The model remains plain PyTorch; Lightning owns training. Native execution settings
such as device, batch size, retries and server workers belong to the runtime, while
user parameters describe the requested inputs/area/time and outputs. Future flow
entries should keep parameters and configuration separate.

The retained ingestion flow writes a completed local artifact. Live generators,
models and accumulators are not completed job artifacts. No replacement prediction
flow, serving adapter, tensor batching policy or model loader is implemented here.

# Modular ML Construction Design

## Intent

GeoSave provides an opinionated semantic-segmentation Lightning pair for fast
training and public construction helpers for user-authored Lightning modules.
Configured object construction, model execution, training policy, data loading,
and Hugging Face publication have separate ownership.

This phase restructures those modules and removes the generic Hub mixin from
`ModelChain`. Prediction postprocessing, dense merging, threshold calibration,
and the final supervised dataset representation remain separate future work.

## Package Structure

```text
src/geosave_engine/ml/
├── data/
│   └── semantic_segmentation.py
├── encoding/
│   └── time.py
├── model_chain/
│   ├── chain.py
│   ├── published.py
│   ├── routing.py
│   └── step.py
├── models/
│   ├── decoder/
│   ├── encoder/
│   ├── head/
│   └── monolith/
├── registry/
│   ├── criterion.py
│   ├── factory.py
│   ├── model.py
│   ├── optimizer.py
│   └── scheduler.py
├── tasks/
│   └── semantic_segmentation.py
└── huggingface.py
```

`models` contains model implementations only. The vague
`ml.models.contract` package is removed without a compatibility alias. Its
public replacement is `geosave_engine.ml.model_chain`.

`encoding` contains conversions from native data and metadata into named model
inputs. It is not a generic utility package and is not nested under a particular
encoder implementation.

## Registry Ownership

`geosave_engine.ml.registry` owns construction from configuration parsed by
LightningCLI. A shared resolver validates exactly one registered `name` or
importable `class_path`, validates `init_args`, checks imported classes against
their required PyTorch base type, and returns the selected factory.

The public registry interface includes:

- `BuildSpec` and shared factory resolution.
- `CriterionSpec` and `build_criterion`.
- `StageSpec`, `register_model`, `list_models`, and `build_model`.
- `OptimizerSpec` and `build_optimizer`.
- `SchedulerSpec` and `build_scheduler`.

Each focused builder owns its domain-specific behavior. User-defined classes
remain available through `class_path`; a user need not mutate a global registry
for ordinary custom classes. Constructor errors remain native so bad YAML
arguments point to the actual PyTorch constructor.

### Criterion

`build_criterion` resolves concise registered names for GeoSave and common
PyTorch criteria or an importable `nn.Module` class, then passes `init_args`.

### Model

`build_model(stage_specs)` is the configured model-construction path. It builds
registered or importable stages in declaration order, supplies uniquely matched
`Published` constructor values from earlier stages, snapshots resolved
constructor arguments, and returns a `ModelChain`.

The returned chain retains an independent resolved `stage_specs` recipe for
inspection and optional Hugging Face export. Direct `ModelChain` construction
from module instances remains available for experimentation but has no recipe.
`ModelChain` itself does not parse registry specifications.

### Optimizer

`build_optimizer` resolves a registered or importable `Optimizer` class and
binds it to a supplied `nn.Module`.

Optional `groups` name exact direct child modules. Unknown names and parameters
shared by multiple selected children fail before optimizer construction. Frozen
parameters are excluded. Every unselected trainable parameter enters one final
default group. Models with no trainable parameters fail clearly.

No parameter-name substring heuristics or implicit freezing strategies are
restored.

### Scheduler

`build_scheduler` resolves a registered or importable `LRScheduler`, attaches it
to the supplied optimizer, and returns the Lightning scheduler mapping with any
declared `interval`, `frequency`, `monitor`, `strict`, or `name` metadata. The
task does not parse scheduler classes or metadata itself.

## Model Chain Ownership

`geosave_engine.ml.model_chain` owns composition and execution over actual
PyTorch modules:

- `chain.py` registers modules, accepts named inputs, executes selected steps,
  and exposes a copied construction recipe when the registry supplied one.
- `step.py` declares `@chain_step`, validates annotations, invokes the selected
  method, and validates its output.
- `routing.py` selects one declared method per stage and orders dependencies by
  matching named, typed inputs and outputs.
- `published.py` defines `Published[T]` and resolves constructor values offered
  by earlier stages.

Only `ModelChain`, `chain_step`, and `Published` are public. Introspection,
routing, and constructor-wiring helpers are implementation details.

The readability pass preserves behavior while replacing vague names, reducing
duplicated validation, keeping helpers next to the behavior they support, and
making execution order explicit. It does not redesign the chain algorithm.

## Hugging Face Ownership

`ModelChain` subclasses only `torch.nn.Module`. It does not expose
`save_pretrained`, `from_pretrained`, `push_to_hub`, `_save_pretrained`, or any
Hugging Face mixin behavior.

`geosave_engine.ml.huggingface.GeoSaveModel`, a Transformers
`PreTrainedModel`, is the sole Hugging Face adapter. `GeoSaveModel.from_chain`
converts a configured chain's copied `stage_specs` into `GeoSaveConfig` while
sharing the trained modules and weights. Loading reconstructs the chain through
`build_model` and strictly loads the exported state.

A chain built directly from module instances runs normally but export rejects it
because it has no reproducible construction recipe. Local saving, remote upload,
AutoModel registration, and remote-code export are tested only through
`GeoSaveModel`.

After removing `PyTorchModelHubMixin`, the direct `huggingface-hub` runtime
dependency is removed if no other base installation path imports it. Transformers
and its Hub dependency remain under the existing `hub` optional dependency.

## Built-in Lightning Pair

`SemanticSegmentationTask` remains a supported `LightningModule`. It owns
task-specific training steps, metrics, and threshold state, while delegating
model, criterion, optimizer, and scheduler construction to the public registry
builders.

`SemanticSegmentationDataModule` moves into the GeoSave library as its matching
`LightningDataModule`. This phase supplies the normal Lightning lifecycle and
loader structure but leaves its concrete supervised PyTorch `Dataset` factory
explicitly unimplemented. No LitData dependency or nested dataset-construction
grammar is introduced.

LightningCLI independently instantiates the pair through top-level
`model.class_path` and `data.class_path` declarations. Generated workspace YAML
points both paths at the library implementations. Users replace either top-level
class when their training or data behavior differs.

## Model Inputs and Context

Image tensors, acquisition time, geographic coordinates, prompts, and masks are
all named arguments to `model(**inputs)`. The model spec must not create a
separate `context` category that makes some forward arguments structurally
special.

Constructor constants such as channel counts and wavelengths stay in model
registry configuration. Per-sample values derived from native raster metadata
belong in model-spec inference inputs. Request-time values use explicit
references in those same inputs. Training batches produce the identical named
mapping expected during inference.

`ml.encoding` owns reusable conversions from native xarray/geodata values to
those model inputs. The shared datetime validation currently under
`ml.models.context.time` moves to `ml.encoding.time`; Clay and Prithvi remain
its consumers during this phase. A later model-spec inference change will expose
model-specific functions such as `prithvi_time` and `clay_time` directly from
YAML and remove encoder-owned context callbacks.

The current phase does not add an incomplete `inference.inputs` schema or a
second execution path. That schema and runtime migrate together when inference
orchestration is implemented.

## Migration

All live source, tests, templates, and documentation move to
`geosave_engine.ml.model_chain`. The old `ml.models.contract` import path and
ModelChain Hub methods are breaking removals with no aliases. Notebook files are
out of scope.

Tests move from `tests/ml/models/contract` to `tests/ml/model_chain`. Hugging
Face adapter tests move to a dedicated Hugging Face test module rather than
remaining chain tests. Obsolete ModelChain mixin tests are deleted, and adapter
round trips retain fresh-process coverage.

## Deferred Work

- The concrete supervised semantic-segmentation `Dataset`.
- Model-spec inference input declarations and their runtime execution.
- Model-specific postprocessing and output types.
- Threshold calibration ownership and export with inference artifacts.
- Orchestration of tile inference, dense-logit merging, and scene output.

## Verification

Focused tests prove:

- every builder supports registered names and importable classes;
- malformed selectors, wrong base classes, invalid model groups, and native
  constructor failures behave consistently;
- model construction preserves stage order, published values, and independent
  resolved specifications;
- direct and configured model chains preserve routing and runtime validation;
- `ModelChain` has no Hugging Face persistence methods;
- `GeoSaveModel` alone saves, reloads, and publishes identical weights and
  outputs, including a fresh offline process;
- `SemanticSegmentationTask` delegates all construction;
- LightningCLI instantiates the built-in task and matching data-module skeleton
  from generated YAML;
- the full suite, scoped Ruff, and diff checks pass.

# ML construction and training configuration

## Intent

Make the ML construction path readable and native to PyTorch and Lightning. Keep
only the model-stage catalog as a registry, make the LightningModule own training
configuration, and ship one coherent pair of training and workflow documents in
the generated semantic-segmentation workspace.

This is an Alpha breaking change. Remove obsolete names and execution paths rather
than adding aliases or configuration adapters.

## Ownership

- `ml.registry` owns named model-stage factories and stage construction for
  `ModelChain` persistence.
- `ModelChain` owns ordered model stages and their data flow.
- `SemanticSegmentationTask` owns its criterion, optimizer parameter groups,
  learning-rate scheduler, and Lightning return structure.
- Lightning owns training orchestration and scheduler lifecycle.
- PyTorch owns optimizer, criterion, and scheduler implementations.
- `workflow.ModelSpec` owns required source data and model-specific processing.
- Task templates own the canonical example configurations copied into workspaces.

There is no general optimizer, loss, or scheduler registry. There is no shared
training-setup module until a second task demonstrates the same interface.

## Model construction

Rename the task's public `stages` argument to `model_chain`:

```yaml
model:
  class_path: geosave_engine.ml.tasks.SemanticSegmentationTask
  init_args:
    model_chain:
      encoder: {name: dinov3}
      decoder: {name: dpt}
      head: {name: dense}
```

The rename applies to the task signature, saved hyperparameters, templates, tests,
and documentation. Inside `ModelChain`, each mapping entry remains a stage and the
constructor may retain the locally precise `stages` name.

Replace the generic `BuildSpec` name with `StageSpec`. A stage still selects exactly
one registered `name` or importable `class_path`, with optional `init_args`.
`build_stages` retains published-value wiring and reproducible constructor capture.
Delete the one-line `build_model` pass-through; callers construct `ModelChain`
directly. Keep `register_model` and `list_models` as the actual registry interface.

## Criterion

Rename the task parameter `loss` to `criterion`. Its configuration is an importable
PyTorch module class and optional constructor arguments:

```yaml
criterion:
  class_path: torch.nn.CrossEntropyLoss
```

`None` selects `CrossEntropyLoss`. The task supplies its `ignore_index` unless the
criterion configuration explicitly overrides it. Constructor failures remain native
Python/PyTorch errors. `ml.loss` remains because it contains the real custom
`ProbOhemCrossEntropy2d` implementation; `ml.registry.loss` is deleted.

## Optimizer

The task accepts one optimizer configuration:

```yaml
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
    weight_decay: 0.01
  groups:
    encoder: {lr: 0.0001}
    head: {lr: 0.002}
```

- `class_path` selects a native PyTorch optimizer class.
- `init_args` are constructor defaults shared by all parameter groups.
- `groups` maps exact top-level `ModelChain` stage names to native PyTorch
  parameter-group overrides.
- Unlisted stages and direct chain parameters form one default group.
- With no `groups`, construction is the ordinary
  `optimizer(model.parameters(), **init_args)` path.
- The implementation filters parameters that do not require gradients.
- Unknown stage names fail before optimizer construction. Native PyTorch errors
  report duplicate/shared parameters and unsupported group options.

Delete `ml.optimizer`, its parameter-name heuristics, `ml.utils.torch_params`, and
`ml.registry.optimizer`. The removed `single`, `split`, `no_wd`, `freeze_encoder`,
and `layerwise` strategies receive no compatibility aliases. Freezing belongs to
model configuration. Gradient clipping, accumulation, and precision remain Trainer
settings.

The initial interface deliberately supports one optimizer. A task needing multiple
optimizers must own manual optimization and expose its own configuration. Do not add
a generic list form in this change.

## Learning-rate scheduler

Rename the task parameter `scheduler` to `lr_scheduler`:

```yaml
lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 100
  interval: epoch
  frequency: 1
```

`class_path` and `init_args` construct the native scheduler with the task's optimizer.
The optional `interval`, `frequency`, `monitor`, `strict`, and `name` fields are passed
through as Lightning scheduler metadata. `None` disables scheduling. Lightning owns
metadata validation, including monitor requirements for `ReduceLROnPlateau`.

Delete `ml.registry.scheduler`. The task returns either the optimizer or Lightning's
native `{"optimizer": ..., "lr_scheduler": ...}` mapping.

## LightningCLI

Optimization stays under `model.init_args` because it is LightningModule behavior.
GeoSave's generated entry point disables LightningCLI's automatic top-level optimizer
and scheduler injection with `auto_configure_optimizers=False`. This prevents a second
configuration seam from overriding the task's `configure_optimizers` method.

## Template documents

The semantic-segmentation template contains two complementary files:

- `configs/model.yaml`: complete LightningModule and Trainer configuration; the
  selected workspace data module remains a separate LightningCLI concern.
- `configs/model_spec.yaml`: portable workflow source and processing contract.

`model.yaml` includes the previously separate required task values (`image_key`,
`label_key`, `ignore_index`, `class_map`, and `band_map`) so it is understandable
without the vague `metadata.yaml` overlay. Delete `metadata.yaml`. Keep
`augmentation.yaml` as an optional overlay.

The template model uses `sentinel_2_l1c` and ordered bands `B02`, `B03`, `B04`, and
`B08`. The workflow spec uses the same source name and variable order:

```yaml
schema_version: 2
sources:
  sentinel_2_l1c:
    variables: [B02, B03, B04, B08]
    require_crs: true
preprocessing:
  selected:
    call: !ref sentinel_2_l1c.__getitem__
    kwargs:
      key: [B02, B03, B04, B08]
  valid_pixels:
    call: !ref selected.gs.to_nan
  image:
    call: !ref valid_pixels.gs.unpack
```

The documents stay separate: training policy does not become part of `ModelSpec`,
and workflow declarations do not become LightningCLI fields. The template and its
tests enforce these shared example values; runtime code does not add a cross-schema
adapter.

Move the canonical `model_spec.yaml` example from `workflow/examples` into the task
template. Remove the duplicate workflow example and update live documentation and
tests to point to the template-owned document. Do not migrate notebooks.

## Validation and errors

Keep checks that protect domain invariants: one stage selector, known model-chain
group names, valid imports, and required model-spec structure. Remove validation
whose only purpose was defending the old dot-encoded optimizer strategy grammar.
Let native constructor and Lightning errors surface without wrapping them in generic
registry messages.

## Verification

Tests cover:

- named and imported model stages, published values, and saved stage configuration;
- the `model_chain` rename through LightningCLI and checkpoint reload;
- default and imported criteria with `ignore_index` behavior;
- the default optimizer path and exact top-level stage parameter groups;
- arbitrary native optimizer group options;
- scheduler construction and Lightning metadata, including plateau monitoring;
- absence of top-level LightningCLI optimizer injection in the generated entry point;
- loading and lazily executing the template-owned `model_spec.yaml`;
- agreement between the template source name/band order and `image_key`/`band_map`;
- generated workspaces containing both configuration documents.

Run focused ML registry/task/CLI and workflow-spec/template tests first, then scoped
Ruff and the broader affected ML/workflow suites. Preserve unrelated working-tree
changes throughout.

## Breaking changes

- `stages` becomes `model_chain` on `SemanticSegmentationTask`.
- `loss` becomes `criterion`.
- `scheduler` becomes `lr_scheduler`.
- `BuildSpec` becomes `StageSpec`.
- `build_model`, `build_loss`, `build_optimizer`, and `build_scheduler` are removed.
- `AdamW.split` and every optimizer strategy name are removed.
- `CELoss`, `OHEMLoss`, and named scheduler aliases are removed.
- The workflow example path moves into the task template.

No compatibility aliases or dual configuration paths are included.

## Deferred

- Multiple optimizers and manual optimization.
- Nested module-path parameter groups.
- Layer-wise learning-rate decay and automatic bias/norm decay rules.
- Dataset-adapter design connecting workflow processing directly to training batches.
- Model loading and inference execution for workflow declarations.

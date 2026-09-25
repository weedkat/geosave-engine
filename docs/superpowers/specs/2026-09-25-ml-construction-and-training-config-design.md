# ML construction and training configuration

## Intent

Make the ML construction path readable and native to PyTorch and Lightning. Keep
only the model-stage catalog as a registry, make the LightningModule own training
configuration, define one direct tensor batch contract, and ship coherent training
and workflow documents in generated workspaces.

This is an Alpha breaking change. Remove obsolete names and execution paths rather
than adding aliases or configuration adapters.

## Ownership

- `ml.registry` owns named model-stage factories and stage construction for
  `ModelChain` persistence.
- `ModelChain` owns ordered model stages and their data flow.
- `SemanticSegmentationTask` owns its criterion, optimizer parameter groups,
  learning-rate scheduler, and Lightning return structure.
- A compatible `LightningDataModule` owns datasets, data loading, source-to-tensor
  conversion, label remapping, and data-side transforms.
- Lightning owns training orchestration and scheduler lifecycle.
- PyTorch owns optimizer, criterion, and scheduler implementations.
- `workflow.ModelSpec` owns required source data, model-specific processing, and
  the semantic legend for published outputs.
- Task templates own the canonical example configurations copied into workspaces.

There is no general optimizer, loss, or scheduler registry. There is no shared
training-setup module until a second task demonstrates the same interface.

## Model construction

Rename the task's public `stages` argument to `model_chain`:

```yaml
model:
  class_path: geosave_engine.ml.tasks.SemanticSegmentationTask
  init_args:
    in_channels: 4
    num_classes: 2
    model_chain:
      encoder: {name: dinov3}
      decoder: {name: dpt}
      head: {name: dense}
```

The rename applies to the task signature, saved hyperparameters, templates, tests,
and documentation. Inside `ModelChain`, each mapping entry remains a stage and the
constructor may retain the locally precise `stages` name.

`SemanticSegmentationTask` accepts explicit `in_channels` and `num_classes`
values. It supplies those authoritative values to the first and last model-chain
stages respectively; stage declarations do not repeat them. They replace the
channel counts previously inferred from `band_map` and `class_map`; the task does
not open a workflow file or inspect source metadata while constructing tensor
shapes.

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

## Batch contract

A DataModule returns DataLoaders; its datasets and optional collate functions emit
batches already shaped for the consuming LightningModule. The built-in supervised
segmentation task accepts exactly one training, validation, and test contract:

```python
model_inputs, target = batch
logits = self(**model_inputs)
```

`model_inputs` is always a `dict[str, Tensor]` whose keys are external
`ModelChain.forward` input names. A conventional image batch is
`({"image": image}, target)`. Models needing context receive it as ordinary named
inputs, for example `({"image": image, "temporal_coords": temporal_coords}, target)`.
The task does not accept a bare tensor alternative and contains no type branch for
multiple batch shapes.

Prediction uses the same two-element structure:

```python
model_inputs, index = batch
logits = self(**model_inputs)
return logits, index
```

`index` owns the reconstruction information required by `TileMerger`; it may become
a richer typed value without changing the outer tuple contract. `TileDataset` keeps
its lazy reads, tensor conversion, context extraction, and reconstruction index, but
emits the tuple above. Remove the outer `layers` and `model_context` mappings and
the task's `image_key` and `label_key` parameters.

This is an interface, not a one-to-one task/DataModule ownership rule. Multiple
DataModules may feed the same task when they emit the same batch contract. Generated
task templates may bundle one compatible DataModule for a usable starting point;
the library does not add a task-specific DataModule hierarchy or registry.

The batch is already model-ready, so `SemanticSegmentationTask` no longer owns or
applies `ImageAugmenter`. The compatible DataModule applies paired image/target
augmentation in its dataset or collate function before returning the batch.
`augmentation.yaml` therefore configures `data.init_args.augmentations`, not the
model.

## Output legend

`class_map` and `color_map` describe the meaning and presentation of a categorical
model output. They belong to a typed output legend in `model_spec.yaml`, not to
Lightning training policy or an unstructured metadata mapping:

```yaml
outputs:
  prediction:
    legend:
      class_map:
        0: background
        1: vegetation
      color_map:
        0: "#000000"
        1: "#00ff00"
```

Add a named `outputs` mapping to `ModelSpec`. Each output declaration may contain a
`Legend`; model it as an `OutputSpec` rather than an arbitrary metadata dictionary.
The semantic-segmentation template requires a legend for its categorical prediction.
Reuse the existing geodata `Legend` validation and attachment behavior instead of
creating another class/color metadata model. `color_map` remains optional.
Postprocessing attaches the declared legend to the resulting xarray label variable,
so Zarr and GeoTIFF persistence receive the same semantics.

An output name refers to a value with the same name in the completed postprocessing
namespace, including values supplied as postprocessing inputs. After operations
finish, `Processor(stage="postprocessing")` applies each declared legend to its
named xarray `DataArray` or one-variable `Dataset`. A missing name or incompatible
native object raises a direct error. Processing other stages does not apply outputs.

The Lightning task retains explicit `num_classes` because it defines tensor shape,
loss, and metric construction. It does not load `model_spec.yaml` or retain class
names and colors. Template tests check that the task's `num_classes` equals the
dense output legend size; cross-document publication validation remains deferred
with model publication. Learned `class_thresholds` remain task/checkpoint state;
`ignore_index` remains training behavior. Remove `class_map`, `color_map`, and
`band_map` from task configuration.

## Template documents

The semantic-segmentation template contains two complementary files:

- `configs/model.yaml`: complete LightningModule and Trainer configuration; the
  selected workspace DataModule remains a separate LightningCLI object.
- `configs/model_spec.yaml`: portable workflow source and processing contract.

It also contains a workspace-owned `modules/data.py` with a LitData-backed
`LightningDataModule`. Its paths, batching, workers, augmentation, and prepared
sample decoding are data configuration. Each supervised sample is decoded directly
to `(model_inputs, target)`. This is the supplied compatible adapter, not a library
DataModule coupled to `SemanticSegmentationTask`.

`model.yaml` includes the explicit `in_channels`, `num_classes`, `ignore_index`,
criterion, optimizer, scheduler, metrics, and model chain. It contains no dataset
keys, source-band names, class names, or colors. Delete `metadata.yaml`.
`augmentation.yaml` remains an optional overlay under
`data.init_args.augmentations`; the template DataModule owns its application.

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
outputs:
  prediction:
    legend:
      class_map: {0: background, 1: vegetation}
      color_map: {0: "#000000", 1: "#00ff00"}
```

The documents stay separate: training policy does not become part of `ModelSpec`,
and workflow declarations do not become LightningCLI fields. The LightningModule
does not accept a metadata path. Template tests enforce agreement between source
variable count and `in_channels`, and between legend size and `num_classes`;
runtime model construction does not add a cross-schema adapter.

Move the canonical `model_spec.yaml` example from `workflow/examples` into the task
template. Remove the duplicate workflow example and update live documentation and
tests to point to the template-owned document. Do not migrate notebooks.

## Custom task template

Add a separate custom-task template with a workspace-owned LightningModule and
LightningDataModule:

```text
templates/tasks/custom/lightning/
  description.txt
  configs/model.yaml
  modules/task.py
  modules/data.py
```

`configs/model.yaml` points LightningCLI at the workspace classes. `modules/task.py`
contains direct `training_step`, `validation_step`, and `configure_optimizers`
boilerplate using the conventional `(inputs, target)` batch; it does not inherit
segmentation-specific model-chain, criterion, metric, or output assumptions.
`modules/data.py` contains an importable DataModule and a small deterministic
in-memory Dataset so the generated workspace and CLI configuration run without
external data. Both files identify the narrow methods users replace for their own
task and data. The custom template demonstrates ordinary Lightning ownership rather
than introducing GeoSave base task or DataModule classes.

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
- supervised tuple batches with ordinary and contextual named model inputs;
- prediction tuple batches and unchanged tile reconstruction indices;
- ordinary Python unpacking or model invocation surfacing incompatible DataModule
  batches without custom defensive schema handling;
- `ModelSpec.outputs` legend YAML and Python round trips;
- attachment of the declared legend to postprocessed categorical output;
- absence of top-level LightningCLI optimizer injection in the generated entry point;
- loading and lazily executing the template-owned `model_spec.yaml`;
- agreement between template source-variable count and `in_channels`, and between
  output legend size and `num_classes`;
- generated semantic-segmentation workspaces containing both configuration documents
  and a compatible DataModule;
- generated custom-task workspaces importing and instantiating their workspace-owned
  LightningModule and LightningDataModule.

Run focused ML registry/task/CLI and workflow-spec/template tests first, then scoped
Ruff and the broader affected ML/workflow suites. Preserve unrelated working-tree
changes throughout.

## Breaking changes

- `stages` becomes `model_chain` on `SemanticSegmentationTask`.
- supervised batches become `(model_inputs, target)` and prediction batches become
  `(model_inputs, index)`; `model_inputs` is always a mapping.
- prediction steps return `(logits, index)`.
- `image_key`, `label_key`, `band_map`, and `class_map` are removed from
  `SemanticSegmentationTask`; explicit `in_channels` and `num_classes` replace
  map-length inference.
- task-owned `augmentations` and `ImageAugmenter` application are removed; the
  DataModule emits already augmented model inputs and targets.
- `loss` becomes `criterion`.
- `scheduler` becomes `lr_scheduler`.
- `BuildSpec` becomes `StageSpec`.
- `build_model`, `build_loss`, `build_optimizer`, and `build_scheduler` are removed.
- `AdamW.split` and every optimizer strategy name are removed.
- `CELoss`, `OHEMLoss`, and named scheduler aliases are removed.
- The workflow example path moves into the task template.
- `ModelSpec` gains named output legends; class and color maps move there.
- `metadata.yaml` is removed.

No compatibility aliases or dual configuration paths are included.

## Deferred

- Multiple optimizers and manual optimization.
- Nested module-path parameter groups.
- Layer-wise learning-rate decay and automatic bias/norm decay rules.
- Model loading and inference execution for workflow declarations.
- Cross-document validation when publishing a trained model with its
  `model_spec.yaml`.
- Automatically sharing the workflow output legend with optional training-only image
  logging callbacks.

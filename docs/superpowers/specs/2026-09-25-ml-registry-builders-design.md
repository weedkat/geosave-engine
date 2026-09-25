# ML Registry Builders Design

## Intent

GeoSave provides a built-in semantic-segmentation Lightning pair for fast
training while keeping its lower-level model and training construction helpers
usable from user-authored Lightning modules.

This phase restores shared optimizer and criterion construction. It does not
settle prediction postprocessing, dense merging, threshold calibration, or the
final supervised dataset representation.

## Ownership

`geosave_engine.ml.registry` owns declarative construction from LightningCLI
configuration. It validates a selector, resolves a registered name or importable
class, checks the expected PyTorch base type, and passes `init_args` to the
constructor.

The public registry interface includes:

- `BuildSpec`: a `name` or `class_path` selector with optional `init_args`.
- `CriterionSpec`: the criterion construction specification.
- `OptimizerSpec`: optimizer construction plus optional explicit parameter
  groups.
- `build_criterion(spec) -> nn.Module`.
- `build_optimizer(spec, model) -> Optimizer`.

Model construction remains owned by the existing model registry and
`ModelChain`. This phase does not add a parallel model abstraction.

## Criterion Construction

`build_criterion` resolves either a concise registered criterion name or an
importable `nn.Module` class. GeoSave registers its useful built-in criteria and
common PyTorch defaults. User-defined criteria remain available through
`class_path`; users do not need to mutate a global registry for ordinary custom
classes.

Constructor errors remain native so invalid YAML arguments identify the actual
PyTorch constructor problem.

## Optimizer Construction

`build_optimizer` resolves either a concise registered optimizer name or an
importable `Optimizer` class.

An optional `groups` mapping keys overrides by exact direct-child module name.
Each named child contributes its trainable parameters to one optimizer group.
Unknown names fail before optimizer construction. Every remaining trainable
parameter is placed in one default group. Frozen parameters are excluded.

This explicit rule works for `ModelChain` and ordinary `nn.Module` trees without
guessing from parameter-name substrings. The removed `split`, `no_wd`,
`freeze_encoder`, and layer-name heuristic strategies are not restored.

## Built-in Lightning Pair

`SemanticSegmentationTask` remains a supported `LightningModule`. It delegates
criterion and optimizer construction to the public registry helpers rather than
implementing class resolution and grouping itself. Its task-specific training,
metrics, and threshold state remain local until later phases refine them.

`SemanticSegmentationDataModule` moves into the GeoSave library as the matching
`LightningDataModule`. For this phase it is a deliberate skeleton with normal
Lightning lifecycle and loader structure. The concrete supervised PyTorch
dataset contract is deferred; no LitData or nested dataset-construction grammar
is introduced.

LightningCLI continues to instantiate the pair independently through top-level
`model.class_path` and `data.class_path` declarations. Generated workspace YAML
points both paths at the library implementations.

## Deferred Work

- Scheduler construction and Lightning scheduler metadata.
- The concrete supervised semantic-segmentation `Dataset`.
- Postprocessing interfaces and model-specific prediction interpretation.
- Threshold calibration ownership and export with inference artifacts.
- Orchestration of tile inference, dense-logit merging, and scene output.

## Verification

Focused tests will prove:

- registered names and class paths construct criteria and optimizers;
- malformed selectors and wrong base classes are rejected;
- optimizer groups select exact direct children, reject unknown names, exclude
  frozen parameters, and retain remaining trainable parameters;
- `SemanticSegmentationTask` delegates to the shared builders;
- LightningCLI instantiates the built-in task and matching data-module skeleton
  from the generated YAML;
- existing model-registry and task behavior remains intact.


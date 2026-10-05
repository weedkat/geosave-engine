# Package Restructure Design

**Status:** draft for review, 2026-10-03

## Purpose

Three features are next: a HoloViz explorer, more training modules (pixelwise
regression, classification, detection, semi-supervised methods), and a predict
workflow with postprocessing. Today's layout makes each of them touch many
packages:

- Model construction is spread over `ml/registry`, `ml/model_chain`, and
  `ml/models`, and `release` reaches into all three.
- One training setup is spread over `ml/metrics`, `ml/transforms`,
  `ml/lightning/tasks`, `ml/lightning/data`, and `ml/callbacks`.
- `geodata/datasets` imports torch although only `ml` uses it.

This restructure separates what a release contains from how it is trained. It
moves files and edits import paths. It changes no behaviour.

## Scope

In scope: moving packages, renaming two classes, deleting one empty test file,
and tests that pin the dependency direction.

Out of scope, each a later spec:

1. Prediction and postprocessing. The head owns `postprocess`; the model spec's
   `postprocessing` block carries only its keyword arguments. Calibrated
   thresholds stay in the weights.
2. New training modules and their datamodules.
3. The HoloViz explorer, including where `geodata/viz` should live.

Left alone on purpose:

- `geodata/sensors/`. Nothing imports it, but its YAML holds per-band
  wavelength, mean, std, and GSD that normalization will need.
- `geodata/viz/` and `utils/colorize.py`. Both are used by `geodata` and by
  `ml/callbacks/prediction_logger.py`; the viz spec decides their home.
- `ml/transforms/augmenter.py`. `ImageAugmenter` builds a Kornia pipeline from
  the training YAML, so the augmentation setup is editable and recorded with the
  run. Nothing imports it yet; wiring it into the training modules belongs to
  the training-modules spec.
- `workspace/`. It is a generated consumer workspace and already names paths
  that no longer exist.

## Target layout

```text
src/geosave_engine/
├── geodata/                 # no torch
├── model/                   # what a release contains; torch, no Lightning of its own
│   ├── __init__.py          # empty, so model.spec stays torch-free
│   ├── chain/               # was ml/model_chain
│   ├── factory.py           # was ml/registry/factory.py (BuildSpec)
│   ├── registry.py          # was ml/registry/model.py
│   ├── encoder/             # was ml/models/encoder, plus time.py from ml/encoding
│   ├── decoder/  head/  monolith/
│   ├── spec/                # was model_spec
│   ├── release/             # was release
│   └── README.md            # was ml/models/README.md
├── ml/                      # Lightning training
│   ├── cli.py               # was ml/lightning/cli.py
│   ├── builders/            # criterion, optimizer, scheduler; was ml/registry
│   ├── criterion/
│   ├── callbacks/           # prediction_logger.py
│   ├── datasets/            # tiles.py; was geodata/datasets
│   ├── transforms/          # augmenter.py (ImageAugmenter), shared by every head type
│   └── segmentation/
│       ├── metrics.py       # shared by every segmentation method
│       ├── calibrate.py     # ThresholdCalibrator
│       ├── transforms.py    # softmax_argmax, apply_thresholds
│       └── supervised/
│           ├── module.py    # Module(LightningModule)
│           └── data.py      # DataModule(LightningDataModule)
├── workflow/  cli/  templates/  utils/
```

Dependency direction: `geodata <- model <- ml`. `workflow` uses `geodata` and
`model.spec`. Nothing imports upward.

## Training layout

A training setup is one concrete LightningModule and the DataModule that feeds
it. The two are a matched pair: the only contract between them is the batch the
loaders yield and `training_step` receives. So each pair lives in one package,
named by the head type and the training method:

```yaml
model: {class_path: geosave_engine.ml.segmentation.supervised.Module}
data:  {class_path: geosave_engine.ml.segmentation.supervised.DataModule}
```

```python
from geosave_engine.ml.segmentation import supervised

module = supervised.Module(in_channels=4, model_chain={...})
```

- A new method for a head type is one new package beside `supervised/`. It may
  subclass `supervised.Module` and override `training_step`.
- Code every method of a head type uses, such as metrics, sits at the head
  type's root.
- A new head type is one new folder under `ml/`.
- Nothing is shared across head types until two written modules contain the
  same code. No base class or registry is added in advance.
- Only combinations that make sense are written. There is no grid to fill.

The class names `Module` and `DataModule` repeat across packages, so callers
import the package and write `supervised.Module`.

## Changes that are not pure moves

1. **Registration imports.** `ml/models/__init__.py` walks and imports every
   submodule so each `@register_model` runs. In `model/` that walk would also
   import `spec` and `release`. It is removed. `registry.build_model` and
   `registry.list_models` import the four stage packages by name instead; each
   package's `__init__` already imports every class it holds.
2. **Class names.** `SemanticSegmentationTask` becomes
   `ml.segmentation.supervised.Module`, and `SemanticSegmentationDataModule`
   becomes `ml.segmentation.supervised.DataModule`.
3. **Deleted.** `tests/ml/models/encoder/test_clay.py` is empty.

## Rules pinned by tests

`tests/test_layering.py` runs each import in a fresh interpreter:

- `geosave_engine.geodata` loads no `torch`.
- `geosave_engine.model.spec` and `geosave_engine.workflow.flows` load no `torch`.
- `geosave_engine.model.chain`, `.registry`, `.head`, and `.release` load no
  `lightning`. Encoders that wrap terratorch do load it; that is theirs.
- `geosave_engine.model_spec`, `geosave_engine.release`,
  `geosave_engine.ml.models`, `geosave_engine.ml.registry`,
  `geosave_engine.ml.lightning`, `geosave_engine.ml.metrics`,
  `geosave_engine.ml.transforms.semantic_segmentation`, and
  `geosave_engine.geodata.datasets` are not importable.

The existing suite is the behaviour check. The affected suites hold 507 passing
tests before the first move, and the same tests pass after every step with only
import lines edited.

## Breaking changes

- Every import of `geosave_engine.model_spec`, `geosave_engine.release`,
  `geosave_engine.ml.model_chain`, `.ml.registry`, `.ml.models`,
  `.ml.lightning`, `.ml.metrics`, `.ml.transforms.semantic_segmentation`,
  `.ml.encoding`, and
  `geosave_engine.geodata.datasets`. No alias is kept.
- `train.yaml` `class_path` values for the module, the datamodule, and
  `ThresholdCalibrator`.
- Saved releases are unaffected: publication already refuses `class_path`
  stages, so every release names its stages by registry key.
- A `config.yaml` saved by LightningCLI for an earlier run names the old
  `class_path` values and no longer loads without editing them.
- This supersedes Task 3 of `plans/2026-10-03-stac-objects-and-ml-datasets.md`.

## Risks

- Several files to be moved hold uncommitted work, and the four new heads are
  untracked. The moves happen on top of that work and never check out, restore,
  or stash a file.
- Renames are made file by file. No regex sweep.

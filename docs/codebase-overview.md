# Codebase Overview

## Package Layout

Main Python package lives under `src/geomoka/`.

Key modules:

- `cli/`: command-line entrypoints for training, splitting, scaffolding, templating, calibration
- `dataloader/`: dataset classes, transforms, mask conversion, wrappers
- `model/`: model builders/wrappers and architecture modules (including DINOv2 + DPT)
- `train/`: trainer loops and training orchestration
- `eval/`: evaluation pipeline and segmentation metrics
- `inference/`: inference engines (including rasterio-based paths)
- `losses/`: custom losses such as OHEM
- `util/`: distributed and general helper utilities

## Configuration and Templates

- Runtime configs: `config/`
- Built-in templates: `src/geomoka/template/config/`

Template command copies these into project `config/<name>/`.

## Data and Outputs

- Dataset storage: `dataset/`
- Experiment outputs/checkpoints: `output/`
- Utility scripts: `script/`

## Tests and Experiments

- Notebook experiments are under `test/` and `src/geomoka/template/test/`

When adding new modules, prefer keeping the same separation:

- CLI + config at boundaries
- Training/inference orchestration in dedicated subpackages
- Reusable pure components in `model/`, `dataloader/`, `eval/`, `losses/`

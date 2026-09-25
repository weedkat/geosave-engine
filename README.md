# geosave-engine

GeoSave Engine is a local-first product for building geospatial AI workflows end to end. It standardizes the full path from data acquisition, environment setup, model training, and prediction to serving-ready outputs, so teams do not need to reinvent a different workflow for every project.

It generates a ready-to-use boilerplate and applies proven best practices out of the box, including access to state-of-the-art models and multiple training methods with minimal coding. Instead of building model pipelines from scratch, users can focus on dataset creation and preprocessing, then run the resulting pipeline on fresh satellite data directly from their own machine.

Visit official Documentation : <https://weedkat.github.io/geosave-engine/>

## Features

- **Geospatial core redesign** — CF-conformant `xr.Dataset` rasters and flat,
  same-grid `xr.DataTree` stacks use `.gs` accessors and typed persistence
  adapters for Zarr, netCDF, GeoTIFF/COG, GeoJSON, GeoPackage, and GeoParquet.
  Native transforms, temporal windows, tiling, and tensor conversion support
  lazy raster processing.
- **Model workflows** — [YAML model specifications](src/geosave_engine/workflow/README.md)
  connect reusable preparation to Prefect ingestion and prediction, with named
  raster inputs, context, temporal sampling, and completed Zarr outputs.
- **DataArray features** — spectral indices and masks consume explicitly
  ordered band DataArrays and return DataArrays.
- **Training task** — `SemanticSegmentationTask` provides model construction,
  training, evaluation, and prediction. Dataset adapter design is pending.
- **Pretrained model registry** — encoders (DINOv3, Prithvi, Prithvi-TL,
  Clay), decoders (DPT, UNet), heads, selected by registry key, chained
  together automatically, no manual import wiring or hand-glued forward pass.
- **Sensor-aware band metadata** — wavelength/GSD/mean/std per sensor
  (Sentinel-2, Landsat, MODIS, more), feeding model config directly (Clay's
  wavelength conditioning, normalization stats) — a geodata concern, not
  hardcoded into any model.
- **MLflow run tracking** — set `MLFLOW_TRACKING_URI` and training logs to
  MLflow alongside the local logger, no config change. Rebuilding a trained
  model from its checkpoint and registering it is still pending.
- **Editable scaffolding, not a framework lock-in** — `geosave create`
  hands you real, editable files. No required base class your code has to
  obey to keep working.

## Installations

Requires Python 3.12+.

```bash
pip install geosave-engine
# or
uv add geosave-engine
```

Want the rolling dev build (rebuilt on every push to `main`) instead of the
latest stable tag:

```bash
pip install --pre --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ geosave-engine
```

Working on GeoSave Engine itself (clone + `uv sync`), or installing an
exact dev build off a GitHub release — see the
[documentation](https://weedkat.github.io/geosave-engine/).

## Quick Start

```bash
uv run geosave create my-project
cd my-project
# fill in .env with your CDSE (or other STAC provider) credentials
```

`geosave create` prompts for a task and method; pass `-t`/`-m` to skip the
prompts and `-d` to set the project description. Then follow the
[documentation](https://weedkat.github.io/geosave-engine/) for the full
step-by-step — explore a pipeline, build a dataset, train.

## Generated Workspace

```text
my-project/
├── artifacts/     # checkpoints, logs, saved configs (created by training)
├── configs/       # LightningCLI YAML configs
├── data/          # project datasets
├── logs/
├── modules/       # editable project modules
├── notebooks/
├── predictions/
├── scripts/       # `geosave make scripts <file>` copies boilerplate here
├── .env           # CDSE credentials, filled in with placeholders
├── geosave.toml   # project name, description, task/method, engine version
└── main.py        # LightningCLI entry point — do not need to touch this
```

## Development Workflow

TO BE ADDED LATER

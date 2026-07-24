# geosave-engine

GeoSave Engine is a local-first product for building geospatial AI workflows end to end. It standardizes the full path from data acquisition, environment setup, model training, and prediction to serving-ready outputs, so teams do not need to reinvent a different workflow for every project.

It generates a ready-to-use boilerplate and applies proven best practices out of the box, including access to state-of-the-art models and multiple training methods with minimal coding. Instead of building model pipelines from scratch, users can focus on dataset creation and preprocessing, then run the resulting pipeline on fresh satellite data directly from their own machine.

## Features

- **Geospatial data pipeline** — `GeoAnchor`/`GeoTile`/`GeoStack` model
  location+time, fetched pixels, and multi-layer samples. Pull from a live
  STAC catalog (Copernicus, Planetary Computer, Element84, or any
  self-hosted endpoint) or local GeoTIFF, derive layers (cloud masks, NDVI,
  labels), save to disk as `.geostack` folders or stream straight into
  prediction with no disk round trip.
- **Training, config-only** — `SemanticSegmentationTask` +
  `SemanticSegmentationDataModule` cover plain supervised segmentation
  entirely from a LightningCLI YAML config, no Python to write. A pipeline's
  own per-sample context (e.g. a Prithvi/Clay encoder's real acquisition
  time/location) wires straight in via one config field. Drop to a
  hand-written `LightningModule` when you need full control.
- **Pretrained model registry** — encoders (DINOv3, Prithvi, Prithvi-TL,
  Clay), decoders (DPT, UNet), heads, selected by registry key, chained
  together automatically, no manual import wiring or hand-glued forward pass.
- **Sensor-aware band metadata** — wavelength/GSD/mean/std per sensor
  (Sentinel-2, Landsat, MODIS, more), feeding model config directly (Clay's
  wavelength conditioning, normalization stats) — a geodata concern, not
  hardcoded into any model.
- **MLflow model registry integration** — `geosave upload` rebuilds a
  trained model from its checkpoint and registers it, ready to serve.
- **Editable scaffolding, not a framework lock-in** — `geosave create`
  hands you real, editable files. No required base class your code has to
  obey to keep working.

## Installations

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:weedkat/geosave-engine.git
cd geosave-engine
uv sync
uv run geosave --help
```

## Quick Start

```bash
uv run geosave create -d my-project
cd my-project
# fill in .env with your CDSE (or other STAC provider) credentials
```

Then follow [docs/guide/workflow.md](docs/guide/workflow.md) for the full
step-by-step — explore a pipeline, build a dataset, train, register.

## Generated Workspace

```text
my-project/
├── artifacts/     # checkpoints, logs, saved configs (created by training)
├── configs/       # LightningCLI YAML configs
├── data/          # ingested layers land here
├── logs/
├── modules/       # your pipeline (Path A); data module + lightning module too, if Path B
├── predictions/
├── .env           # CDSE credentials, filled in with placeholders
├── geosave.toml   # workspace identity (task/method/catalog), read by the CLI
└── main.py        # LightningCLI entry point — do not need to touch this
```

## Development Workflow

TO BE ADDED LATER

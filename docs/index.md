# GeoSave Engine

## What It Is

GeoSave Engine is a local-first product for building geospatial AI workflows end to end. It standardizes the full path from data acquisition, environment setup, model training, and prediction to serving-ready outputs, so teams do not need to reinvent a different workflow for every project.

It generates a ready-to-use boilerplate and applies proven best practices out of the box, including access to state-of-the-art models and multiple training methods with minimal coding. Instead of building model pipelines from scratch, users can focus on dataset creation and preprocessing, then run the resulting pipeline on fresh satellite data directly from their own machine.

## Who It Is For

GeoSave Engine is built for geospatial practitioners who want to implement AI in real workflows, and for users entering geospatial processing who need a structured path to get started.

It is especially useful for teams and individuals who want to move fast from raw geospatial data to deployable predictions without spending most of their time designing project architecture, wiring training stacks, or maintaining custom pipeline glue code.

## Main Workflow

This is a usage overview. For setup, follow the installation guide first.

### Build a New Workspace

Start by generating a new project scaffold.

```bash
geosave build my-project
```

Outcome: A ready workspace is created with project structure, config entry points, and runnable scripts.

### Create and Run Dataset Scripts

Dataset ingestion and preprocessing are project-specific. Edit scripts based on your data design, then use the CLI UI to discover and execute scripts.

```bash
geosave run my-project
```

Outcome: Your dataset creation and preprocessing flow is integrated into the project pipeline.

### Train the Model

Start training from your workspace. If no config is passed, GeoSave prompts you to select a config file interactively.

```bash
geosave fit my-project
```

Outcome: Training runs with your selected config and produces model artifacts for later evaluation and prediction.

### Benchmark with Test

Run evaluation to benchmark model performance. If no artifact path is passed, GeoSave prompts you to select model artifacts interactively.

```bash
geosave test my-project
```

Outcome: You get evaluation metrics to validate quality and decide whether to iterate.

### Inspect Logs and Iterate Configs

Review training logs and compare runs, then create a new config if you want to train a different model variant.

```bash
tensorboard --logdir my-project/artifacts
geosave fit my-project -c configs/another-model.yaml
```

Outcome: You can compare experiments and retrain with new model choices without rebuilding the workflow.

### Predict on Fresh Data

Prepare fresh-data scripts, run them through CLI UI, then run prediction. If no artifacts path is passed, GeoSave prompts artifact selection interactively.

```bash
geosave run my-project
geosave predict my-project
```

Outcome: You produce predictions on fresh satellite data using trained model artifacts.

## CLI UI

Placeholder: add a screenshot of the GeoSave CLI interactive UI here.

![GeoSave CLI UI Placeholder](images/cli-ui-placeholder.png)

## Tech Stack

GeoSave Engine combines PyTorch Lightning and TorchGeo for model training and geospatial sampling, STAC tooling (pystac, pystac-client, odc-stac, odc-geo) for data ingestion, and geospatial IO libraries (geopandas, rasterio, rioxarray) for dataset preparation and processing. The workflow is exposed through a script-first CLI built with Typer.

## Workspace Boilerplate

A generated workspace gives you a ready structure to start quickly:

my-project/
	geosave.toml
	main.py
	README.md
	configs/
	data/
	scripts/
	src/
	artifacts/

In short: config, data staging, runnable scripts, source code, and output artifacts are prepared in one place.

## Next Steps

Start by building a workspace, then adapt ingestion for your dataset. A Dynamic World example can be added later as a concrete dataset structure reference.

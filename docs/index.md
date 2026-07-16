# GeoSave Engine

## What It Is

GeoSave Engine is a local-first product for building geospatial AI workflows end to end. It standardizes the full path from data acquisition, environment setup, model training, and prediction to serving-ready outputs, so teams do not need to reinvent a different workflow for every project.

It generates a ready-to-use boilerplate and applies proven best practices out of the box, including access to state-of-the-art models and multiple training methods with minimal coding. Instead of building model pipelines from scratch, users can focus on dataset creation and preprocessing, then run the resulting pipeline on fresh satellite data directly from their own machine.

## Philosophy

The library does two things, deliberately, and nothing more: gives you a
pipeline abstraction for turning raw geospatial data into a trainable
dataset, and wires that up to an established training framework (PyTorch
Lightning) instead of making you glue one together yourself. It isn't
trying to be its own ML framework or its own geospatial format — it
composes existing best-practice tools (STAC, zarr, Lightning, MLflow) around
three consistent objects, and gets out of your way past that point:

- `GeoAnchor` — where + when, no pixel data yet. A location (`GeoBox`) and
  datetime; what ingest sources produce and what `GeoPipeline.ingest()`
  takes in.
- `GeoTile` — a `GeoAnchor` plus its fetched raster data.
- `GeoStack` — multiple named `GeoTile` layers for one anchor, saved
  together as one `<anchor>.geostack/` folder.

Templates are scaffolding, not a contract. `geosave create` hands you real,
editable files — no required base class your code has to obey to keep
working.

## Who It Is For

GeoSave Engine is built for geospatial practitioners who want to implement AI in real workflows, and for users entering geospatial processing who need a structured path to get started.

It is especially useful for teams and individuals who want to move fast from raw geospatial data to deployable predictions without spending most of their time designing project architecture, wiring training stacks, or maintaining custom pipeline glue code.

## CLI UI

Placeholder: add a screenshot of the GeoSave CLI interactive UI here.

![GeoSave CLI UI Placeholder](images/cli-ui-placeholder.png)

## Tech Stack

GeoSave Engine combines PyTorch Lightning for training, STAC tooling
(pystac, pystac-client, odc-stac, odc-geo) for data ingestion, and
geospatial IO libraries (zarr, rasterio, rioxarray, geopandas) for dataset
preparation and processing. The CLI is built with Typer + questionary.

## Next Steps

**[workflow.md](guide/workflow.md)** is the full step-by-step: `geosave
create` through exploring data, building a dataset, training, and
registering a model — workspace layout and every command, in one ordered
read. Deep reference material lives in **[docs/concept/](concept/)**:
[geotile.md](concept/geotile.md) (`GeoAnchor`/`GeoTile`/`GeoStack`),
[pipeline.md](concept/pipeline.md) (`GeoPipeline`, sources, STAC),
[model.md](concept/model.md) (`GeoDataset`,
`SemanticSegmentationTask`/`DataModule`, config.yaml).

The `workspace/` directory in this repo (Sentinel-2 + DynamicWorld land
cover segmentation, Path A) is a concrete, real example to read alongside
it — not a placeholder to wait for.

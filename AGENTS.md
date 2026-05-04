# Geosave Engine Agent Guidelines

## Project Goal

Generate ready-to-use geospatial ML workspaces by reliably copying curated templates through the GeoSave CLI.

## Agent Defaults

- Keep replies short.
- Edit code first, then explain briefly.
- Use real APIs, files, deps, and params from repo or docs.
- Match existing project structure and style.
- Ask questions when blocked or risk is high.
- Keep durable rules here; task flow goes elsewhere.

## Coding Rules

- Prefer typed params and returns; avoid Any unless justified.
- Add docstrings for public classes and functions.
- Use named constants for repeated literals.
- Comment only for non-obvious intent or edge cases.
- Validate inputs early and raise clear errors.
- Keep functions and classes focused on one job.
- Keep diffs scoped to the task.
- Preserve public APIs unless a change is requested.
- Use clear, descriptive names.
- Choose the simplest correct solution and reuse repo patterns.

## Exception Policy

- If you break a rule, leave a brief reason.
- If an exception repeats, update the guideline.

## Repository Map

```text
src
├── geosave_engine
│   ├── cli
│   ├── geodata
│   ├── ml
│   └── utils
└── templates
    ├── plugins
    │   ├── notebook
    │   └── scripts
    └── workspace
        ├── common
        ├── object_detection
        ├── pixelwise_regression
        └── semantic_segmentation
```

## Required Stack

- Machine Learning: PyTorch Lightning.
- Dataset Management: TorchGeo-compatible geospatial datasets and aligned sampling.
- Data Ingestion: odc-stac with xarray-based preprocessing.
- Raster IO: rioxarray and rasterio for GeoTIFF read/write and CRS-aware raster operations.
- Dataset Manifests: GeoPackage (GPKG) for dataset manifest handling.

## Testing

- Use `workspace/` folder (generated via `geosave build`) for CLI integration tests.
- Pipeline integration tests hit real CDSE STAC + S3. Mark with `@pytest.mark.integration`. Run via `pytest -m integration`. Credentials loaded from `tests/.env`.
- Unit tests for pipeline components (Anchor, Derived, Pipeline error cases) do not need network and run unmarked.
- Test data lives in `tests/data/`
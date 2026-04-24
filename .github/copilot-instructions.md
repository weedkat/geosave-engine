# Geosave Engine Agent Guidelines

## Project Goal

Generate ready-to-use geospatial ML workspaces by reliably copying curated templates through the GeoSave CLI.

## Agent Defaults

- Keep outputs concise, actionable, and implementation-first.
- Prefer direct code changes over long explanations.
- Do not hallucinate APIs, dependencies, file paths, or parameters.
- Follow existing project structure, naming, and style.
- Ask for clarification only when ambiguity blocks implementation.
- Keep this file for durable defaults; put task-specific workflow in repo instructions.

## Coding Rules

- Use strict typing for all function inputs and returns; avoid Any.
- Add docstrings for public functions and non-trivial internal contracts.
- Use constants instead of magic numbers or strings.
- Add comments only for non-obvious intent.
- Fail fast with clear errors; avoid silent fallback behavior.
- Prioritize modularization; each function and class should have one responsibility.
- Prefer minimal diffs and avoid unrelated refactors.
- Preserve public APIs unless the user explicitly requests a breaking change.
- Use clear names; avoid abbreviations unless they are standard in the domain.
- Keep logic simple: KISS, YAGNI, and DRY.
- Use established libraries/patterns already present in this repository.

## Exception Policy

- If you must break a rule, add a short comment explaining why.
- If the same exception appears repeatedly, propose updating the guideline.

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

use workspace/ folder, a generated workspace from geosave build cli to test geosave functionalities (example: geosave run workspace)

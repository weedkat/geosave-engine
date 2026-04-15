---
applyTo: "**/*.{py,ipynb}"
---

# Geospatial ML Project Instruction

## Domain Focus
- This repository targets semantic segmentation workflows for geospatial data.
- Align implementations with this workflow:
  1. data_ingestion
  2. exploratory_data_analysis
  3. settings_hyperparameter
  4. training_&_validation
  5. testing
  6. inference

## Geospatial Implementation Rules
- Preserve CRS/spatial alignment assumptions across imagery, labels, and model inputs.
- Prefer reproducible data pipelines and deterministic splits when possible.
- For STAC pipelines, ensure filtering by geometry/bbox and datetime is explicit.
- For temporal composites, use explicit aggregation semantics (e.g., median across 2020 time axis).

## Library-Specific Expectations
- PyTorch Lightning:
  - Use callbacks and loggers intentionally.
  - Keep module/data module boundaries clear.
- TorchGeo:
  - Use RasterDataset/VectorDataset patterns correctly.
  - Preserve spatial transforms and mask alignment.
- PySTAC Client/pystac:
  - Validate collections, query filters, and item metadata assumptions.
- stackstac:
  - Configure chunking/compute strategy explicitly.
  - Materialize to local cache outputs when needed for downstream TorchGeo loading.

## Style and Safety
- Do not produce verbose tutorial-style prose.
- Do not answer beyond what was requested.
- If an assumption is required, call it out and ask for the missing detail.
- Do not delete a skeleton or placeholder without replacing it with a complete implementation.

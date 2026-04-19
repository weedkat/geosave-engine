---
applyTo: "**/*.{py,ipynb}"
---

# Geospatial ML Project Instruction

## Domain Focus
- This repository targets semantic segmentation workflows for geospatial data.
- The primary use case is building and training models on satellite imagery with spatially aligned labels.
- The project emphasizes geospatial data handling, model training orchestration, and STAC-based data pipelines.

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

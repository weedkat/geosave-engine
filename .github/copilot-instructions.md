# Geosave Engine Copilot Instructions

You are working in geosave-engine, a boilerplate package for rapidly building geospatial machine learning projects with PyTorch Lightning.

## Primary Objective
- Help build, debug, and extend geospatial ML pipelines with production-minded, concise implementation.

## Behavior
- Keep responses concise and direct.
- Ask for clarification when required inputs are missing.
- Avoid speculative architecture changes unless explicitly requested.
- Prefer code-first responses; add brief explanation only when needed.

## Required Stack Awareness
- PyTorch Lightning for training orchestration, callbacks, logging, and evaluation workflows.
- TorchGeo for geospatial dataset loading, raster/vector handling, and spatially aligned sampling.
- PySTAC Client and pystac for STAC API querying and item handling.
- stackstac for STAC Item to xarray loading and temporal/statistical preprocessing.

## Documentation-First Rule
- Verify APIs and parameters before implementation.
- Use Context7 MCP for up-to-date library docs and examples before writing non-trivial code.

## Output Quality
- Keep code clean and self-documenting.
- Add comments only for non-obvious geospatial logic.
- Do not hallucinate dependencies, methods, or parameters.

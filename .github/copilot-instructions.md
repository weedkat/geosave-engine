# Geosave Engine Project Guidelines

## Code Style
- Keep responses and implementations concise and direct.
- Prefer code-first changes; add brief explanation only when needed.
- Ask for clarification when required inputs are missing.
- Avoid speculative architecture changes unless explicitly requested.
- Follow existing repository patterns and naming/style conventions.
- Apply KISS, YAGNI, and DRY.
- Add comments only for non-obvious geospatial logic.
- Do not hallucinate dependencies, methods, or parameters.

## Architecture
- Core package code lives in `src/geosave_engine`; generated project blueprints live in `src/templates`.
- CLI entrypoint is `geosave` (`src/geosave_engine/cli/main.py`) and orchestrates `build`, `fit`, `test`, `predict`, `run`, and `docs` workflows.
- Keep PyTorch Lightning boundaries clear:
  - `LightningDataModule` handles data loading/sampling/transforms.
  - `LightningModule` handles model, loss, optimization, and step logic.
- Prefer class-path driven yaml configuration for models/optimizers/schedulers/losses/callbacks and instantiate through existing resolver patterns.

## Required Stack
- PyTorch Lightning for training orchestration, callbacks, and evaluation workflows.
- TorchGeo for geospatial dataset loading and spatially aligned sampling.
- PySTAC Client and `pystac` for STAC API querying and item handling.
- `odc-stac` for STAC Item loading into xarray and temporal/statistical preprocessing.

## Build and Test
- Environment setup: `uv sync --locked --no-editable`
- CLI sanity check: `uv run geosave --help`
- Run workflows from repo root or project workspace:
  - `uv run geosave build`
  - `uv run geosave fit`
  - `uv run geosave test`
  - `uv run geosave predict`
  - `uv run geosave run`
- Quality checks:
  - `uv run ruff check .`
  - `uv run pytest`
  - `uv run mypy src/geosave_engine tests`

## Geospatial Conventions
- Preserve CRS and spatial alignment assumptions across imagery, labels, and model inputs.
- Prefer deterministic/reproducible split behavior when possible.
- For STAC pipelines, make geometry/bbox and datetime filters explicit.
- For temporal composites, use explicit aggregation semantics.

## Documentation-First
- Verify APIs/parameters before implementing non-trivial logic.
- Use Context7 MCP for up-to-date library documentation and code examples.
- Link to existing project docs instead of duplicating detail:
  - `docs/GEOSPATIAL.md`
  - `docs/TORCHGEO.md`

# GeoSave Engine

GeoSave makes geospatial ML development feel like normal Python: ingest geospatial data into standard ecosystem objects, preserve their metadata and lazy behavior, and move from data preparation to Lightning-based model development with minimal glue code.

The project is in active Alpha development. Prefer simple, replaceable designs. Change an obsolete design instead of building compatibility layers around it unless compatibility is explicitly required.

Lead with code. Explain after showing the implementation. When comparing design options, show each option as the code a caller and a maintainer would see, then explain the difference. Always do smoke test before the design is settled. Ignore ipynb file unless asked to. Choose variable and parameter name carefully, don't make naming too verbose.

## Commands

```bash
uv run pytest
uv run pytest tests/path/to/test_file.py
uv run pytest -m integration
uv run pytest -m slow

uv run ruff check .
uv run pre-commit run --all-files
```

`pyproject.toml` is the source of truth for dependencies and tool configuration.

## Structure

```text
src/geosave_engine/
├── cli/          # CLI and AI workspace generation
├── geodata/      # raster/vector data, I/O, STAC, transforms, metadata
├── model/        # what a release contains: chain, registry, encoder/decoder/head, spec, release
├── ml/           # Lightning training: modules, datamodules, builders, callbacks, datasets
├── templates/    # source for generated workspaces
└── workflow/     # Prefect workflows

tests/            # mirrors src/geosave_engine
workspace/        # generated consumer workspace, not library source
```

`src/geosave_engine/templates/` is the source for generated workspace files.
It contains `common/` startup, `workspaces/` starters, and optional `scaffolds/`.
Public geospatial I/O lives in `geodata/io/`; original benchmark downloads live
in `geodata/benchmarks/`. Internal helpers stay with their owning package.

Dependencies point one way: `geodata <- model <- ml`. `geodata` imports no torch, and `model` does not import `ml`.

## Design

GeoSave composes the existing geospatial and ML ecosystem instead of replacing it with parallel abstractions.

```text
Raster representation   -> xarray Dataset / DataArray
Vector representation   -> GeoDataFrame
Training                -> PyTorch Lightning
Workflow orchestration  -> Prefect
Model publishing        -> Hugging Face Hub
Remote training data    -> LitData
Visualization           -> HoloViz ecosystem
```

Prefer native ecosystem objects in public APIs. Introduce a GeoSave-specific wrapper only when it owns an invariant that the underlying object cannot represent cleanly.

Use capabilities provided by dependencies before implementing equivalent behavior.

## Geospatial Contracts

Raster operations should remain lazy when possible.

Do not eagerly materialize an entire raster unless the operation requires it.

Preserve metadata required for CF/GDAL-compatible round trips, including applicable CRS, coordinates, spatial dimensions, transform, nodata, dtype, and band or variable identity.

Reprojection, resampling, dtype conversion, coordinate changes, and eager computation are explicit operations.

Built-in persistence:

```text
Raster -> Zarr / NetCDF / GeoTIFF
Vector -> Parquet / GeoParquet
```

## Machine Learning

Use Lightning as the training framework rather than building a parallel training loop.

```text
LightningModule -> training behavior
Trainer         -> orchestration
Callback        -> reusable lifecycle behavior
LightningCLI    -> workspace entry point and configuration
Logger          -> experiment logging
```

A training setup is one LightningModule and the DataModule that feeds it, kept together in `ml/<head type>/<method>/`:

```text
ml/segmentation/
├── metrics.py        # shared by every segmentation method
└── supervised/
    ├── module.py     # Module(LightningModule)
    └── data.py       # DataModule(LightningDataModule)
```

Share code across head types only once two written modules repeat it.

Generated workspaces compose GeoSave and Lightning APIs; they do not own library implementations.

## Tests

Tests mirror the source tree:

```text
src/geosave_engine/geodata/core/raster.py
tests/geodata/core/test_raster.py
```

Persistence changes require round-trip tests.

Operations that promise lazy execution must test that laziness is preserved.

## Public API and Docstrings

Public APIs use concise Google-style docstrings.

Show the public object users work with rather than inspecting its internal attributes.

```python
def rename_vars(self, mapping: dict[str, str]) -> xr.Dataset:
    """Rename data variables without changing their order or pixels.

    Args:
        mapping: Existing variable names mapped to replacement names.

    Returns:
        Dataset with renamed data variables.

    Raises:
        KeyError: If a source variable is absent.
        ValueError: If a replacement is empty or creates a duplicate.

    Examples:
        >>> renamed = ds.gs.rename_vars({"B04": "red", "B08": "nir"})
        >>> renamed
        <xarray.Dataset> ...
    """
```

Keep obvious accessors short:

```python
@property
def variable_count(self) -> int:
    """Return the number of data variables."""
```

Comments explain non-obvious domain constraints, not syntax or implementation history.

## Changes

* Inspect the relevant implementation and tests first.
* Replace a wrong abstraction instead of adding adapters around it.
* Do not add compatibility aliases or duplicate execution paths unless required.
* Keep unrelated code unchanged.
* Use existing dependencies before adding new ones.
* Do not discard unrelated working-tree changes.

## Release

Packaging uses Hatchling.

PyPI releases are handled by the existing GitHub Actions workflow.

Documentation uses Zensical.

## Handoff

Report only:

1. What changed.
2. Checks run.
3. Breaking changes or remaining risks.

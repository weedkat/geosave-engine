# Project structure

> **Status:** Directory ownership is stable; individual modules are still moving during redesign.

## Source ownership

- `src/geosave_engine/cli`: CLI commands and workspace scaffolding.
- `src/geosave_engine/templates`: files emitted into generated workspaces.
- `src/geosave_engine/geodata`: Spatial types, geospatial I/O, STAC, pipelines, datasets, and datastores.
- `src/geosave_engine/ml`: model contracts, architectures, tasks, transforms, and callbacks.
- `tests`: unit and integration tests.
- `docs`: user and developer documentation.

`workspace/` is a generated consumer and integration example. Library fixes belong in `src/geosave_engine` or its templates.

A detailed module tree will be regenerated after the redesign to avoid documenting transient paths.

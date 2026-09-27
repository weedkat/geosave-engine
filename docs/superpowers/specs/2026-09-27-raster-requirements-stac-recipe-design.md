# Raster Requirements and STAC Recipes

Status: approved direction; pending written review.

## Goal

Give each model a complete, reproducible raster contract without conflating a
model raster, a STAC loader, and per-run configuration under the word
`source`.

The model specification declares:

- which named rasters the model accepts;
- the variables, coordinates, structure, grid, and metadata each raster must
  satisfy; and
- an optional STAC recipe describing the preferred way to acquire that raster.

A user may instead supply an existing native `xarray.Dataset`. Existing
rasters bypass STAC acquisition but must satisfy the same raster requirement
before preprocessing.

## Vocabulary

The public vocabulary is:

| Name | Responsibility |
| --- | --- |
| `ModelSpec.rasters` | Named raster requirements owned by the model. |
| `RasterRequirement` | Select and validate one native raster. |
| `RasterRequirement.stac` | Optional preferred STAC acquisition recipe. |
| `StacRecipe` | Collection, endpoints, query, and loading behavior. |
| `StacSource` | Live collection-bound loader in the geodata layer. |
| `load_stac_raster` | Execute one recipe on a target and validate the result. |

The obsolete workflow vocabulary is removed rather than aliased:

```text
ModelSpec.sources     -> ModelSpec.rasters
SourceConfig          -> removed
load_raster           -> load_stac_raster
sources=              -> removed from runnable flows
--sources             -> removed from workflow commands
```

`source` remains appropriate only for concrete external providers such as
`StacSource`. It no longer describes a model requirement or an independent
workflow configuration mapping.

## Model specification

One raster declaration combines its output contract with an optional
acquisition recipe:

```yaml
rasters:
  sentinel_2_l2a:
    variables: [B04, B08]
    coordinates: [time, y, x]
    dims: [time, y, x]
    dtypes: [uint16]
    require_crs: true
    resolution: 10

    attrs:
      data_vars:
        "*":
          models:
            packing:
              required: [scale_factor]

    stac:
      collection: sentinel-2-l2a
      endpoints:
        - https://planetarycomputer.microsoft.com/api/stac/v1
      query:
        filter:
          op: <=
          args:
            - {property: eo:cloud_cover}
            - 20
        max_items: 8
      load:
        groupby: solar_day
        resampling: bilinear
```

`RasterRequirement` retains its existing variable/channel selection, grid,
dtype, and attrs behavior. It gains:

```python
coordinates: tuple[Text, ...] = ()
stac: StacRecipe | None = None
```

The existing top-level `collection` and `endpoints` fields move into
`StacRecipe`. Query and load declarations that were supplied through
`SourceConfig` move into the same recipe. No second mapping needs to be joined
to `ModelSpec.rasters` at execution time.

## Coordinate requirements

`dims` constrains the dimension order of each selected data variable, but an
xarray dimension can exist without a coordinate array. `coordinates` therefore
requires named coordinate arrays explicitly.

Validation is intentionally small:

- coordinate names must be unique in the model specification;
- each declared coordinate must exist after variable/channel selection;
- extra coordinates are allowed;
- coordinate arrays are not reordered, converted, or computed; and
- coordinate metadata remains under `attrs.coords`.

A named `attrs.coords` declaration continues to require that coordinate as it
does today. `coordinates` provides a direct presence requirement when no
coordinate metadata predicate is needed. There is no separate
`CoordinateRequirement`, exact-values grammar, monotonicity rule, or coordinate
dtype grammar in this change.

## Attrs requirements

`AttrsRequirement` remains unchanged and optional. It describes metadata the
model genuinely depends on at root, data-variable, and coordinate scope. It is
not populated automatically from the STAC recipe.

Both acquisition paths use the same existing validation:

```python
validated = requirement.select_raster(raster)
```

This selects required variables or channels and validates coordinates,
dimensions, stored dtype, grid, resolution, and attrs without computing lazy
pixels.

## STAC recipe

`StacRecipe` is a model-owned, YAML-safe declaration:

```python
class StacRecipe(SpecModel):
    collection: Text
    endpoints: tuple[HttpUrl, ...]
    query: QueryConfig = Field(default_factory=QueryConfig)
    load: StacSourceConfig = Field(default_factory=StacSourceConfig)
```

The existing `QueryConfig` and `SortConfig` become model-spec types rather than
runtime source configuration. `StacRecipe` applies the existing primitive-load
validation so callables, non-finite values, and other non-YAML runtime objects
cannot enter `model_spec.yaml`.

Collection names are non-empty. Endpoint order is preserved for fallback,
endpoints must be unique HTTP(S) URLs, and at least one endpoint is required.

When named variables are declared, they are authoritative for loaded bands. A
recipe must omit `load.bands` or use the same ordered names. For positional
`channels`, an explicit recipe band selection remains allowed and the resulting
raster must contain at least the required channel count.

Credentials and provider signing remain environment/runtime concerns. The
model recipe stores no secrets or callables.

## Recipe-first target behavior

The parsed `GeoAnchor` is called the `target`. It defines the output grid and
provides fallback spatial and temporal search bounds. It is not an acquisition
policy and does not override an explicit recipe query.

The effective query follows these rules:

1. Collection, filters, ordering, limits, and explicit selectors come from the
   recipe.
2. Explicit item IDs run without target-derived spatial or temporal bounds.
3. Otherwise, an explicit recipe `bbox` or `intersects` takes priority.
4. If neither spatial selector is present, the target footprint supplies the
   bounding box.
5. An explicit recipe `datetime` takes priority; otherwise the target timespan
   is used.
6. Regardless of query selection, matched pixels are loaded onto the target's
   exact geobox.

Supplying both `bbox` and `intersects` remains invalid. This removes the current
possibility of always injecting an anchor bounding box beside a recipe geometry.

Label-derived targets normally supply both extent and time because dense
recipes omit those query fields. Fixed-vintage or explicitly pinned recipes may
override either.

## STAC acquisition

The workflow task becomes explicitly STAC-specific:

```python
load_stac_raster(
    target: GeoAnchor,
    requirement: RasterRequirement,
) -> xr.Dataset
```

It performs the following operations:

1. Revalidate the raster requirement.
2. Require its optional `stac` recipe.
3. Open the first configured endpoint publishing the collection.
4. Build the recipe-first effective query for the target.
5. Configure a fresh mutable `StacSource` with the recipe load declaration.
6. Load lazily onto the target geobox.
7. Return `requirement.select_raster(raster)`.

The existing cached `StacClient` and fresh-per-load `StacSource` behavior is
preserved.

## Existing-raster path

An existing native raster skips acquisition only:

```python
model = ModelSpec.load("model_spec.yaml")

with io.read_raster("data/optical.tif") as raster:
    optical = model.rasters["sentinel_2_l2a"].select_raster(raster)
    prepared = preprocess({"sentinel_2_l2a": optical}, model)
```

`preprocess` continues to select and validate every consumed model raster before
invoking the first preprocessing declaration. This makes the shorter direct
form safe as well:

```python
with io.read_raster("data/optical.tif") as raster:
    prepared = preprocess({"sentinel_2_l2a": raster}, model)
```

The STAC recipe is ignored when validating an existing raster. No fake request,
provider wrapper, or workflow configuration is required.

## Runnable flows and CLI

The runnable flow signatures become:

```python
ingest(
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str,
) -> str

prepare_dense_data(
    labels: str,
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
    max_concurrency: PositiveInt = 1,
) -> str
```

Each flow loads `ModelSpec.rasters`, validates that every required raster has a
STAC recipe before submitting pixel work, and calls `load_stac_raster`.

The CLI removes `--sources`. The minimal commands are:

```bash
geosave workflow ingest \
  --anchor '{"kind":"raster","path":"data/reference.tif"}' \
  --output data/raw.zarr \
  --spec model_spec.yaml

geosave workflow prepare-dense-data \
  --labels data/labels \
  --output data/prepared \
  --spec model_spec.yaml
```

The safe dense default remains `--max-concurrency 1`. It still limits complete
sample ingestions from STAC search through atomic Zarr publication. There is no
named Prefect global concurrency limit.

## Failure and persistence behavior

Before pixel work, flows validate the model spec, raster requirements, STAC
recipes, anchor/label discovery, and concurrency. Existing atomic local Zarr
publication, completed-sample reuse, fail-fast bounded submission, and
manifest-last publication remain unchanged.

A STAC workflow fails clearly when any required raster lacks a recipe. A user
with an existing raster uses the native validation/preprocessing path instead
of the STAC workflow.

## Package migration

The source layout follows the new ownership:

```text
workflow/
├── specs/
│   ├── rasters.py        # raster and attrs requirements
│   └── stac.py           # recipe, query, and sort declarations
├── configs/
│   └── anchor.py         # per-run target construction only
├── tasks/
│   ├── load.py           # load_stac_raster
│   ├── dense.py
│   └── process.py
└── flows/
    ├── ingest.py
    └── prepare_dense_data.py
```

The obsolete `workflow/configs/source.py` and `workflow/specs/sources.py`
modules are removed after their declarations move to the new owners. Tests move
with their source modules. No import aliases, old YAML field acceptance, or
duplicate execution paths are retained.

Generated model specs, scripts, CLI tests, workflow tests, and the workflow
guide migrate in the same change.

## Tests

Behavioral coverage must prove:

- `ModelSpec` accepts `rasters` and rejects the removed `sources` field;
- coordinate requirements reject missing coordinates without computing pixels;
- extra coordinates remain valid and coordinate attrs still work;
- STAC recipe validation covers collection, endpoints, primitive load values,
  conflicting band declarations, and query shape;
- recipe bbox/intersects/datetime take priority over target fallbacks;
- target extent/time fill only omitted recipe selectors;
- item IDs do not gain target-derived bounds;
- `load_stac_raster` loads lazily and validates the result;
- missing recipes fail before STAC or raster work;
- existing rasters validate and preprocess without a STAC recipe;
- both runnable flows need no parallel source/config mapping;
- CLI help and forwarding no longer expose `--sources`;
- dense concurrency, failure, resume, and manifest behavior remain intact; and
- shipped model specs and generated workspace examples use the new schema.

Focused tests run first, followed by the affected workflow/CLI suite, slow
Prefect/STAC coverage, Ruff, BasedPyright, Zensical, and `git diff --check`.

## Breaking changes

- `model_spec.yaml` uses `rasters` instead of `sources`.
- `ModelSpec.sources` becomes `ModelSpec.rasters`.
- `collection` and `endpoints` move under each requirement's `stac` recipe.
- `SourceConfig` and its source-config module are removed.
- `load_raster` becomes `load_stac_raster`.
- Flow `sources` arguments and CLI `--sources` options are removed.

The project is in alpha, so these obsolete names are replaced directly without
compatibility aliases.

## Deliberately deferred

- A CLI command for validating already available raster files.
- A generic provider or acquisition registry.
- Non-STAC acquisition recipes.
- Exact coordinate values, monotonicity, or coordinate dtype declarations.
- Runtime query overrides that silently change the model recipe.
- Remote Zarr destinations and Prefect deployment submission.

# Model specifications

A model spec accompanies a model and describes its required data and processing.
It should remain useful to someone using the model without GeoSave's Prefect flows.
Job-specific locations, dates, paths, query/load options and server settings stay
in the caller's Python parameters and runtime configuration.

Use
`Processor.load(path, stage="preprocessing")` or `stage="postprocessing"` to run
one stage. See the [runnable example](../examples/README.md) first.

```yaml
schema_version: 2
sources:
  optical:
    variables: [red, nir]
preprocessing:
  selected:
    call: !ref optical.__getitem__
    kwargs:
      key: [red, nir]
```

Input: an `optical` Dataset. Output: `selected`, containing its red and nir bands.

## Fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Required; currently `2` |
| `sources` | Required raster requirements; use `{}` when none are needed |
| `preprocessing` | Ordered preparation steps; defaults to `{}` |
| `postprocessing` | Ordered interpretation steps; defaults to `{}` |
| `inference` | Saved declarations only; execution and model loading are deferred |

Each step names its complete return value. `call` is a module-level import path
(such as `xarray.merge`) or a reference to a supplied callable or bound method.
`kwargs` contains the actual keyword arguments and defaults to `{}`. A bound method
already has its receiver; calls never add arguments or convert outputs implicitly.

`sources` names logical data requirements and optional STAC acquisition identity.
Acquisition can use these requirements to select bands; processors can use them
to validate supplied data. Processing `kwargs` describe the model's transformations,
such as band order or normalization. They are not a second set of flow parameters.

When a model requires merging, its reconstruction calls belong in its own
`postprocessing` section. The caller supplies request-owned objects and controls
batching and accumulator lifetime. There is no separate merging configuration file.

## References and results

- `!ref optical` uses the supplied object; `!ref optical.gs.unpack` uses its bound method.
- References may appear inside lists and mappings. Plain strings and `{ref: optical}`
  remain literal values.
- Dotted paths follow Python attributes, without expressions or implicit dictionary lookup.
- Steps run in order. References resolve before the result replaces its named binding.
- Returns may be rasters, tensors, tables, tuples, scalars or `None`.
- Each invocation returns a fresh dictionary. Referenced objects retain their identity;
  explicitly called mutating methods can change those objects.

A caller-supplied name can be replaced by a step result, including across stages.
Duplicate names within one YAML mapping are rejected. Use distinct intermediate
names within a stage. Missing external names and incompatible raster inputs fail
before calls run; attributes and call arguments are checked when available.
Positional-only calls are outside this keyword-argument grammar.

## Source requirements

Requirements select and validate lazy raster views without converting values or
computing pixels. Only sources needed by the active stage are checked.

| Setting | What it checks |
| --- | --- |
| `variables` | Required variable names, in a nonempty list |
| `channels` | First N positional channels; mutually exclusive with variables |
| `collection` | STAC collection ID; must be supplied with endpoints |
| `endpoints` | Nonempty unique HTTP(S) endpoints in acquisition fallback order |
| `dims` | Exact dimension order for each required variable |
| `dtypes` | Allowed stored data types |
| `require_crs` | Whether a spatial grid must be present |
| `resolution` | Pixel size, as a scalar or `[x, y]` |
| `attrs` | Required root, variable or coordinate metadata |

Exactly one of `variables` or `channels` is required. The source `type` defaults to
`raster`, the currently supported validator. Named variables preserve declared
order; channels selects the first N ordinary variables or the first N bands of a
single band variable. Multiple variables mixed with a band dimension are ambiguous.
Other Python objects can be supplied directly without raster requirements.
Unpacking, alignment and tensor conversion remain explicit operations.

Existing raster requirements omit both `collection` and `endpoints`. Acquisition
through `ingest` requires both, for example:

```yaml
sources:
  imagery:
    channels: 3
    collection: sentinel-2-l2a
    endpoints:
      - https://primary.test/stac
      - https://backup.test/stac
```

Endpoints serialize as strings and preserve their declared order. Runtime settings
for `imagery` contain only optional `query` and `load` mappings, never a URL or
collection override. Endpoint fallback checks catalogue and collection availability;
empty search results and later lazy reads do not retry another endpoint.

## Create and save a spec in Python

```python
from geosave_engine.workflow.spec import ModelSpec, OperationSpec, Ref

spec = ModelSpec(
    schema_version=2,
    sources={},
    preprocessing={
        "selected": OperationSpec(
            call=Ref("optical.__getitem__"), kwargs={"key": ["red", "nir"]}
        ),
    },
)
path = spec.save("artifacts/model")
restored = ModelSpec.load(path)
```

`save` accepts a YAML filename or a directory receiving `model_spec.yaml`. Loading
and saving preserve values, references and declaration order, but not comments or
formatting. `ModelSpec.model_validate(spec.model_dump())` also preserves the spec.
This does not translate arbitrary Python source code.

Use `ModelSpec.load` to read `!ref` tags. Parsing does not import or execute calls;
running a processor executes the declared Python code. Load specs from your model
artifact, rather than accepting executable configuration in request bodies.

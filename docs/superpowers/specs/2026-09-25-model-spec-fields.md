# Workflow model-spec fields

Status: implemented core. Inference execution and model loading are deferred.
Companion to [the core design](2026-09-25-workflow-core-design.md).

## Document fields

| Field | Required/default | Meaning |
| --- | --- | --- |
| `schema_version` | Required, `2` | The supported declaration grammar |
| `sources` | Required mapping, may be empty | External native raster requirements |
| `preprocessing` | `{}` | Ordered named call declarations |
| `inference` | `{}` | Preserved call declarations; no inference executor implied |
| `postprocessing` | `{}` | Ordered named call declarations |

Names are identifiers without dots. Reject unknown top-level fields and duplicate
keys. There is one workflow-owned schema; version 1 is rejected. Model identity,
weights and revision remain the artifact loader's responsibility.

## Call fields

| Field | Required/default | Meaning |
| --- | --- | --- |
| Enclosing key | Required identifier | Name bound to the complete return value |
| `call` | Required import path or `!ref` | Callable/class or supplied bound method |
| `kwargs` | `{}` | Actual keyword arguments |

An imported call names a module-level callable. Imports occur only when building
the selected processor. A dotted reference follows native attributes from its root
value. It is never an expression or an implicit call. Bound methods already have
`self`. Positional-only calls are outside the initial grammar.

Kwargs contain finite scalar literals, lists, string-keyed mappings and `Ref`
markers. Duplicate/non-string keys, malformed paths, unknown tags, unsupported
objects and cyclic configuration containers are rejected. A resolved object stays
opaque and keeps its native identity.

Assignment resolves the previous bindings first, calls the target, then replaces
the enclosing name. A mutating method keeps its actual return value, including
`None`. A failed call does not replace the name, but explicit mutations performed
by that call are not rolled back.

## Raster requirements

| Field | Required/default | Meaning |
| --- | --- | --- |
| `type` | `raster` | Only implemented source kind |
| `variables` | Required nonempty unique list | Variables that must be present |
| `dims` | Unconstrained | Exact dimensions/order for each required variable |
| `dtypes` | Unconstrained | Accepted stored dtypes |
| `require_crs` | `false` | Require a locatable CRS/grid |
| `resolution` | Unconstrained | Positive scalar or `[x, y]` pixel sizes in CRS units |
| `attrs` | `{}` | Existing root/data_vars/coords metadata requirements |

Requirements inspect structure and metadata without computing pixels, selecting
bands, casting or resampling. Extra variables are retained. Only external sources
needed by the active stage are validated. General operation results are not
restricted to raster types.

Metadata uses registered `models` and unregistered `foreign` fields with `required`,
`equals` and `one_of`, through the existing attrs field parsers/equality. The
variable wildcard applies to required variables. Requiring packing metadata does
not apply its scale or offset; `.gs.unpack()` is an explicit processing call.

## Executable example

```yaml
schema_version: 2
sources:
  optical:
    type: raster
    variables: [red, nir]
    require_crs: true
preprocessing:
  selected:
    call: !ref optical.__getitem__
    kwargs:
      key: [red, nir]
  valid_pixels:
    call: !ref selected.gs.to_nan
  reflectance:
    call: !ref valid_pixels.gs.unpack
postprocessing:
  prediction:
    call: !ref logits.to_dataset
    kwargs:
      name: prediction
```

Preprocessing receives `optical`. Postprocessing independently receives an xarray
DataArray named `logits`; it does not rerun preprocessing or assume how prediction
happened. Other models declare other postprocessing calls and native result types.

The supported Python equivalents are `ModelSpec`, `OperationSpec` and `Ref` from
`geosave_engine.workflow.spec`, and `Processor` from
`geosave_engine.workflow.processing`. The Python/YAML round trip preserves values
and reference tags, not arbitrary source code, formatting or comments.

## Runtime ownership

Model loading, endpoint clients and the inference mechanism remain undecided.
Their implementation must not be inferred from the presence of an `inference`
mapping. There are no schema-wide tensor layouts, output channels, class counts,
merging policies or hidden model kwargs.

Native sampling owns tile/window arguments; task-specific interpretation owns
class labels and thresholds. Resource configuration owns model batch size, device,
Prefect settings and LitServe worker/request-batch settings. User parameters specify
the requested action and inputs, separately from how execution handles them.

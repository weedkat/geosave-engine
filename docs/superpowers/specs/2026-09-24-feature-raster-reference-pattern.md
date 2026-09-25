# Feature raster and workflow reference pattern

## Status and scope

This note preserves the workflow pattern discovered while refining the
`features` module. The examples are a design catalogue, not the current schema:
this pass implements the feature API only. Workflow parsing and execution are
deferred. Postprocessing is intentionally not designed here.

## One execution rule

Every operation names one importable module-level Python function with `call`.
There is no second `method` dispatch mechanism and no callable registry.

```yaml
- call: geosave_engine.geodata.features.ndvi
  references:
    raster:
      raster: reflectance
  kwargs:
    name: ndvi
    nir: nir
    red: red
    eps: 0.0
```

The keys under `references` and `kwargs` are the function's argument names.
The workflow resolves reference descriptors, leaves literal kwargs unchanged,
combines both mappings, and makes an ordinary call:

```python
features.ndvi(
    raster=rasters["reflectance"],
    name="ndvi",
    nir="nir",
    red="red",
    eps=0.0,
)
```

This separation is important: `red` in `kwargs` is the literal variable name
`"red"`; `reflectance` appears inside a tagged raster descriptor and is looked
up. The workflow never guesses whether an arbitrary string is a reference.

`call` also covers custom model-owned computation. A project may expose one
function that composes xarray, GeoSave geodata functions, and ML utilities; the
workflow does not need to know which modules the function uses internally.

## Reference descriptors

All reference-bearing blocks use the same explicit descriptors. A descriptor
states both where a value comes from and what Python value should be passed.

| Descriptor | Resolved value | Use |
| --- | --- | --- |
| `{raster: optical}` | complete `Dataset` | Pass a source or previous preprocessing raster. |
| `{raster: optical, variables: [nir, red]}` | selected `Dataset` | Pass several variables while retaining raster context. |
| `{raster: clear-mask, variable: clear}` | one `DataArray` | Pass a single mask or band to an operation that requires an array. |
| `{current: true}` | current operation result | Chain operations within one named preprocessing recipe. |
| `{sequence: [...]}` | ordered Python list | Pass multiple rasters to functions such as `merge_bands`. |

`raster` names share one namespace containing declared sources and completed
preprocessing results. Dependency validation can therefore be topological:
forward references, unknown names, and cycles are errors before pixel work.

`variables` and `variable` are mutually exclusive. `variables` preserves a
Dataset even when it selects one variable; `variable` deliberately extracts a
DataArray. A sequence contains descriptors rather than bare raster-name strings,
so its elements cannot be confused with literals.

## Feature interface

Public feature functions:

- accept one raster `Dataset`;
- accept semantic variable selectors such as `nir` and `red`;
- accept algorithm settings such as `eps` and thresholds as ordinary keywords;
- require the caller to provide the output variable `name`;
- return a one-variable raster `Dataset` carrying that name;
- preserve coordinates, geospatial metadata, root provenance, and Dask laziness.

The implementation may use a `DataArray` internally. A `DataArray` is not the
public feature result because the result becomes a named raster node and can be
used directly as model context. A consumer can still select its one variable
explicitly through a reference when an operation needs a mask DataArray.

## Preprocessing use cases

### Select and prepare a source raster

`current` means the previous operation result in this recipe. The final result
is published as `reflectance`.

```yaml
preprocessing:
  reflectance:
    operations:
      - call: geosave_engine.geodata.transform.nodata.to_nan
        references:
          data:
            raster: optical
            variables: [blue, red, nir, swir1]

      - call: geosave_engine.geodata.transform.packing.unpack
        references:
          data:
            current: true
```

### Derive a model feature

Feature selectors remain literals because they name variables inside the
resolved raster. The returned Dataset is published as `vegetation` while its
single variable is named `ndvi`.

```yaml
  vegetation:
    operations:
      - call: geosave_engine.geodata.features.ndvi
        references:
          raster:
            raster: reflectance
        kwargs:
          name: ndvi
          nir: nir
          red: red
          eps: 1.0e-6
```

Keeping the node name separate from the variable name permits a recipe name to
describe its workflow role without silently renaming raster data.

### Build and consume a mask

A mask feature is still a one-variable Dataset. `variable` performs the explicit
Dataset-to-DataArray selection required by `nodata.mask`.

```yaml
  clear-mask:
    operations:
      - call: geosave_engine.geodata.features.scl_valid_mask
        references:
          raster:
            raster: optical
        kwargs:
          name: clear
          scl: scl
          valid_classes: [4, 5, 6, 7]

  clear-reflectance:
    operations:
      - call: geosave_engine.geodata.transform.nodata.mask
        references:
          data:
            raster: reflectance
          valid:
            raster: clear-mask
            variable: clear
        kwargs:
          fill: -9999.0
```

The same shape supports CDI, cirrus, s2cloudless, or shadow masks. Metadata such
as sun azimuth remains a raster coordinate; the feature call receives the
raster and names the coordinate with a literal selector when needed.

### Combine an ordered collection of rasters

`sequence` is an explicit collection descriptor. It resolves to a Python list
in the written order and is passed as the `rasters` argument.

```yaml
  model-context:
    operations:
      - call: geosave_engine.geodata.transform.merge.merge_bands
        references:
          rasters:
            sequence:
              - raster: clear-reflectance
              - raster: vegetation
              - raster: terrain
                variables: [height]
```

This means:

```python
merge_bands(
    rasters=[
        rasters["clear-reflectance"],
        rasters["vegetation"],
        rasters["terrain"][["height"]],
    ]
)
```

The operation itself remains responsible for its domain contract. For example,
`merge_bands` rejects mismatched grids instead of the workflow silently
reprojecting or aligning them.

### Pass multiple rasters as distinct arguments

A custom function may need inputs with different meanings rather than a
collection. Reference keys map directly to those named parameters.

```yaml
  fused-context:
    operations:
      - call: project.preprocessing.fuse_optical_and_terrain
        references:
          optical:
            raster: clear-reflectance
          terrain:
            raster: terrain
            variables: [height, slope]
        kwargs:
          name: fused
          interpolation: nearest
```

No special multi-input operation type is required: `optical` and `terrain` are
ordinary function parameters.

## Inference correspondence

Inference uses the same raster descriptors, but each key under `references` is
a model input name rather than a Python preprocessing-function argument. Layout
and dtype describe tensor conversion after the reference is resolved.

```yaml
inference:
  references:
    image:
      raster: model-context
      variables: [blue, red, nir, ndvi]
      layout: CHW
      dtype: float32
    elevation:
      raster: terrain
      variables: [height]
      layout: CHW
      dtype: float32

  tiling:
    reference:
      raster: model-context
    tile_shape: [256, 256]
    overlap: 32
```

Here `image` and `elevation` are concrete model input names. They do not imply a
variable lookup, and the word `inference` is not treated as a magic reference.
The tiling reference states which resolved grid defines the windows.

This inference shape is illustrative and is not part of the feature refactor.
Its exact schema must be reconciled with temporal windows, tensor conversion,
and the existing inference executor before implementation.

## Deliberately deferred

This pattern does not yet define postprocessing or how raw inference outputs
become named rasters. In particular, a form such as
`references: {logits: inference}` is rejected as ambiguous: it does not say
which model output, representation, dimensions, or raster grid is referenced.

Also deferred are schema migration, executor changes, persistence rules for
intermediate nodes, and whether nested reference descriptors beyond `sequence`
are useful. Those questions should be settled against real workflow tests after
the feature module refactor, without reviving `method` or adding a registry.

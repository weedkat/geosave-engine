# Model usage specification

```python
from geosave_engine.workflow import preprocess
from geosave_engine.workflow.examples.reflectance import make_spec, sample_rasters
from geosave_engine.workflow.spec import ModelSpec

spec = make_spec()
path = spec.save('artifacts/model')  # artifacts/model/model_spec.yaml
restored = ModelSpec.load(path)
prepared = preprocess(sample_rasters(), spec=restored)
prepared['reflectance'].to_dataset()
```

The [runnable YAML example](../examples/model_spec.yaml) uses native `gs.to_nan` and `gs.unpack` methods.
Recipes can select ordered `variables` before their operations. Native method
signatures are checked against their stage target before execution; custom
Python `call` paths are imported and checked on loading, without executing them. Save/load accepts a YAML file or artifact
directory, rejects duplicate YAML keys, and preserves existing model files.

| Field | Meaning |
| --- | --- |
| `schema_version` | Supported YAML format version, currently `1` |
| `sources` | Named raw raster requirements and ordered variable selections |
| `preprocessing` | Output names mapped to a starting `raster`, optional `variables`, and ordered `operations` |
| `inference.inputs` | Model argument names mapped to raster, variables, dtype, layout and optional normalization |
| `inference.tiling` | Reference raster, tile dimensions, overlap and merger window |
| `inference.time_window` | Optional size, tolerance, stride and strict/drop temporal sampling |
| `postprocessing` | Optional segmentation settings, native method or custom callable consuming merged logits |

An operation declares exactly one native `method` or custom Python `call`,
optional primitive `kwargs`, and optional
`inputs` mapping keyword argument names to other raster names. It receives the
current Dataset first for a custom call; native methods are bound to the Dataset.
Both return a Dataset. Recipes may refer to later recipes;
cycles, unknown raster names and source/output collisions fail before pixels
are read. Preprocessing retains selected sources and adds outputs to one native
DataTree. Each branch gets independent arrays and metadata; Dask data stays lazy.
Native methods and installed functions own explicit computation, reprojection or alignment.

Raster names are flat DataTree group names and may contain hyphens and dots.
When selected sources share a grid, its spatial and CRS coordinate names are
reserved for the stack root; preprocessing rejects source or output group names
that collide with them before executing operations. Model argument names are
Python identifiers. Optional source resolution is a
positive scalar or `(x, y)` pair in CRS units and validates the existing grid;
it does not reproject. Source requirements reuse `attrs.create_header` namespaces:
`root`, `data_vars`, and `coords`, with `models` for registered attrs and
`foreign` for unregistered keys. `required`, `equals`, and `one_of` use the
registered field parsers. Validation preserves lazy data and native metadata.

Inference bindings omit `variables` to use the prepared order. Channel counts
for derived recipes are checked against their actual outputs during sampling.
`tiling.raster` selects the common grid; preparation must align other inputs
explicitly. `CHW` and `TCHW` are unbatched layouts. `time_window` reuses native
`window_stack` duration and strict/drop semantics. Segmentation interprets
spatially merged logits, preserving separate temporal outputs.

Python additionally validates installed callables and dependency references.
`validated_copy()` revalidates mutable nested dictionaries before execution and
saving. See the [workflow guide](../README.md) for ingestion and prediction.

Coordinate values stay in xarray, including per-time `sun_azimuth` loaded through
ODC `with_properties`. `attrs.coords` checks their metadata. `models` and `foreign`
are siblings within each attrs scope; foreign keys must have no registered owner.
Segmentation class labels use native Legend tokens, such as `dry_land`.

# Attrs scoped models

## Intent

GeoSave owns the metadata vocabulary its pipeline needs from training to
deployment, so users never write attrs glue of their own. Every attrs model
lives in one table that also says where the model belongs, following the
attribute usage CF Appendix A assigns: G (global), D (data variable), and
C (coordinate). Value validation stays inside each model class.

This replaces the import-time registration hook and the shared-key machinery
it needs, and fixes two defects:

- an edit on a stacked array is lost on `to_raster`;
- a variable with `units` parses as both `CFVariable` and `CFCoordinate`.

## Model table

```python
# attrs/models/__init__.py
type Scope = Literal["dataset", "variable", "coordinate"]

MODELS: Mapping[Scope, tuple[type[AttrsModel], ...]] = {
    "dataset": (ACDD, GeoTIFFTags, StacMetadata, ZarrOrder),
    "variable": (CFVariable, GDALVariable, Legend, Nodata, Packing),
    "coordinate": (CFCoordinate, TimeSpec, StackedAttrs),
}
```

Each attrs mapping is parsed against exactly one scope:

| Mapping | Scope |
|---|---|
| Dataset or DataTree root | `dataset` |
| data variable | `variable` |
| DataArray root | `variable` |
| coordinate | `coordinate` |

A DataArray is one variable, as `ds["B04"].attrs is ds["B04"].variable.attrs`
already says in xarray. Nodata, packing, and legends therefore live only on
variables. A `nodata` key on a Dataset root is a foreign attr.

One rule replaces the registration checks, as a test:

```python
def test_each_attr_key_has_one_owner_per_scope():
    for models in MODELS.values():
        keys = [key for model in models for key in model.attr_keys()]
        assert len(keys) == len(set(keys))
```

Each model's `NAME` is its class name in snake_case, treating GeoTIFF as one
word: `cf_variable`, `cf_coordinate`, `gdal_variable`, `geotiff_tags`,
`stac_metadata`, `stacked_attrs`, `time_spec`, `zarr_order`, `acdd`, `legend`,
`nodata`, `packing`. A test enforces it, so a `rebase` keyword follows from the
class name.

`units` and `standard_name` may belong to both `CFVariable` and
`CFCoordinate` because no mapping is parsed against both scopes. Model `NAME`s
stay unique across the whole table, since `rebase` keywords and model-spec
configs use them.

## AttrsModel

The Pydantic subclass hook, `_field_has_converter`, `REGISTERED_MODELS`, and
`REGISTERED_ATTR_KEYS` are removed. A model is a plain Pydantic model with a
`NAME` and optional `field_keys` for fields that write several spellings.

```python
class AttrsModel(BaseModel):
    NAME: ClassVar[str]
    field_keys: ClassVar[Mapping[str, tuple[str, ...]]] = {}

    @classmethod
    def attr_keys(cls, field: str | None = None) -> tuple[str, ...]:
        """Return the attr keys one field writes, or every key the model writes."""

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, object]) -> Self | None:
        """Parse this model from a flat mapping, None when it carries none of its keys."""

    def to_attrs(self) -> dict[str, Any]: ...

    @classmethod
    def merge(cls, models: Sequence[Self | None]) -> tuple[Self, set[str]]: ...
```

`from_attrs` owns the check that a field's spellings agree, for example
`_FillValue` and `nodata`. Models that read another model use it directly
(`ACDD.from_attrs(obj.attrs)`), so `models/gdal.py` and `models/geotiff.py` no
longer import `AttrsNamespace` or rely on local imports to avoid cycles.

`resolve_model(name)` searches the table. `parse_field_value(model, field,
value)` keeps its signature and caches one `TypeAdapter` per field.

## Namespace and header

```python
@dataclass(frozen=True)
class AttrsNamespace:
    scope: Scope
    models: Mapping[str, AttrsModel]
    foreign: Mapping[str, object]

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any], scope: Scope) -> Self: ...
```

- `__post_init__` refuses a model outside `scope` and a foreign key that a
  model in `scope` writes.
- `merge` refuses namespaces of different scopes. It drops the re-parse it
  needed for shared keys.
- `to_attrs` drops the cross-model disagreement check, since one scope has one
  owner per key.

`AttrsHeader.from_attrs(root=..., data_vars=..., coords=...)` parses `root` as
`dataset`. `attrs.create_header(obj)` parses a DataArray root as `variable`.

## Rebase

`rebase` keeps its three forms. It now refuses a model or namespace whose
scope does not match the target:

```python
ds.gs.rebase(TimeSpec.from_resample("MS"), target="B04")
# ValueError: timespec belongs on a coordinate; 'B04' is a data variable
ds.gs.rebase(Nodata(fill_value=0))
# ValueError: nodata belongs on a variable; target None is this Dataset's root
```

## Merge policy

A field that changes how stored pixels decode or what they mean is marked on
its own line:

```python
MUST_AGREE = MustAgree()   # attrs/model.py

class Nodata(AttrsModel):
    fill_value: Annotated[int | float | None, MUST_AGREE] = None

class CFVariable(AttrsModel):
    standard_name: Annotated[CFPhrase, MUST_AGREE] = None
    long_name: CFPhrase = None
    units: Annotated[CFPhrase, MUST_AGREE] = None
    cell_methods: Annotated[CFPhrase, MUST_AGREE] = None
```

The marker wraps the whole field type. Pydantic drops metadata nested inside
a union member.

Marked fields:

| Model | Fields |
|---|---|
| `Nodata` | `fill_value` |
| `Packing` | `scale_factor`, `add_offset` |
| `CFVariable` | `standard_name`, `units`, `cell_methods` |
| `Legend` | `flag_values`, `flag_masks`, `flag_meanings` |

`AttrsModel.merge` raises `ValueError` when the joined objects carry a marked
field differently, including one object carrying it and another not. Unmarked
fields keep the drop-and-warn behaviour. `AttrsHeader.merge` adds the variable
name to the error with `add_note`.

This replaces `attrs.xarray._SEMANTICS` and `_check_semantics`. Because the
policy runs on namespaces, it covers DataArray, Dataset, and DataTree joins
alike, and a join mixing a DataArray with a Dataset fails on the scope check.

## Stacked round trip

The stacked array's root is the only copy of what every band shares.
`StackedAttrs` keeps each band's remaining attrs and the whole Dataset root.
It has two fields and no methods:

```python
class StackedAttrs(AttrsModel):
    variable_attrs: dict[str, dict[str, object]] | None = None   # each band's own attrs
    dataset_attrs: dict[str, object] | None = None               # the Dataset root
```

```python
# GeoRaster.to_array
array.attrs = shared                       # keys every band carries with equal values
band_attrs = StackedAttrs(
    variable_attrs={name: {key: value for key, value in ds[name].attrs.items()
                           if key not in shared} for name in names},
    dataset_attrs=dict(ds.attrs),
)

# GeoArray.to_raster
raster[name].attrs = {**band_attrs.variable_attrs[name], **array.attrs}
raster.attrs = band_attrs.dataset_attrs
```

`shared` is a flat intersection compared with `attrs_equal`, not a model
merge. Bands with different nodata or units are legitimate in one stack, so
the merge policy must not apply here.

Guarantees:

- Dataset → DataArray → Dataset restores every band's attrs and the Dataset
  root.
- An edit to the stacked array's root reaches every band on `to_raster`.
- Values come back JSON-normalised (`np.float32(0)` becomes `0.0`, a tuple
  becomes a list, a `Path` becomes a string), as today. `AttrsModel.to_attrs`
  converts NumPy values for every model.

`StackedAttrs.shared`, `restore`, and `restore_root` are removed. In
`to_array` and `to_raster`, the locals `carried` and `parked` are renamed to
`band_attrs` and `shared`.

## Model spec

`AttrsRequirement` validates each section against its scope: `root` names
dataset models, and `data_vars` and `coords` name variable and coordinate
models. The foreign-key collision check moves from `NamespaceRequirement` to
`AttrsRequirement` and uses that section's scope keys. It no longer uses the
removed `REGISTERED_ATTR_KEYS`.

## User-visible changes

| Before | After |
|---|---|
| A user subclass of `AttrsModel` registers itself | Only models in `MODELS` are parsed |
| `nodata` on a Dataset root parses as `Nodata` | It is a foreign attr and round-trips verbatim |
| `_FillValue` on a coordinate parses as `Nodata` | It is a foreign attr |
| A data variable with `units` also carries `CFCoordinate` | Only `CFVariable` |
| `rebase` writes any model anywhere | Refuses a model outside the target's scope |
| A stacked array carries dataset models on its root | Its root is one variable; dataset attrs ride on `band` |
| An edit on a stacked array is lost on `to_raster` | It reaches every band |
| A `Path` in Dataset attrs survives `to_array` as a `Path` | It comes back as its JSON spelling, a string; a value with no JSON spelling raises `TypeError` |
| Decoding conflicts are checked only on Dataset data variables and DataArray roots | Checked on every join, raised by the model |

## Risks

- Stacked Zarr stores written before this change hold `dataset_attrs` with
  only shadowed keys. Restoring them loses the other Dataset root attrs. This
  is accepted in Alpha.
- Model-spec configs that place a model in the wrong section now fail
  validation. None exist in this repository.

## Testing

- **Table:** one owner per key per scope; unique `NAME`s across scopes;
  `resolve_model` finds every model.
- **Parsing:** a variable with `units` carries only `CFVariable`; a Dataset
  root `nodata` and a coordinate `_FillValue` are foreign; a DataArray root
  parses `Nodata`.
- **Rebase:** each wrong-scope model or namespace raises before any attrs
  change, including `inplace=True`.
- **Merge:** each marked field raises on a value or presence mismatch with
  the variable named; an unmarked field drops with `DroppedAttrsWarning`;
  mixing a DataArray and a Dataset raises; STAC provenance still accumulates.
- **Stacking:** the round trip restores band and root attrs; a root edit
  reaches every band; bands with different nodata stack without error.
- **Model spec:** a requirement naming a model outside its section's scope
  fails validation.
- **Suites:** the focused attrs, core, transform, io, and model_spec suites,
  then the full suite and `ruff check`. The existing tests that pin the old
  behaviour are rewritten to the new contract: Dataset-root nodata, the
  shadow logic, the `Path` root value (now its string spelling), and
  test-registered models.

# Attrs Scoped Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Paths moved 2026-10-03.** `src/geosave_engine/model_spec/` is now `src/geosave_engine/model/spec/`, and `tests/model_spec/` is now `tests/model/spec/`. Read every such path below accordingly.

**Goal:** Replace the attrs registration hook with one scoped model table, move the pixel-semantics merge policy onto model fields, and make the stacked-array round trip store each fact once.

**Architecture:** `attrs/models/__init__.py` owns `MODELS`, mapping each scope (dataset, variable, coordinate) to its models. Every `AttrsNamespace` is parsed against exactly one scope, `rebase` refuses writes outside a target's scope, and `AttrsModel.merge` refuses disagreement on fields marked `MUST_AGREE`. `GeoRaster.to_array` keeps shared band attrs only on the array root, and `StackedAttrs` keeps the rest.

**Tech Stack:** Python 3.12, Pydantic 2, xarray, odc-geo, pytest

**Spec:** `docs/superpowers/specs/2026-10-03-attrs-scoped-models-design.md`

## Global Constraints

- Run plan A (`2026-10-03-attrs-isolated-fixes.md`) first.
- No compatibility aliases for `REGISTERED_MODELS`, `REGISTERED_ATTR_KEYS`, `StackedAttrs.from_variables`, `shared`, `restore`, or `restore_root`.
- `attrs/models/*.py` must not import `attrs.namespace` or `attrs.header`.
- Name locals for what they hold; no `carried`, `parked`, `placed`, or similar past participles.
- Metadata-only operations keep pixels lazy.
- Commit only the listed paths with `git commit -m ... -- <paths>`; the working tree holds unrelated user work.

## Review Focus

- A Dataset written to GeoTIFF, Zarr, or NetCDF and read back still carries the same typed models per scope; Task 6 runs the io suites.
- `ds.gs.rebase(Nodata(fill_value=0))` on a Dataset root must now raise; library writers must not rely on it. Task 3 runs the core and io suites to catch any writer that does.
- Bands with different nodata must still stack and unstack; Task 5 tests this.
- A DataArray without `StackedAttrs` must unstack its root attrs onto every band, not onto the Dataset root; Task 5 tests this.
- `attrs.merge` of DataArrays that disagree on nodata must name the conflict; Task 4 tests this.

---

### Task 1: Scoped model table and `AttrsModel.from_attrs`

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/model.py`
- Modify: `src/geosave_engine/geodata/attrs/models/__init__.py`
- Modify: `src/geosave_engine/geodata/attrs/models/gdal.py`, `models/geotiff.py`, `models/stacked.py`
- Modify: `src/geosave_engine/geodata/attrs/__init__.py`
- Test: `tests/geodata/attrs/test_models.py` (create), `tests/geodata/attrs/test_field_values.py`, `tests/geodata/attrs/test_header_stamping.py`

**Interfaces:**
- Produces: `Scope`, `MODELS`, `resolve_model(model) -> type[AttrsModel]`, `model_scope(model) -> Scope`, `scope_keys(scope) -> frozenset[str]` in `attrs.models`; `AttrsModel.attr_keys(field=None)`, `AttrsModel.from_attrs(attrs) -> Self | None`; `MUST_AGREE` in `attrs.model`.

- [ ] **Step 1: Write table tests**

Create `tests/geodata/attrs/test_models.py`:

```python
import pytest

from geosave_engine.geodata.attrs import MODELS, Nodata, TimeSpec, resolve_model
from geosave_engine.geodata.attrs.models import model_scope


def test_each_attr_key_has_one_owner_per_scope() -> None:
    for models in MODELS.values():
        keys = [key for model in models for key in model.attr_keys()]
        assert len(keys) == len(set(keys))


def test_model_names_are_unique_across_scopes() -> None:
    names = [model.NAME for models in MODELS.values() for model in models]
    assert len(names) == len(set(names))


def test_every_model_resolves_by_name_and_class() -> None:
    for models in MODELS.values():
        for model in models:
            assert resolve_model(model.NAME) is model
            assert resolve_model(model) is model


def test_a_model_outside_the_table_is_refused() -> None:
    class Calibration(Nodata):
        NAME = "calibration"

    with pytest.raises(TypeError, match="not a GeoSave attrs model"):
        resolve_model(Calibration)
    with pytest.raises(KeyError, match="calibration"):
        resolve_model("calibration")


def test_scopes_follow_cf_usage() -> None:
    assert model_scope(Nodata) == "variable"
    assert model_scope(TimeSpec) == "coordinate"


def test_one_field_reads_any_of_its_spellings() -> None:
    assert Nodata.from_attrs({"nodata": 0}) == Nodata(fill_value=0)
    assert Nodata.from_attrs({"units": "1"}) is None
    with pytest.raises(ValueError, match="they spell one Nodata.fill_value"):
        Nodata.from_attrs({"_FillValue": 0, "nodata": -9999})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/attrs/test_models.py -q`
Expected: ImportError, because `MODELS` does not exist.

- [ ] **Step 3: Replace the hook with plain class methods**

In `model.py`, delete `_MODEL_TYPES`, `_FIELD_BY_ATTR_KEY`, `REGISTERED_MODELS`, `REGISTERED_ATTR_KEYS`, `_RESERVED_MODEL_NAMES`, `__pydantic_init_subclass__`, `_field_parsers`, `_field_has_converter`, and `resolve_model`. Add:

```python
class MustAgree:
    """Mark a field that changes how stored pixels decode or what they mean."""


MUST_AGREE = MustAgree()


class AttrsModel(BaseModel):
    NAME: ClassVar[str]
    field_keys: ClassVar[Mapping[str, tuple[str, ...]]] = {}

    @classmethod
    def attr_keys(cls, field: str | None = None) -> tuple[str, ...]:
        """Return the attr keys one field writes, or every key this model writes."""
        if field is not None:
            return cls.field_keys.get(field, (field,))
        return tuple(key for name in cls.model_fields for key in cls.attr_keys(name))

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> Self | None:
        """Parse this model from a flat attrs mapping.

        Args:
            attrs: Flat attrs mapping, possibly carrying other models' keys.

        Returns:
            The model, or None when the mapping carries none of its keys.

        Raises:
            ValueError: A field's spellings are set to different values.
            ValidationError: A value does not satisfy its field.
        """
        values: dict[str, Any] = {}
        for field in cls.model_fields:
            spellings = [key for key in cls.attr_keys(field) if key in attrs]
            if not spellings:
                continue
            # One store may hold a spelling as text and another as a number.
            parsed = [parse_field_value(cls, field, attrs[key]) for key in spellings]
            for spelling, other in zip(spellings[1:], parsed[1:], strict=True):
                if not attrs_equal(other, parsed[0]):
                    raise ValueError(
                        f"{spellings[0]!r} is {parsed[0]!r} but {spelling!r} is "
                        f"{other!r}; they spell one {cls.__name__}.{field}, so "
                        f"set one of them"
                    )
            values[field] = attrs[spellings[0]]
        return cls(**values) if values else None
```

`to_attrs` and `merge` use `cls.attr_keys(field)` instead of `cls.field_keys[field]`. `parse_field_value` caches its adapter:

```python
@cache
def _field_adapter(model: type[AttrsModel], field: str) -> TypeAdapter[Any]:
    return TypeAdapter(model.model_fields[field].rebuild_annotation())


def parse_field_value(model: type[AttrsModel], field: str, value: object) -> Any:
    return _field_adapter(model, field).validate_python(value)
```

`to_attrs` passes `fallback=_json_value` to `model_dump`, converting NumPy scalars and arrays with `.tolist()` and raising `TypeError` for anything else. This moves the conversion that `StackedAttrs.from_variables` did into every model.

- [ ] **Step 4: Write the table**

In `models/__init__.py`:

```python
type Scope = Literal["dataset", "variable", "coordinate"]

# Where each model lives, after the attribute usage in CF Appendix A.
MODELS: Mapping[Scope, tuple[type[AttrsModel], ...]] = MappingProxyType(
    {
        "dataset": (ACDD, GeoTIFFTags, StacMetadata, ZarrOrder),
        "variable": (CFVariable, GDALVariable, Legend, Nodata, Packing),
        "coordinate": (CFCoordinate, TimeSpec, StackedAttrs),
    }
)


def resolve_model(model: type[AttrsModel] | str) -> type[AttrsModel]:
    """Find a GeoSave attrs model by class or `NAME`."""
    for models in MODELS.values():
        for candidate in models:
            if candidate is model or candidate.NAME == model:
                return candidate
    if isinstance(model, str):
        raise KeyError(f"no attrs model is named {model!r}")
    raise TypeError(f"{model!r} is not a GeoSave attrs model")


def model_scope(model: type[AttrsModel] | str) -> Scope:
    """Name the scope a model belongs to."""
    model = resolve_model(model)
    return next(scope for scope, models in MODELS.items() if model in models)


@cache
def scope_keys(scope: Scope) -> frozenset[str]:
    """Return every attr key a model in `scope` writes."""
    return frozenset(key for model in MODELS[scope] for key in model.attr_keys())
```

- [ ] **Step 5: Stop models importing the namespace**

- `GDALVariable.rgb_indices`: `bands = (cls.from_attrs(variable.attrs) for variable in ds.data_vars.values())`.
- `GeoTIFFTags.from_xarray`: `carried = cls.from_attrs(obj.attrs) or cls()` becomes `tags = cls.from_attrs(obj.attrs) or cls()`, and `acdd = ACDD.from_attrs(obj.attrs)`. Move the `ACDD` import to module level and delete the local-import comment.
- `StackedAttrs` drops `from_variables`, `shared`, `restore`, `restore_root`, and `_numpy_value`. It keeps its two fields, with the docstrings from the spec.

- [ ] **Step 6: Update the exports**

`attrs/__init__.py` exports `MODELS`, `Scope`, `MUST_AGREE`, `resolve_model`, and `model_scope`. It drops `REGISTERED_MODELS`.

- [ ] **Step 7: Retire tests of the hook**

In `test_header_stamping.py`, delete:
- `test_field_keys_are_nonempty_tuples_of_attr_names`;
- the registry cleanup in its helpers;
- `test_spellings_set_to_different_values_are_refused`, now covered in `test_models.py`.

Rewrite `test_only_the_accumulating_model_writes_its_own_merge` to iterate `MODELS.values()`. In `test_field_values.py`, delete the `try/finally` registry pop around `DecoratedField`.

- [ ] **Step 8: Run the attrs suite**

Run: `uv run pytest tests/geodata/attrs/test_models.py tests/geodata/attrs/test_field_values.py -q`
Expected: pass. Other attrs tests fail until Task 2 lands, because the namespace still imports the deleted registry.

Tasks 1 and 2 share one commit, since the namespace cannot import a registry that no longer exists.

---

### Task 2: Scoped namespace and header

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/namespace.py`, `header.py`, `headers/xarray.py`, `headers/gdal.py`
- Modify: `src/geosave_engine/model_spec/rasters.py`
- Test: `tests/geodata/attrs/test_namespace.py` (rewrite), `tests/geodata/attrs/test_header_stamping.py`, `tests/model_spec/test_rasters.py`

**Interfaces:**
- Consumes: `MODELS`, `Scope`, `resolve_model`, `model_scope`, `scope_keys`.
- Produces: `AttrsNamespace(scope, models, foreign)`, `AttrsNamespace.from_attrs(attrs, scope)`, and an `AttrsHeader` whose root scope is `dataset` or `variable`.

- [ ] **Step 1: Write the scope tests**

Replace `tests/geodata/attrs/test_namespace.py`:

```python
import pytest
import xarray as xr

from geosave_engine.geodata.attrs import (
    AttrsNamespace,
    CFCoordinate,
    CFVariable,
    Nodata,
    create_header,
)


def test_a_variable_with_units_carries_only_variable_semantics() -> None:
    namespace = AttrsNamespace.from_attrs({"units": "1"}, "variable")
    assert namespace.get(CFVariable) == CFVariable(units="1")
    assert namespace.get(CFCoordinate) is None


def test_nodata_on_a_dataset_root_is_foreign() -> None:
    header = create_header(xr.Dataset(attrs={"nodata": 0}))
    assert header.root.get(Nodata) is None
    assert header.root.foreign == {"nodata": 0}


def test_fill_value_on_a_coordinate_is_foreign() -> None:
    header = create_header(xr.Dataset(coords={"x": ("x", [0.0], {"_FillValue": 0.0})}))
    assert header.coords["x"].foreign == {"_FillValue": 0.0}


def test_a_dataarray_root_is_one_variable() -> None:
    header = create_header(xr.DataArray([1], dims="x", attrs={"nodata": 0}))
    assert header.root.scope == "variable"
    assert header.root.get(Nodata) == Nodata(fill_value=0)


def test_a_namespace_refuses_a_model_from_another_scope() -> None:
    with pytest.raises(ValueError, match="nodata belongs on a variable"):
        AttrsNamespace("dataset", models={"nodata": Nodata(fill_value=0)})


def test_namespaces_of_different_scopes_do_not_merge() -> None:
    with pytest.raises(ValueError, match="one kind"):
        AttrsNamespace.merge(
            [AttrsNamespace("dataset"), AttrsNamespace("variable")]
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/attrs/test_namespace.py -q`
Expected: FAIL, because `from_attrs` takes no scope.

- [ ] **Step 3: Implement the scoped namespace**

```python
@dataclass(frozen=True)
class AttrsNamespace:
    scope: Scope
    models: Mapping[str, AttrsModel] = field(default_factory=dict[str, AttrsModel])
    foreign: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        for name, model in self.models.items():
            expected = resolve_model(name)
            if not isinstance(model, expected):
                raise TypeError(
                    f"attrs model {name!r} must be {expected.__name__}, "
                    f"got {type(model).__name__}"
                )
            if expected not in MODELS[self.scope]:
                raise ValueError(
                    f"{name} belongs on a {model_scope(expected)}, "
                    f"not a {self.scope}"
                )
        collisions = sorted(self.foreign.keys() & scope_keys(self.scope))
        if collisions:
            raise ValueError(f"foreign attrs collide with model keys: {collisions}")

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any], scope: Scope) -> Self:
        flat = {str(key): value for key, value in attrs.items()}
        models = {}
        for model_type in MODELS[scope]:
            model = model_type.from_attrs(flat)
            if model is not None:
                models[model_type.NAME] = model
        keys = scope_keys(scope)
        foreign = {key: value for key, value in flat.items() if key not in keys}
        return cls(scope, models, foreign)
```

`merge` refuses mixed scopes with `ValueError("... a join combines objects of one kind")`, deletes the re-parse line, and passes `scope` to the result. `to_attrs` drops the `owners` disagreement check:

```python
        values: dict[str, Any] = {}
        for model in self.models.values():
            values.update(model.to_attrs())
        return {
            **self.foreign,
            **{key: value for key, value in values.items() if value is not None},
        }
```

- [ ] **Step 4: Scope the header**

- `AttrsHeader.root` defaults to `AttrsNamespace("dataset")`.
- `AttrsHeader.__post_init__` refuses a root not scoped `dataset` or `variable`, a data variable not scoped `variable`, and a coordinate not scoped `coordinate`.
- `AttrsHeader.from_attrs` parses `root` as `dataset`, data variables as `variable`, and coordinates as `coordinate`.
- In `_merge_namesakes`, a `ValueError` from `AttrsNamespace.merge` gets `error.add_note(f"in {name!r}")` and is re-raised.

In `headers/xarray.py`, a DataArray builds its header directly:

```python
    if isinstance(obj, xr.DataArray):
        return AttrsHeader(
            root=AttrsNamespace.from_attrs(obj.attrs, "variable"),
            coords={
                str(name): AttrsNamespace.from_attrs(obj.coords[name].attrs, "coordinate")
                for name in sorted(obj.coords)
            },
        )
```

In `headers/gdal.py`, `Legend.from_attrs(band_attrs)` and `GDALVariable.from_attrs(band_attrs)` replace the two `AttrsNamespace.from_attrs(...).get(...)` calls, and `carried` is renamed `band_attrs`.

- [ ] **Step 5: Scope the model-spec sections**

In `model_spec/rasters.py`, `NamespaceRequirement` drops the `REGISTERED_ATTR_KEYS` check and gains:

```python
    def validate_scope(self, scope: Scope, *, where: str) -> None:
        """Refuse models and foreign keys that do not belong in `scope`."""
        if collision := self.foreign.fields & scope_keys(scope):
            raise ValueError(f"{where}: use models for attrs {sorted(collision)}")
        for name in self.models:
            if (owner := model_scope(name)) != scope:
                raise ValueError(f"{where}.{name} belongs on a {owner}, not a {scope}")
```

`AttrsRequirement` calls it from a `model_validator(mode="after")` with `("root", "dataset")`, `("data_vars", "variable")`, and `("coords", "coordinate")`. Add to `tests/model_spec/test_rasters.py`:

```python
def test_a_model_outside_its_section_scope_is_refused() -> None:
    with pytest.raises(ValidationError, match="nodata belongs on a variable"):
        AttrsRequirement.model_validate({"root": {"models": {"nodata": {}}}})
```

- [ ] **Step 6: Give every direct construction its scope**

In `test_header_stamping.py` (24 sites), `test_field_values.py`, `tests/geodata/stac/test_source.py`, `tests/geodata/utils/io/test_geotiff.py`, and the source sites in `core/raster.py`, `features/_raster.py`, `transform/composite.py`, and `utils/io/geotiff.py`:
- each `AttrsNamespace(...)` gets the scope of the mapping it describes as its first argument;
- each `AttrsNamespace.from_attrs(x)` gets the scope as its second argument.

Read each site; don't regex-replace.

- [ ] **Step 7: Run the suites**

Run: `uv run pytest tests/geodata/attrs tests/model_spec -q`
Expected: pass.

- [ ] **Step 8: Commit Tasks 1 and 2**

```bash
git commit -m "refactor: parse attrs against one scoped model table" -- \
  src/geosave_engine/geodata/attrs src/geosave_engine/model_spec/rasters.py \
  tests/geodata/attrs tests/model_spec/test_rasters.py \
  <each source and test file touched in Step 6>
```

---

### Task 3: `rebase` refuses writes outside a target's scope

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/xarray.py`
- Test: `tests/geodata/attrs/test_header_stamping.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_rebase_refuses_a_model_outside_the_target_scope() -> None:
    ds = xr.Dataset({"red": ("time", [1])}, coords={"time": [0]})
    with pytest.raises(ValueError, match="timespec belongs on a coordinate"):
        rebase(ds, TimeSpec.from_resample("MS"), target="red")
    with pytest.raises(ValueError, match="nodata belongs on a variable"):
        rebase(ds, nodata={"fill_value": 0})


def test_rebase_refuses_a_namespace_from_another_scope_before_writing() -> None:
    ds = xr.Dataset({"red": ("x", [1], {"units": "1"})})
    with pytest.raises(ValueError, match="belongs on a variable"):
        rebase(ds, create_header(ds).data_vars["red"], inplace=True)
    assert ds.red.attrs == {"units": "1"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/attrs/test_header_stamping.py -k scope -q`
Expected: FAIL with `DID NOT RAISE`.

- [ ] **Step 3: Check scopes before writing**

Replace `_resolve_target` with a version that also names the target's scope, and check every update against it before any attrs change:

```python
def _resolve_target(obj: XarrayObject, target: str | None) -> tuple[XarrayObject, Scope]:
    """Return the object holding `target`'s attrs and the scope those attrs take."""
    if target is None:
        return obj, "variable" if isinstance(obj, xr.DataArray) else "dataset"
    if target in _var_names(obj):
        return obj[target], "variable"
    if target in _coord_names(obj):
        return obj[target], "coordinate"
    raise ValueError(...)
```

Each update carries the scope it writes:
- a header's root carries `header.root.scope`, and each header variable carries its namespace's scope;
- a namespace patch carries `namespace.scope`;
- a model patch carries `model_scope` of each model, including keyword models.

A mismatch raises:

```python
raise ValueError(
    f"{label} belongs on a {expected}; "
    f"{'the root' if name is None else repr(name)} of this "
    f"{type(obj).__name__} holds {actual} attrs"
)
```

`label` is the model `NAME` for model patches, or "this namespace" or "this header" otherwise. Delete `_check_semantics`, `_SEMANTICS`, and `_MISSING`; Task 4 moves that policy onto the models.

- [ ] **Step 4: Run the core, transform, and io suites**

Run: `uv run pytest tests/geodata -q`
Expected: pass, except the stacked round-trip tests rewritten in Task 5. A writer that puts a variable model on a Dataset root shows up here and must target its variables instead.

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: refuse attrs written outside their scope" -- \
  src/geosave_engine/geodata/attrs/xarray.py tests/geodata/attrs/test_header_stamping.py
```

---

### Task 4: Fields that decode pixels refuse disagreement

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/model.py` (`AttrsModel.merge`)
- Modify: `models/nodata.py`, `models/packing.py`, `models/cf.py`, `models/legend.py`
- Test: `tests/geodata/attrs/test_header_stamping.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"nodata": 0}, {"nodata": -1}),
        ({"nodata": 0}, {}),
        ({"scale_factor": 1e-4}, {"scale_factor": 2e-4}),
        ({"units": "1"}, {"units": "K"}),
        ({"flag_values": [0, 1], "flag_meanings": "a b"}, {}),
    ],
)
def test_merge_refuses_decoding_fields_that_disagree(first, second) -> None:
    rasters = [xr.Dataset({"red": ("x", [1], attrs)}) for attrs in (first, second)]
    with pytest.raises(ValueError, match="must agree") as raised:
        merge(rasters)
    assert "in 'red'" in raised.value.__notes__


def test_merge_drops_a_label_that_disagrees() -> None:
    rasters = [
        xr.Dataset({"red": ("x", [1], {"long_name": name})}) for name in ("a", "b")
    ]
    with pytest.warns(DroppedAttrsWarning, match="red.long_name"):
        assert merge(rasters).data_vars["red"].to_attrs() == {}


def test_merge_refuses_a_dataarray_with_a_dataset() -> None:
    with pytest.raises(ValueError, match="one kind"):
        merge([xr.DataArray([1], dims="x", name="red"), xr.Dataset({"red": ("x", [1])})])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/attrs/test_header_stamping.py -k "decoding or label or dataarray_with" -q`
Expected: decoding cases FAIL with `DID NOT RAISE`, because Task 3 removed `_check_semantics`.

- [ ] **Step 3: Mark the fields**

```python
fill_value: Annotated[int | float | None, MUST_AGREE] = None                     # Nodata
scale_factor: Annotated[float | None, MUST_AGREE] = None                         # Packing
add_offset: Annotated[float | None, MUST_AGREE] = None                           # Packing
standard_name: Annotated[CFPhrase, MUST_AGREE] = None                            # CFVariable
units: Annotated[CFPhrase, MUST_AGREE] = None                                    # CFVariable
cell_methods: Annotated[CFPhrase, MUST_AGREE] = None                             # CFVariable
flag_values: Annotated[list[int] | None, BeforeValidator(parse_collection_text), MUST_AGREE] = None
flag_masks: Annotated[list[int] | None, BeforeValidator(parse_collection_text), MUST_AGREE] = None
flag_meanings: Annotated[str | None, MUST_AGREE] = None                          # Legend
```

Add a test that each marked field carries `MUST_AGREE` in `model_fields[name].metadata`. Pydantic drops metadata nested in a union member, so this test is the guard.

- [ ] **Step 4: Refuse in `AttrsModel.merge`**

At the top of `merge`, after the empty check:

```python
        for name, field in cls.model_fields.items():
            if MUST_AGREE not in field.metadata:
                continue
            values = [None if model is None else getattr(model, name) for model in models]
            if any(not attrs_equal(value, values[0]) for value in values[1:]):
                raise ValueError(
                    f"{cls.NAME}.{name} must agree, and the joined objects "
                    f"carry {values}; align it before joining them"
                )
```

- [ ] **Step 5: Run the suites**

Run: `uv run pytest tests/geodata -q`
Expected: pass, except the stacked round-trip tests rewritten in Task 5.

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor: let decoding fields refuse disagreement on join" -- \
  src/geosave_engine/geodata/attrs tests/geodata/attrs/test_header_stamping.py
```

---

### Task 5: The stacked round trip stores each fact once

**Files:**
- Modify: `src/geosave_engine/geodata/core/raster.py` (`to_array`), `src/geosave_engine/geodata/core/array.py` (`to_raster`)
- Test: `tests/geodata/core/test_raster.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_an_edit_on_the_stacked_array_reaches_every_band():
    source = raster({"red": np.ones(geobox().shape, "int16")}, geobox())
    source.red.attrs = {"nodata": 0, "units": "1"}
    stacked = rebase(source.gs.to_array(), Nodata(fill_value=-1))
    assert stacked.gs.to_raster().red.gs.attrs.root.get(Nodata).fill_value == -1


def test_bands_with_different_nodata_round_trip():
    source = raster(
        {"red": np.ones(geobox().shape, "int16"), "nir": np.ones(geobox().shape, "int16")},
        geobox(),
    )
    source.red.attrs = {"nodata": 0, "units": "1"}
    source.nir.attrs = {"nodata": -1, "units": "1"}
    source.attrs = {"title": "S2"}
    stacked = source.gs.to_array()
    assert stacked.attrs == {"units": "1"}
    restored = stacked.gs.to_raster()
    assert restored.red.attrs == {"nodata": 0, "units": "1"}
    assert restored.nir.attrs == {"nodata": -1, "units": "1"}
    assert restored.attrs == {"title": "S2"}


def test_an_array_without_stacked_attrs_gives_its_attrs_to_every_band():
    array = source_array().assign_attrs(nodata=0)   # helper already in this file
    restored = array.gs.to_raster()
    assert restored.attrs == {}
    assert all(band.attrs == {"nodata": 0} for band in restored.data_vars.values())
```

Rewrite the four tests that pin the old mixing, as the spec lists:
- `test_band_round_trip_preserves_root_nodata_aliases` and `test_stacking_preserves_nan_nodata_at_each_scope` assert the Dataset-root `nodata` comes back as a foreign attr;
- `test_band_round_trip_keeps_edits_to_root_keys_not_lifted_from_bands` is deleted;
- `test_band_stacking_keeps_unrelated_python_root_values` asserts `to_array` raises `TypeError` for a `Path`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/core/test_raster.py -q`
Expected: the new tests FAIL; `to_array` still calls the removed `StackedAttrs.from_variables`.

- [ ] **Step 3: Implement `to_array`**

```python
        array = self._stacked(dtype)
        band_attrs = {name: dict(self._data[name].attrs) for name in self.variables}
        first, *rest = band_attrs.values()
        # A key every band carries with one value describes the stacked array.
        shared = {
            key: value
            for key, value in first.items()
            if all(key in other and attrs.attrs_equal(other[key], value) for other in rest)
        }
        array.attrs = shared
        stacked = attrs.StackedAttrs(
            variable_attrs={
                name: {key: value for key, value in held.items() if key not in shared}
                for name, held in band_attrs.items()
            },
            dataset_attrs=dict(self._data.attrs),
        )
        attrs.rebase(array, stacked, target=BAND_DIMENSION, inplace=True)
        return cast("DataArray", array)
```

Update the `Returns:` docstring to say the root carries what every band shares, and `band` carries each band's own attrs and the Dataset's.

- [ ] **Step 4: Implement `to_raster`**

```python
        stacked = self.attrs.coords[BAND_DIMENSION].get(attrs.StackedAttrs)
        raster = cast("Dataset", self._data.to_dataset(dim=BAND_DIMENSION))
        own = {} if stacked is None else stacked.variable_attrs or {}
        for name in raster.data_vars:
            raster[name].attrs = {**own.get(str(name), {}), **self._data.attrs}
        raster.attrs = {} if stacked is None else dict(stacked.dataset_attrs or {})
        return raster
```

- [ ] **Step 5: Run the core and io suites**

Run: `uv run pytest tests/geodata/core tests/geodata/utils/io -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git commit -m "fix: keep stacked array edits on unstack" -- \
  src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/array.py \
  tests/geodata/core/test_raster.py
```

---

### Task 6: Full verification and docs

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 2: Lint and docstrings**

Run: `uv run ruff check . && uv run python scripts/check_docstrings.py $(git diff --name-only -- src | tr '\n' ' ')`
Expected: no new findings.

- [ ] **Step 3: Scope probe**

Re-run the scope probe from the spec discussion (wrap `attrs.headers.xarray.create_header` and record model names per scope across the suite). Expected: each model appears only in its own scope.

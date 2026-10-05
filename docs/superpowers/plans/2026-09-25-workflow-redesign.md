# Workflow Package Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scattered workflow implementation with explicit specs, configs, tasks, and flows while preserving lazy xarray processing and making tensor conversion an explicit inference call.

**Architecture:** `workflow.specs` owns portable `model_spec.yaml` declarations, `workflow.configs` validates primitive deployment parameters, `workflow.tasks` contains small Prefect tasks, and `workflow.flows` contains deployable orchestration. Preprocessing executes ordered call declarations lazily; inference reuses those declarations but remains validation-only until sampling is designed.

**Tech Stack:** Python 3.12, Pydantic 2, PyYAML, xarray/Dask, PyTorch, Prefect 3, pystac-client, odc-stac, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-25-workflow-redesign.md`

## Global Constraints

- Preserve unrelated working-tree changes and stage only files owned by the active task.
- Do not inspect or modify notebooks.
- Do not add compatibility aliases, migration adapters, duplicate final execution paths, an operation registry, or inferred string references.
- No public or private symbol under `src/geosave_engine/workflow` may use the word `acquire`.
- `specs` and `configs` must import neither Prefect nor Lightning; Prefect decorators live only under `tasks` and `flows`.
- Tasks are short `@task` functions with one observable operation; flows validate primitive parameters and orchestrate task dependencies.
- Preprocessing remains lazy and returns native values. Tensor conversion occurs only on a bounded sample through an explicit inference call.
- Do not implement sampling, tiling changes beyond removing `TileDataset(dtype=...)`, merging, inference execution, model loading, prediction, or postprocessing.
- `model_spec.yaml` remains the only workflow YAML artifact; JSON is not a workflow-spec persistence format.
- Use `UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache` if the default uv cache is unwritable.

## Review Focus

- A malformed source setting nested under `query` or `load` must fail in `IngestConfig.model_validate` before HTTP is attempted; Task 3 adds this test.
- An inference-only missing import must not prevent model-spec loading or preprocessing execution; Tasks 2 and 5 add this test.
- A dtype string unsupported by PyTorch must fail with the requested name in the error rather than silently using float32; Task 1 adds this test.
- A destination appearing after initial config validation must still be protected from overwrite when `save_stack` publishes its staged store; Task 4 adds this race test.
- A source task returning a lazy raster must not ask Prefect to cache or persist its native result; Task 4 inspects the task configuration and Task 6 exercises it through the real flow.

---

### Task 1: Make native tensor conversion explicit and dtype-preserving

**Files:**
- Modify: `src/geosave_engine/geodata/core/base.py`
- Modify: `src/geosave_engine/geodata/core/array.py`
- Modify: `src/geosave_engine/geodata/core/raster.py`
- Modify: `src/geosave_engine/geodata/core/stack.py`
- Modify: `src/geosave_engine/geodata/datasets/tiles.py`
- Modify: `tests/geodata/core/test_model_io.py`
- Modify: `tests/geodata/datasets/test_tiles.py`
- Modify as required by the signature change: `tests/ml/tasks/test_semantic_segmentation.py`

**Interfaces:**
- Consumes: existing `GeoArray.gs.to_numpy`, `GeoRaster.gs.to_numpy`, and `GeoStack.gs.to_numpy` ordering contracts.
- Produces: `to_tensor(*, dtype: str | torch.dtype | None = None)` on DataArray, Dataset, and DataTree accessors; `TileDataset(tiles, *, model_context=None)` without a dtype policy.

- [ ] **Step 1: Change the tests to state the new dtype contract**

```python
def test_to_tensor_preserves_the_prepared_dtype_by_default() -> None:
    tensor = optical().gs.to_tensor()
    assert tensor.dtype is torch.uint16


def test_to_tensor_accepts_a_yaml_dtype_name() -> None:
    tensor = optical().gs.to_tensor(dtype="float32")
    assert tensor.dtype is torch.float32


def test_to_tensor_rejects_an_unknown_dtype_name() -> None:
    with pytest.raises(ValueError, match="not-a-dtype"):
        optical().gs.to_tensor(dtype="not-a-dtype")


def test_tile_dataset_has_no_sampling_dtype_policy() -> None:
    with pytest.raises(TypeError, match="dtype"):
        TileDataset(Tiles([_raster(256, 256)], (256, 256)), dtype=torch.float16)
```

Add corresponding DataArray and DataTree assertions so all three accessors preserve dtype and accept `"float32"`. Keep the existing axis-order and mixed-variable-dtype tests.

- [ ] **Step 2: Run the focused tests and verify the intended failures**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/geodata/core/test_model_io.py \
  tests/geodata/datasets/test_tiles.py -q
```

Expected: failures show the current implicit float32 default, string dtype rejection, and accepted `TileDataset(dtype=...)` argument.

- [ ] **Step 3: Implement one shared explicit conversion rule**

Replace the current implicit default in `geodata/core/base.py` with the equivalent of:

```python
def tensor(
    pixels: Callable[[DTypeLike | None], np.ndarray],
    dtype: str | torch.dtype | None = None,
) -> torch.Tensor:
    import torch

    if isinstance(dtype, str):
        target = getattr(torch, dtype, None)
        if not isinstance(target, torch.dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
    else:
        target = dtype

    if target is None:
        values = np.ascontiguousarray(pixels(None))
        try:
            return torch.as_tensor(values)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot convert numpy dtype {values.dtype} to a torch tensor"
            ) from error

    try:
        reading = torch.empty(0, dtype=target).numpy().dtype
    except TypeError:
        reading = np.dtype("float32")
    return torch.as_tensor(np.ascontiguousarray(pixels(reading)), dtype=target)
```

Update every accessor annotation/docstring to accept `str | torch.dtype | None` and describe preservation when omitted. Remove `dtype` from `TileDataset.__init__`, its stored state, and its `to_tensor` call.

- [ ] **Step 4: Run affected tensor, tile-dataset, and ML task tests**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/geodata/core/test_model_io.py \
  tests/geodata/datasets/test_tiles.py \
  tests/ml/tasks/test_semantic_segmentation.py -q
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff check \
  src/geosave_engine/geodata/core/base.py \
  src/geosave_engine/geodata/core/array.py \
  src/geosave_engine/geodata/core/raster.py \
  src/geosave_engine/geodata/core/stack.py \
  src/geosave_engine/geodata/datasets/tiles.py \
  tests/geodata/core/test_model_io.py \
  tests/geodata/datasets/test_tiles.py
```

Expected: all selected tests and Ruff checks pass.

- [ ] **Step 5: Commit the tensor seam**

```bash
git add src/geosave_engine/geodata/core/base.py \
  src/geosave_engine/geodata/core/array.py \
  src/geosave_engine/geodata/core/raster.py \
  src/geosave_engine/geodata/core/stack.py \
  src/geosave_engine/geodata/datasets/tiles.py \
  tests/geodata/core/test_model_io.py \
  tests/geodata/datasets/test_tiles.py \
  tests/ml/tasks/test_semantic_segmentation.py
git commit -m "refactor: make tensor conversion explicit"
```

### Task 2: Replace `workflow.spec` with portable `workflow.specs`

**Files:**
- Create: `src/geosave_engine/workflow/specs/__init__.py`
- Create: `src/geosave_engine/workflow/specs/base.py`
- Create: `src/geosave_engine/workflow/specs/model.py`
- Create: `src/geosave_engine/workflow/specs/sources.py`
- Create: `src/geosave_engine/workflow/specs/preprocessing.py`
- Create: `src/geosave_engine/workflow/specs/postprocessing.py`
- Move and rewrite: `tests/workflow/spec/` to `tests/workflow/specs/`
- Modify: `tests/workflow/conftest.py`
- Modify: `tests/cli/core/test_workspace.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml`
- Modify live imports returned by `rg -l 'workflow\.spec' src tests docs --glob '!*.ipynb'`, excluding historical superseded plans/specs under `docs/superpowers/`

**Interfaces:**
- Consumes: Task 1's `to_tensor(dtype="float32")` callable path.
- Produces: `SpecModel`, `Ref`, `OperationSpec`, `PostprocessingSpec`, `RasterRequirement`, and `ModelSpec` from `geosave_engine.workflow.specs`.

- [ ] **Step 1: Write failing specification tests for the settled document**

Add these behaviors under `tests/workflow/specs/test_model.py`:

```python
def test_inference_reuses_explicit_call_declarations(tmp_path):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {},
            "preprocessing": {},
            "inference": {
                "image": {
                    "call": Ref("normalized.gs.to_tensor"),
                    "kwargs": {"dtype": "float32"},
                }
            },
            "postprocessing": {},
        }
    )
    path = spec.save(tmp_path)
    assert ModelSpec.load(path) == spec


def test_postprocessing_rejects_undeclared_semantics():
    with pytest.raises(ValidationError, match="method"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "sources": {},
                "postprocessing": {"method": "segmentation"},
            }
        )


def test_loading_never_imports_inference_calls(tmp_path):
    path = tmp_path / "model_spec.yaml"
    path.write_text(
        "schema_version: 2\nsources: {}\ninference:\n"
        "  image:\n    call: missing_package.to_tensor\n"
        "postprocessing: {}\n"
    )
    assert ModelSpec.load(path).inference["image"].call == "missing_package.to_tensor"
```

Retain existing duplicate-key, invalid-path, primitive-value, reference,
source-requirement, and Python/YAML round-trip tests. Keep template tests in this
task limited to loading and inspecting the inert document; move preprocessing
execution/laziness cases to Task 5. Remove output-legend and executable-
postprocessing expectations.

- [ ] **Step 2: Run the new spec tests and verify they fail against the current package**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/specs/test_model.py -q
```

Expected: collection/import failure because `workflow.specs` and `PostprocessingSpec` do not exist yet, or behavioral failure because non-empty postprocessing is accepted.

- [ ] **Step 3: Implement cohesive declaration owners**

Use strict frozen Pydantic models:

```python
class SpecModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )


class PostprocessingSpec(SpecModel):
    """Reserve postprocessing without assigning execution semantics."""
```

In `preprocessing.py`, keep `Ref` inert and give `OperationSpec` the complete
container-walking API:

```python
class OperationSpec(SpecModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    call: str | Ref
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("call")
    @classmethod
    def validate_call(cls, value: str | Ref) -> str | Ref:
        path = value.path if isinstance(value, Ref) else value
        Ref(path)
        if isinstance(value, str) and "." not in value:
            raise ValueError("An imported call needs a module and callable name")
        return value

    @field_validator("kwargs", mode="before")
    @classmethod
    def validate_kwargs(cls, value: Any) -> Any:
        return cls.validate_value(value)

    @classmethod
    def validate_value(cls, value: Any, active: set[int] | None = None) -> Any:
        if isinstance(value, Ref):
            Ref(value.path)
            return value
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        if not isinstance(value, (dict, list)):
            raise ValueError(
                f"Expected a YAML literal or Ref, got {type(value).__name__}"
            )
        active = set() if active is None else active
        if id(value) in active:
            raise ValueError("Cyclic configuration values are not supported")
        active.add(id(value))
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("Configuration mapping keys must be strings")
                return {
                    key: cls.validate_value(item, active)
                    for key, item in value.items()
                }
            return [cls.validate_value(item, active) for item in value]
        finally:
            active.remove(id(value))

    @classmethod
    def find_references(cls, value: Any) -> tuple[Ref, ...]:
        if isinstance(value, Ref):
            return (value,)
        if isinstance(value, dict):
            return tuple(
                reference
                for item in value.values()
                for reference in cls.find_references(item)
            )
        if isinstance(value, list):
            return tuple(
                reference
                for item in value
                for reference in cls.find_references(item)
            )
        return ()

    @property
    def references(self) -> tuple[Ref, ...]:
        return self.find_references([self.call, self.kwargs])

    @classmethod
    def resolve_value(cls, value: Any, values: Mapping[str, Any]) -> Any:
        if isinstance(value, Ref):
            return value.resolve(values)
        if isinstance(value, dict):
            return {
                key: cls.resolve_value(item, values)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls.resolve_value(item, values) for item in value]
        return value

    def resolve_kwargs(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return self.resolve_value(self.kwargs, values)
```

This preserves fresh argument containers on every run and leaves resolved native
objects untouched. Do not leave `validate_value`, `references`, or `resolve` as
free functions.

In `model.py`, define:

```python
class ModelSpec(SpecModel):
    filename: ClassVar[str] = "model_spec.yaml"
    schema_version: Literal[2]
    sources: dict[str, RasterRequirement]
    preprocessing: dict[Name, OperationSpec] = Field(default_factory=dict)
    inference: dict[Name, OperationSpec] = Field(default_factory=dict)
    postprocessing: PostprocessingSpec = Field(default_factory=PostprocessingSpec)
```

Keep YAML loader/dumper behavior local to `ModelSpec`; move path resolution onto a `ModelSpec` class/static method. Remove `OutputSpec` and the `outputs` field. Move `requirements.py` to `sources.py`, preserve its native attrs validation, and move duplicate checks onto their owning validators instead of a free `unique` function.

- [ ] **Step 4: Update the shipped model YAML and every live import**

The template must end with an explicit inference call and empty postprocessing:

```yaml
inference:
  image:
    call: !ref image.gs.to_tensor
    kwargs:
      dtype: float32
postprocessing: {}
```

Use `rg` to update live Python imports to `geosave_engine.workflow.specs`. Preserve unrelated edits in moved tests and fixtures. Do not add a `workflow.spec` alias.

Update `tests/cli/core/test_workspace.py` to compare the model's input channels
with `spec.sources`, and to assert that `spec.inference["image"]` contains the
explicit `Ref("image.gs.to_tensor")` call and `{"dtype": "float32"}`. Remove
the output-legend assertion because postprocessing now has no semantics.

- [ ] **Step 5: Run the complete specification and template scopes**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/specs \
  tests/cli/core/test_workspace.py -q
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff check \
  src/geosave_engine/workflow/specs \
  tests/workflow/specs \
  tests/workflow/conftest.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised
```

Expected: all selected tests and Ruff checks pass; loading a missing inference import remains inert.

- [ ] **Step 6: Commit the portable specification package**

```bash
git add src/geosave_engine/workflow/specs \
  tests/workflow/specs \
  tests/workflow/conftest.py \
  tests/cli/core/test_workspace.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml
git commit -m "refactor: organize workflow model specs"
```

### Task 3: Parse primitive deployment parameters with Pydantic configs

**Files:**
- Create: `src/geosave_engine/workflow/configs/__init__.py`
- Create: `src/geosave_engine/workflow/configs/base.py`
- Create: `src/geosave_engine/workflow/configs/anchor.py`
- Create: `src/geosave_engine/workflow/configs/source.py`
- Create: `src/geosave_engine/workflow/configs/ingest.py`
- Create: `tests/workflow/configs/test_anchor.py`
- Create: `tests/workflow/configs/test_source.py`
- Create: `tests/workflow/configs/test_ingest.py`

**Interfaces:**
- Consumes: model-independent primitive flow parameters.
- Produces: `CoordinateAnchorConfig.open() -> GeoAnchor`, `GeoJSONAnchorConfig.open() -> GeoAnchor`, discriminated `AnchorConfig`, `QueryConfig.to_query(collection: str) -> StacQuery`, `SourceConfig`, and `IngestConfig.validate_sources(requirements: Mapping[str, object]) -> None`.

- [ ] **Step 1: Write failing tests for discriminated anchors and complete flow input parsing**

```python
def test_ingest_config_dispatches_coordinate_anchor(tmp_path):
    config = IngestConfig.model_validate(
        {
            "sources": {"optical": {"query": {}, "load": {}}},
            "anchor": {
                "kind": "coordinates",
                "latitude": 45,
                "longitude": 12,
                "shape": [4, 6],
                "resolution": 10,
            },
            "output": str(tmp_path / "raw.zarr"),
            "spec": str(tmp_path / "model_spec.yaml"),
        }
    )
    assert isinstance(config.anchor, CoordinateAnchorConfig)
    assert config.anchor.shape == (4, 6)


@pytest.mark.parametrize("kind", [None, "point", "coordinates "])
def test_anchor_kind_is_required_and_exact(kind):
    payload = {"kind": kind, "latitude": 45, "longitude": 12,
               "shape": 4, "resolution": 10}
    with pytest.raises(ValidationError, match="kind"):
        IngestConfig.model_validate(
            {"sources": {}, "anchor": payload,
             "output": "raw.zarr", "spec": "model_spec.yaml"}
        )


def test_unknown_nested_source_fields_fail_before_http(tmp_path):
    with pytest.raises(ValidationError, match="unexpected"):
        IngestConfig.model_validate(
            {
                "sources": {"optical": {"query": {"unexpected": True}}},
                "anchor": {"kind": "coordinates", "latitude": 45,
                           "longitude": 12, "shape": 4, "resolution": 10},
                "output": str(tmp_path / "raw.zarr"),
                "spec": str(tmp_path / "model_spec.yaml"),
            }
        )


def test_source_names_are_checked_without_opening_any_source(tmp_path):
    config = IngestConfig.model_validate(
        {
            "sources": {"extra": {}},
            "anchor": {"kind": "coordinates", "latitude": 45,
                       "longitude": 12, "shape": 4, "resolution": 10},
            "output": str(tmp_path / "raw.zarr"),
            "spec": str(tmp_path / "model_spec.yaml"),
        }
    )
    with pytest.raises(ValueError, match="missing.*optical"):
        config.validate_sources({"optical": object()})
```

Add literal tests for GeoJSON's exactly-one-of shape/resolution rule, coordinate
latitude bounds, non-negative padding, positive shape/resolution, tuple
timespans, remote paths, non-Zarr output, existing output, non-finite query
numbers, invalid WGS84 query bounds, and invalid native load settings.

- [ ] **Step 2: Run the config tests and verify missing-package failures**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/configs -q
```

Expected: import failures because `workflow.configs` does not exist.

- [ ] **Step 3: Implement strict config models and native conversion methods**

Use a strict frozen base:

```python
class ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )
```

In `anchor.py`, define two models selected by `kind`:

```python
class CoordinateAnchorConfig(ConfigModel):
    kind: Literal["coordinates"]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    shape: PositiveInt | tuple[PositiveInt, PositiveInt]
    resolution: Annotated[float, Field(gt=0)]
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None

    def open(self) -> GeoAnchor:
        return GeoAnchor.from_coordinates(
            **self.model_dump(exclude={"kind"}, exclude_none=True)
        )


class GeoJSONAnchorConfig(ConfigModel):
    kind: Literal["geojson"]
    path: Path
    shape: PositiveInt | tuple[PositiveInt, PositiveInt] | None = None
    resolution: Annotated[float, Field(gt=0)] | None = None
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None
    pad: Annotated[float, Field(ge=0)] = 0

    def open(self) -> GeoAnchor:
        geometry = io.read_vector(self.path).footprint
        return GeoAnchor.from_geometry(
            geometry,
            **self.model_dump(exclude={"kind", "path"}, exclude_none=True),
        )


AnchorConfig = Annotated[
    CoordinateAnchorConfig | GeoJSONAnchorConfig,
    Field(discriminator="kind"),
]
```

Add a model validator on `GeoJSONAnchorConfig` for exactly one of `shape` and
`resolution`. Add a `mode="before"` path validator that rejects URI syntax,
requires `.json` or `.geojson`, and returns a `Path`; file reading remains in
`open()` so config parsing performs no geospatial I/O.

In `source.py`, define every supported primitive query field explicitly:

```python
class SortConfig(ConfigModel):
    field: str
    direction: Literal["asc", "desc"]


class QueryConfig(ConfigModel):
    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, JsonValue] | None = None
    datetime: str | tuple[str, str] | None = None
    filter: dict[str, JsonValue] | None = None
    max_items: PositiveInt | None = None
    limit: PositiveInt | None = None
    ids: tuple[str, ...] | None = None
    sortby: tuple[SortConfig, ...] | None = None

    def to_query(self, collection: str) -> StacQuery:
        values = self.model_dump(exclude_none=True)
        if "sortby" in values:
            values["sortby"] = [dict(item) for item in values["sortby"]]
        return StacQuery(collections=[collection], **values)

    @model_validator(mode="after")
    def validate_native_query(self) -> Self:
        self.to_query("validation")
        return self


class SourceConfig(ConfigModel):
    query: QueryConfig = Field(default_factory=QueryConfig)
    load: StacSourceConfig = Field(default_factory=StacSourceConfig)

    @field_validator("load", mode="before")
    @classmethod
    def validate_primitive_load(cls, value: Any) -> Any:
        if isinstance(value, StacSourceConfig):
            return value
        return TypeAdapter(
            dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False)
        ).validate_python(value, strict=True)
```

Do not accept arbitrary nested query/load keys or non-primitive deployment
values. The outer validator prevents fields such as `patch_url` from receiving
a Python callable through flow parameters, and rejects non-finite values before
the native load model is built.

In `ingest.py`, define `sources: dict[Text, SourceConfig]`, `anchor: AnchorConfig`,
`output: Path`, and `spec: Path`. `mode="before"` validators inspect the original
text before `Path` coercion: both paths reject URI syntax, output requires a
`.zarr` suffix and must not exist, while spec accepts a local `.yaml`/`.yml`
file or artifact directory. Add this exact cross-checking interface:

```python
def validate_sources(self, requirements: Mapping[str, object]) -> None:
    if missing := requirements.keys() - self.sources.keys():
        raise ValueError(f"Source bindings are missing: {sorted(missing)}")
    if extra := self.sources.keys() - requirements.keys():
        raise ValueError(f"Unknown source bindings: {sorted(extra)}")
    if not requirements:
        raise ValueError("At least one model source is required")
```

- [ ] **Step 4: Run config behavior, import-isolation, and Ruff checks**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/configs -q
PYTHONPATH=src UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run python -c \
  "import sys; import geosave_engine.workflow.configs; assert 'prefect' not in sys.modules"
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff check \
  src/geosave_engine/workflow/configs tests/workflow/configs
```

Expected: all config tests pass and importing configs loads no Prefect module.

- [ ] **Step 5: Commit flow configuration parsing**

```bash
git add src/geosave_engine/workflow/configs tests/workflow/configs
git commit -m "feat: validate workflow deployment inputs"
```

### Task 4: Add bounded raster loading and completed-stack persistence tasks

**Files:**
- Create: `src/geosave_engine/workflow/tasks/__init__.py`
- Create: `src/geosave_engine/workflow/tasks/load.py`
- Create: `src/geosave_engine/workflow/tasks/save.py`
- Create: `tests/workflow/tasks/test_load.py`
- Create: `tests/workflow/tasks/test_save.py`
- Modify shared fixtures in: `tests/workflow/conftest.py`

**Interfaces:**
- Consumes: `AnchorConfig`, `SourceConfig`, `RasterRequirement`, native `StacClient`, `StacSource`, `stack`, and `geodata.utils.io.zarr.write`.
- Produces: `load_raster(name: str, source: SourceConfig, anchor: AnchorConfig, requirement: RasterRequirement) -> xr.Dataset` and `save_stack(rasters: dict[str, xr.Dataset], output: str | Path) -> str`, both as Prefect tasks with caching/result persistence disabled.

- [ ] **Step 1: Write failing real-behavior tests for each task**

```python
def test_load_raster_reads_and_validates_one_local_source(stac_server):
    url, requests, expected_anchor = stac_server
    result = load_raster.fn(
        "optical",
        SourceConfig(),
        CoordinateAnchorConfig(
            kind="coordinates", latitude=45, longitude=12,
            shape=4, resolution=10, timespan="2025-01",
        ),
        RasterRequirement(
            variables=("red", "nir"), collection="optical",
            endpoints=(url,),
            require_crs=True,
        ),
    )
    assert list(result.data_vars) == ["red", "nir"]
    assert result.odc.geobox == expected_anchor.geobox
    assert requests == [["optical"]]


def test_native_results_are_not_cached_or_persisted():
    assert load_raster.cache_policy is None
    assert load_raster.persist_result is False
    assert save_stack.cache_policy is None
    assert save_stack.persist_result is False


def test_save_stack_rechecks_destination_before_publish(scene, tmp_path, monkeypatch):
    output = tmp_path / "raw.zarr"
    original = io.zarr.write

    def create_competing_output(*args, **kwargs):
        result = original(*args, **kwargs)
        output.mkdir()
        return result

    monkeypatch.setattr(io.zarr, "write", create_competing_output)
    with pytest.raises(FileExistsError, match="already exists"):
        save_stack.fn({"optical": scene}, output)
```

Port endpoint priority, missing-collection, authentication, malformed-document, empty-search, lazy-asset, band-selection, staged-write cleanup, and retry tests from the current runtime/ingestion/I/O test files. Rename all test language and symbols away from the removed term.

Move the real local HTTP catalogue context from `test_prefect.py` into a
`stac_server` fixture in `tests/workflow/conftest.py`. It must yield the URL,
recorded collection requests, and expected anchor while retaining real STAC
document parsing and real local GeoTIFF assets.

- [ ] **Step 2: Run task tests and verify missing task failures**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/tasks/test_load.py \
  tests/workflow/tasks/test_save.py -q
```

Expected: import failures because the task modules do not exist.

- [ ] **Step 3: Implement a cohesive loader and short task wrapper**

In `tasks/load.py`, keep endpoint probing and source construction on one loader class:

```python
class RasterLoader:
    def __init__(self, requirement: RasterRequirement) -> None:
        self.requirement = RasterRequirement.model_validate(requirement)
        if self.requirement.collection is None or self.requirement.endpoints is None:
            raise ValueError("Raster source needs collection and endpoints")

    @staticmethod
    def endpoint_unavailable(error: Exception) -> bool:
        if isinstance(error, APIError):
            status = getattr(error, "status_code", None)
            if status is not None:
                return status == 404 or status >= 500
            cause = error.__cause__ or error.__context__
            seen = {id(error)}
            while cause is not None and id(cause) not in seen:
                if isinstance(cause, RequestException) and not isinstance(
                    cause, ValueError
                ):
                    return True
                seen.add(id(cause))
                cause = cause.__cause__ or cause.__context__
            return False
        return isinstance(error, RequestException) and not isinstance(
            error, ValueError
        )

    def open_client(self) -> StacClient:
        failures = []
        last_error: Exception | None = None
        for endpoint in self.requirement.endpoints:
            url = str(endpoint)
            try:
                client = Client.open(url)
                try:
                    collection = client.get_collection(self.requirement.collection)
                except KeyError as error:
                    expected = (
                        f"Collection {self.requirement.collection} not found on catalog",
                    )
                    if error.args != expected:
                        raise
                    collection = None
            except (RequestException, APIError) as error:
                if not self.endpoint_unavailable(error):
                    raise
                last_error = error
            else:
                if collection is not None and collection.id == self.requirement.collection:
                    return StacClient(client)
                last_error = LookupError(
                    f"Collection {self.requirement.collection!r} not found"
                )
            status = getattr(last_error, "status_code", None)
            cause = (
                f"HTTP {status}: {last_error}"
                if status is not None
                else str(last_error)
            )
            failures.append(f"{url}: {cause}")
        raise ConnectionError(
            "STAC endpoints unavailable: " + "; ".join(failures)
        ) from last_error

    def load(self, config: SourceConfig, anchor: AnchorConfig) -> xr.Dataset:
        client = self.open_client()
        source = StacSource(client, collection=self.requirement.collection)
        source.query = config.query.to_query(self.requirement.collection)
        settings = config.load.model_dump()
        if self.requirement.variables is not None:
            settings["bands"] = self.requirement.variables
        source.config = StacSourceConfig.model_validate(settings)
        raster = source.load(anchor.open())
        return self.requirement.select_raster(raster)


@task(cache_policy=None, persist_result=False)
def load_raster(
    name: str,
    source: SourceConfig,
    anchor: AnchorConfig,
    requirement: RasterRequirement,
) -> xr.Dataset:
    try:
        return RasterLoader(requirement).load(source, anchor)
    except Exception as error:
        error.add_note(f"While loading source {name!r}")
        raise
```

Preserve this exact failover shield: only transport, 404/5xx, and missing collection continue.

- [ ] **Step 4: Implement atomic completed persistence in one task**

```python
@task(cache_policy=None, persist_result=False)
def save_stack(rasters: dict[str, xr.Dataset], output: str | Path) -> str:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = stack(rasters)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        io.zarr.write(tree, staged, compute=True, overwrite=False)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        staged.rename(destination)
    return str(destination)
```

Keep path syntax validation in `IngestConfig`; retain the second existence check here for the publication race.

- [ ] **Step 5: Run task, local STAC, persistence, and Ruff checks**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/tasks/test_load.py \
  tests/workflow/tasks/test_save.py -q
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff check \
  src/geosave_engine/workflow/tasks/load.py \
  src/geosave_engine/workflow/tasks/save.py \
  tests/workflow/tasks/test_load.py \
  tests/workflow/tasks/test_save.py
```

Expected: all task tests and Ruff checks pass. If Zarr stalls because restricted local sockets prevent its async writer, rerun this exact scope with the required local socket permission before classifying it as a product failure.

- [ ] **Step 6: Commit load and save tasks**

```bash
git add src/geosave_engine/workflow/tasks \
  tests/workflow/tasks/test_load.py \
  tests/workflow/tasks/test_save.py
git commit -m "feat: add bounded workflow IO tasks"
```

### Task 5: Execute preprocessing calls through one short task

**Files:**
- Create: `src/geosave_engine/workflow/tasks/preprocess.py`
- Create: `tests/workflow/tasks/test_preprocess.py`
- Migrate behavior from: `tests/workflow/test_processing.py`
- Modify: `tests/workflow/specs/test_examples.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md`

**Interfaces:**
- Consumes: `ModelSpec.preprocessing`, `OperationSpec`, `Ref`, and a mapping of supplied native values.
- Produces: Prefect task `preprocess(values: Mapping[str, Any], spec: ModelSpec) -> dict[str, Any]` and its cohesive `Preprocessor` executor.

- [ ] **Step 1: Write failing tests for preprocessing-only execution**

```python
def test_preprocess_keeps_native_lazy_values(raw):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {},
            "preprocessing": {
                "selected": {
                    "call": Ref("optical.__getitem__"),
                    "kwargs": {"key": ["nir", "red"]},
                }
            },
            "inference": {
                "image": {"call": "missing_package.to_tensor"}
            },
            "postprocessing": {},
        }
    )
    result = preprocess.fn(raw, spec)
    assert list(result["selected"].data_vars) == ["nir", "red"]
    assert result["selected"].nir.chunks is not None


def test_preprocess_validates_all_inputs_before_the_first_call():
    called = []
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "sources": {},
            "preprocessing": {
                "first": {"call": Ref("record"), "kwargs": {}},
                "second": {"call": Ref("missing"), "kwargs": {}},
            },
            "postprocessing": {},
        }
    )
    with pytest.raises(ValueError, match="missing"):
        preprocess.fn({"record": lambda: called.append(True)}, spec)
    assert called == []
```

Port tests for supplied bound methods, imported functions, nested kwargs, literal strings, rebinding, aliases, fresh containers, `None` returns, invalid signatures, noncallable targets, source validation, repeated calls, and inactive inference imports. Delete executable-postprocessing and output-legend cases.

Update the shipped example script to load `ModelSpec` from `workflow.specs` and
run `Preprocessor(ModelSpec.load(path)).run(...)`. Update its README to describe
sample-ready lazy xarray preprocessing, the validated-but-not-yet-executed
inference call, and empty postprocessing. Remove all `Processor`, output legend,
and executable postprocessing guidance. Keep the subprocess example test to
prove the generated script still runs.

- [ ] **Step 2: Run preprocessing tests and verify the missing implementation**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/tasks/test_preprocess.py -q
```

Expected: import failure because `workflow.tasks.preprocess` does not exist.

- [ ] **Step 3: Implement the executor and keep the Prefect task short**

```python
class Preprocessor:
    def __init__(self, spec: ModelSpec) -> None:
        configuration = ModelSpec.model_validate(spec)
        self.operations: list[
            tuple[str, Callable[..., Any] | Ref, OperationSpec]
        ] = []
        required: set[str] = set()
        assigned: set[str] = set()
        for name, operation in configuration.preprocessing.items():
            for reference in operation.references:
                if reference.root not in assigned:
                    required.add(reference.root)
            assigned.add(name)
            target = operation.call
            if isinstance(target, str):
                target = self.import_call(target)
                self.bind(target, operation.kwargs)
            self.operations.append((name, target, operation))
        self.required = frozenset(required)
        self.sources = {
            name: requirement
            for name, requirement in configuration.sources.items()
            if name in self.required
        }

    @staticmethod
    def import_call(path: str) -> Callable[..., Any]:
        module, name = path.rsplit(".", 1)
        target = getattr(import_module(module), name)
        if not callable(target):
            raise TypeError(f"Expected a callable, got {type(target).__name__}")
        return target

    @staticmethod
    def bind(target: Any, kwargs: Mapping[str, Any]) -> None:
        if not callable(target):
            raise TypeError(f"Expected a callable, got {type(target).__name__}")
        try:
            parameters = signature(target)
        except (TypeError, ValueError):
            return
        parameters.bind(**kwargs)

    def validate_inputs(self, values: dict[str, Any]) -> None:
        if missing := self.required - values.keys():
            raise ValueError(f"Missing preprocessing values: {sorted(missing)}")
        for name, requirement in self.sources.items():
            try:
                values[name] = requirement.select_raster(values[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Source {name!r}: {error}") from error

    def invoke(
        self,
        name: str,
        target: Callable[..., Any] | Ref,
        operation: OperationSpec,
        values: Mapping[str, Any],
    ) -> Any:
        try:
            function = target.resolve(values) if isinstance(target, Ref) else target
            kwargs = operation.resolve_kwargs(values)
            self.bind(function, kwargs)
            return function(**kwargs)
        except Exception as error:
            error.add_note(f"While executing preprocessing.{name}")
            raise

    def run(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(supplied)
        self.validate_inputs(values)
        for name, target, operation in self.operations:
            values[name] = self.invoke(name, target, operation, values)
        return values


@task(cache_policy=None, persist_result=False)
def preprocess(
    values: Mapping[str, Any], spec: ModelSpec
) -> dict[str, Any]:
    return Preprocessor(spec).run(values)
```

Import `Callable` and `Mapping` from `collections.abc`, `import_module` from
`importlib`, and `signature` from `inspect`. Keep callable importing, signature
binding, required-root collection, source selection, and exception notes on
methods of `Preprocessor` or `OperationSpec`; do not introduce free private
helper functions. Compile only `preprocessing`. Do not import, validate
signatures for, or execute `inference` calls.

- [ ] **Step 4: Run preprocessing, specification, import-isolation, and Ruff checks**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/tasks/test_preprocess.py \
  tests/workflow/specs/test_examples.py -q
PYTHONPATH=src UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run python -c \
  "import sys; import geosave_engine.workflow.specs; assert 'prefect' not in sys.modules"
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff check \
  src/geosave_engine/workflow/tasks/preprocess.py \
  tests/workflow/tasks/test_preprocess.py \
  tests/workflow/specs/test_examples.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py
```

Expected: all selected tests and Ruff checks pass, including the missing inference import case.

- [ ] **Step 5: Commit preprocessing execution**

```bash
git add src/geosave_engine/workflow/tasks/preprocess.py \
  tests/workflow/tasks/test_preprocess.py \
  tests/workflow/specs/test_examples.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py \
  src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md
git commit -m "feat: add explicit preprocessing task"
```

### Task 6: Install the deployable ingestion monoflow and remove obsolete paths

**Files:**
- Create: `src/geosave_engine/workflow/flows/__init__.py`
- Create: `src/geosave_engine/workflow/flows/ingest.py`
- Create: `tests/workflow/flows/test_ingest.py`
- Modify: `src/geosave_engine/workflow/__init__.py`
- Modify live template/docs/import files returned by the removal searches below
- Delete: `src/geosave_engine/workflow/flows.py`
- Delete: `src/geosave_engine/workflow/ingestion.py`
- Delete: `src/geosave_engine/workflow/io.py`
- Delete: `src/geosave_engine/workflow/processing.py`
- Delete: `src/geosave_engine/workflow/runtime.py`
- Delete: `src/geosave_engine/workflow/README.md`
- Delete: `src/geosave_engine/workflow/spec/`
- Delete migrated legacy tests: `tests/workflow/test_ingestion.py`, `tests/workflow/test_io.py`, `tests/workflow/test_processing.py`, `tests/workflow/test_runtime.py`, `tests/workflow/test_prefect.py`

**Interfaces:**
- Consumes: `IngestConfig`, `ModelSpec`, `load_raster`, and `save_stack`.
- Produces: deployable `geosave_engine.workflow.flows.ingest(sources, anchor, *, output, spec) -> str`.

- [ ] **Step 1: Write the failing deployment-oriented flow tests**

```python
@pytest.mark.slow
def test_ingest_validates_primitives_and_writes_completed_stack(
    tmp_path, stac_server, prefect_server
):
    url, requests, expected_anchor = stac_server
    model_spec = ModelSpec(
        schema_version=2,
        sources={
            "optical": RasterRequirement(
                variables=("red", "nir"),
                collection="optical",
                endpoints=(url,),
                require_crs=True,
            )
        },
    ).save(tmp_path / "model_spec.yaml")
    output = tmp_path / "raw.zarr"

    result = ingest(
        sources={"optical": {"query": {}, "load": {}}},
        anchor={
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": 4,
            "resolution": 10,
            "timespan": "2025-01",
        },
        output=str(output),
        spec=str(model_spec),
    )

    assert result == str(output)
    assert requests == [["optical"]]
    with io.read_stack(output, chunks="auto") as tree:
        assert list(tree.gs.rasters) == ["optical"]
        assert tree.gs.geobox == expected_anchor.geobox


def test_ingest_rejects_source_names_before_task_submission(tmp_path, monkeypatch):
    submitted = []
    monkeypatch.setattr(load_raster, "submit", lambda *a, **k: submitted.append(a))
    model_spec = ModelSpec(
        schema_version=2,
        sources={"optical": RasterRequirement(variables=("red", "nir"))},
    ).save(tmp_path / "model_spec.yaml")
    with pytest.raises(ValueError, match="Unknown source bindings"):
        ingest.fn(
            sources={"extra": {}},
            anchor={"kind": "coordinates", "latitude": 45, "longitude": 12,
                    "shape": 4, "resolution": 10},
            output=str(tmp_path / "raw.zarr"),
            spec=str(model_spec),
        )
    assert submitted == []
```

Retain real Prefect failure propagation and exact model-source binding coverage. Flow tests assert primitive parameters and completed paths, not private helper calls.

- [ ] **Step 2: Run the flow tests and verify the package/module conflict**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow/flows/test_ingest.py -q
```

Expected: import failure because `workflow.flows` is still a module rather than the designed package.

- [ ] **Step 3: Implement the monoflow**

```python
@flow(
    name="ingest",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    persist_result=False,
)
def ingest(
    sources: dict[str, dict[str, JsonValue]],
    anchor: dict[str, JsonValue],
    *,
    output: str,
    spec: str,
) -> str:
    config = IngestConfig.model_validate(
        {"sources": sources, "anchor": anchor, "output": output, "spec": spec}
    )
    model = ModelSpec.load(config.spec)
    config.validate_sources(model.sources)
    pending = {
        name: load_raster.submit(
            name, config.sources[name], config.anchor, requirement
        )
        for name, requirement in model.sources.items()
    }
    return save_stack.submit(pending, config.output).result()
```

Put source-name cross-validation on `IngestConfig.validate_sources` so the flow stays orchestration-only. Export `ingest` from `workflow.flows.__init__`; keep the root `workflow.__init__` free of convenience exports.

- [ ] **Step 4: Remove the obsolete implementation and migrate every live caller**

Before deletion, run:

```bash
rg -n "workflow\.(spec|runtime|ingestion|io|processing)|from geosave_engine\.workflow import|\bacquire\b" \
  src tests --glob '!*.ipynb'
```

Confirm retained behavior tests already live in the new mirrored package scopes,
then delete the old source/test modules and both workflow READMEs. Update any
remaining live callers found by the search, but do not change historical
`docs/superpowers/` records except this active plan/spec.

- [ ] **Step 5: Run the complete workflow and affected geodata scopes**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest \
  tests/workflow \
  tests/geodata/core/test_model_io.py \
  tests/geodata/datasets/test_tiles.py \
  tests/cli/core/test_workspace.py -q
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff check \
  src/geosave_engine/workflow \
  tests/workflow \
  src/geosave_engine/geodata/core/base.py \
  src/geosave_engine/geodata/core/array.py \
  src/geosave_engine/geodata/core/raster.py \
  src/geosave_engine/geodata/core/stack.py \
  src/geosave_engine/geodata/datasets/tiles.py
```

Expected: all selected tests and Ruff checks pass.

- [ ] **Step 6: Prove the obsolete live surface is gone**

Run:

```bash
test ! -e src/geosave_engine/workflow/runtime.py
test ! -e src/geosave_engine/workflow/ingestion.py
test ! -e src/geosave_engine/workflow/io.py
test ! -e src/geosave_engine/workflow/processing.py
test ! -e src/geosave_engine/workflow/README.md
! rg -n "workflow\.(spec|runtime|ingestion|io|processing)|from geosave_engine\.workflow import|\bacquire\b" \
  src/geosave_engine/workflow tests/workflow --glob '!*.ipynb'
PYTHONPATH=src UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run python -c \
  "import sys; import geosave_engine.workflow.specs; import geosave_engine.workflow.configs; assert 'prefect' not in sys.modules"
```

Expected: every command exits zero and the search prints nothing.

- [ ] **Step 7: Run repository-wide verification**

Run:

```bash
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run pytest -q
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff check . --exclude '*.ipynb'
UV_CACHE_DIR=/tmp/geosave-workflow-redesign-cache uv run ruff format --check . --exclude '*.ipynb'
git diff --check
```

Expected: the full default test suite, Ruff lint, Ruff formatting, and whitespace checks pass. Report any unrelated pre-existing failure by exact test or file name rather than omitting it.

- [ ] **Step 8: Commit the flow migration and deletion**

```bash
git add src/geosave_engine/workflow \
  tests/workflow \
  src/geosave_engine/templates \
  tests/cli/core/test_workspace.py
git commit -m "refactor: rebuild workflow package hierarchy"
```

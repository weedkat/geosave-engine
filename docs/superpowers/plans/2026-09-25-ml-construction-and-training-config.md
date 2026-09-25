# ML Construction and Training Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic ML construction registries with a model-stage-only registry, give the segmentation LightningModule native PyTorch training configuration and a direct tuple batch contract, and make source/output workflow contracts and generated templates agree with that training interface.

**Architecture:** `ml.registry` constructs only named `ModelChain` stages. `SemanticSegmentationTask` constructs one native criterion, optimizer, and optional scheduler from import paths and consumes model-ready batches. `RasterRequirement` selects and validates every raster through one lazy path while `ModelSpec` owns STAC resolution and output legends. Template-owned LightningDataModules bridge prepared datasets to the task contract; Lightning and PyTorch retain their native orchestration responsibilities.

**Tech Stack:** Python 3.12, PyTorch, Lightning/LightningCLI, xarray/Dask, Pydantic, PySTAC Client, LitData, PyYAML, pytest, Ruff.

**Spec:** [ML construction and training configuration](../specs/2026-09-25-ml-construction-and-training-config-design.md)

## Global Constraints

- Preserve unrelated working-tree edits. Several workflow and task files are already modified; read and patch their current versions instead of restoring `HEAD` versions.
- This is an Alpha breaking change. Do not add aliases for `BuildSpec`, `build_model`, `stages`, `loss`, `scheduler`, old dictionary batches, optimizer strategies, or workflow-source URL fields.
- Ignore notebooks. Audit live Python, tests, Markdown, and YAML outside `*.ipynb`.
- Keep `ml.loss.ProbOhemCrossEntropy2d`; delete only the generic loss registry.
- Keep source selection lazy. No test or implementation may compute pixels merely to select or validate variables/channels.
- Runtime source settings contain only `query` and `load`; `ModelSpec.sources` owns STAC collection/endpoints.
- Use one optimizer. Do not introduce a general training-config module or multi-optimizer schema.
- Commit only task-related paths at each checkpoint; never stage the entire dirty worktree.

## Review Focus

- Exact top-level model-stage grouping must neither guess parameter names nor omit unlisted trainable parameters.
- Criterion and scheduler construction must preserve native PyTorch/Lightning errors and Lightning scheduler metadata.
- The segmentation task must contain no old batch-shape fallback, dataset key selection, or augmentation path.
- Named and positional raster selection must share validation and retain stored order, object-backed Dask arrays, CRS, and attrs.
- Endpoint fallback must be narrow: availability or missing collection only, never authentication, malformed STAC, empty search, or asset-read failure.
- Output legends must attach to the actual categorical variable after postprocessing and survive native persistence paths.
- Template YAML, template Python modules, and generated workspaces must be executable and cross-document counts must agree.

---

## Task 1: Reduce the registry to model stages

**Files:**

- Modify: `src/geosave_engine/ml/registry/model.py`
- Modify: `src/geosave_engine/ml/registry/__init__.py`
- Modify: `src/geosave_engine/ml/models/contract/chain.py`
- Modify: `src/geosave_engine/ml/models/contract/published.py`
- Modify: `src/geosave_engine/ml/huggingface.py`
- Modify: `src/geosave_engine/ml/tasks/semantic_segmentation.py`
- Modify: `src/geosave_engine/ml/models/README.md`
- Modify: `src/geosave_engine/ml/models/head/dense.py`
- Replace: `tests/ml/registry/test_builders.py` with `tests/ml/registry/test_model.py`
- Modify: `tests/ml/models/contract/conftest.py`
- Modify: `tests/ml/models/contract/test_automodel.py`
- Modify: `tests/ml/models/contract/test_hub.py`
- Modify: `tests/ml/models/encoder/test_model_context.py`

- [ ] Split the model-stage coverage out of `test_builders.py`. Add tests that import `StageSpec`, construct `ModelChain(stages=stage_specs)` for registered names and class paths, preserve published values/default snapshots, reject malformed selectors, and confirm that `build_model` is no longer exported. Leave the old training-builder tests in place until Task 2 replaces them.

```python
stages: dict[str, StageSpec] = {
    "encoder": {"name": "test"},
    "head": {"class_path": f"{__name__}.Head"},
}
model = ModelChain(stages=json.loads(json.dumps(stages)))
assert model(torch.tensor(3.0)).item() == 8.0

with pytest.raises(ImportError):
    exec("from geosave_engine.ml.registry import build_model")
```

- [ ] Run the new tests and record the expected initial failure from missing `StageSpec` and the still-present `build_model` export.

```bash
uv run pytest tests/ml/registry/test_model.py tests/ml/models/contract -q
```

- [ ] Move the strict selector/import resolver and renamed typed dictionary directly into `registry/model.py`. Keep the existing model factory registry, published-value wiring, and captured constructor defaults.

```python
class StageSpec(TypedDict, total=False):
    name: str
    class_path: str
    init_args: dict[str, Any]


def _resolve_stage(
    spec: StageSpec,
    factories: Mapping[str, ModelFactory],
) -> ModelFactory:
    unknown = set(spec) - {"name", "class_path", "init_args"}
    if unknown:
        raise ValueError(f"Unknown stage fields: {sorted(unknown)}; use init_args")
    if ("name" in spec) == ("class_path" in spec):
        raise ValueError("Specify exactly one of name or class_path")
    if "name" in spec:
        name = spec["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty registered factory name")
        matched = {name.casefold(): factory for name, factory in factories.items()}
        if name.casefold() not in matched:
            raise ValueError(f"Unknown model stage {name!r}; available: {list(factories)}")
        return matched[name.casefold()]
    path = spec["class_path"]
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"class_path must include a module and class: {path!r}")
    module, _, attribute = path.rpartition(".")
    stage = getattr(import_module(module), attribute)
    if not isinstance(stage, type) or not issubclass(stage, nn.Module):
        raise TypeError(f"{path!r} must name an nn.Module subclass")
    return stage
```

- [ ] Delete `build_model`; update all live callers to `ModelChain(stages=stage_specs)`, all annotations to `StageSpec`, and all user-facing text to say construction specifications rather than `build_model()`. In the task, change only the internal model construction in this task: retain its old public training fields until Task 2.
- [ ] Export `StageSpec`, `register_model`, and `list_models` alongside the temporary training builders that the task still uses. `build_stages` remains an internal import used by `ModelChain`. Task 2 removes the remaining generic exports/files in the same commit that replaces their consumers.
- [ ] Run the model construction, hub, encoder-context, and registry tests.

```bash
uv run pytest tests/ml/registry/test_model.py tests/ml/models/contract tests/ml/models/encoder/test_model_context.py -q
uv run ruff check src/geosave_engine/ml/registry src/geosave_engine/ml/models/contract src/geosave_engine/ml/huggingface.py tests/ml/registry tests/ml/models/contract
```

- [ ] Commit only the registry/model-chain migration.

```bash
git add src/geosave_engine/ml/registry src/geosave_engine/ml/models/contract src/geosave_engine/ml/huggingface.py src/geosave_engine/ml/tasks/semantic_segmentation.py src/geosave_engine/ml/models/README.md src/geosave_engine/ml/models/head/dense.py tests/ml/registry tests/ml/models/contract tests/ml/models/encoder/test_model_context.py
git commit -m "refactor: keep registry focused on model stages"
```

## Task 2: Give the segmentation task native training configuration

**Files:**

- Modify: `src/geosave_engine/ml/tasks/semantic_segmentation.py`
- Modify: `src/geosave_engine/ml/cli.py`
- Modify: `src/geosave_engine/templates/common/main.py`
- Delete: `src/geosave_engine/ml/registry/base.py`
- Delete: `src/geosave_engine/ml/registry/loss.py`
- Delete: `src/geosave_engine/ml/registry/optimizer.py`
- Delete: `src/geosave_engine/ml/registry/scheduler.py`
- Delete: `src/geosave_engine/ml/optimizer/__init__.py`
- Delete: `src/geosave_engine/ml/utils/torch_params.py`
- Modify or delete if empty: `src/geosave_engine/ml/utils/__init__.py`
- Modify: `tests/ml/tasks/test_semantic_segmentation.py`
- Modify: `tests/ml/callbacks/test_task_callbacks.py`

- [ ] Rewrite the task fixture/tests around explicit `in_channels`, `num_classes`, `model_chain`, `criterion`, `optimizer`, and `lr_scheduler`. Cover the default criterion/optimizer, criterion `ignore_index` override, exact stage groups, a default group for unlisted stages, unknown groups, scheduler metadata, plateau monitoring, LightningCLI parsing, and checkpoint hyperparameter restoration.

```python
task = SemanticSegmentationTask(
    in_channels=2,
    num_classes=2,
    model_chain=stages,
    criterion={"class_path": "torch.nn.CrossEntropyLoss"},
    optimizer={
        "class_path": "torch.optim.AdamW",
        "init_args": {"lr": 1e-3, "weight_decay": 1e-2},
        "groups": {"model": {"lr": 1e-4, "weight_decay": 0.0}},
    },
    lr_scheduler={
        "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "init_args": {"patience": 2},
        "monitor": "val_loss",
        "interval": "epoch",
    },
)
configured = task.configure_optimizers()
assert configured["lr_scheduler"]["monitor"] == "val_loss"
```

- [ ] Run the task tests and record failures caused by the old names, inferred maps, and registry builders.

```bash
uv run pytest tests/ml/tasks/test_semantic_segmentation.py tests/ml/callbacks/test_task_callbacks.py -q
```

- [ ] Define task-local configuration types with only the fields this task owns. Resolve `class_path` with `import_module`; require a subclass of the expected PyTorch base without wrapping constructor errors.

```python
class ModuleSpec(TypedDict, total=False):
    class_path: Required[str]
    init_args: dict[str, Any]


class OptimizerSpec(ModuleSpec, total=False):
    groups: dict[str, dict[str, Any]]


class LRSchedulerSpec(ModuleSpec, total=False):
    interval: Literal["step", "epoch"]
    frequency: int
    monitor: str
    strict: bool
    name: str
```

- [ ] Change the constructor signature and saved hyperparameters. Defaults are `CrossEntropyLoss` with the task's `ignore_index`, `AdamW(lr=1e-3)`, and no scheduler. Explicit criterion `init_args.ignore_index` wins.
- [ ] In `configure_model`, copy `model_chain`, inject authoritative `in_channels`/`input_size` into the first stage and `num_classes` into the last stage, then call `ModelChain(stages=stage_specs)`.
- [ ] Build optimizer groups from `self.model._modules` exact stage names. Filter `requires_grad`; put explicitly grouped stage parameters first and all remaining trainable parameters in one default group. Fail on unknown group keys before calling the optimizer.

```python
named = dict(self.model.named_children())
unknown = groups.keys() - named.keys()
if unknown:
    raise ValueError(f"Unknown model-chain optimizer groups: {sorted(unknown)}")

selected = {id(parameter) for stage in groups for parameter in named[stage].parameters()}
param_groups = [
    {"params": [p for p in named[stage].parameters() if p.requires_grad], **options}
    for stage, options in groups.items()
]
remaining = [p for p in self.model.parameters() if p.requires_grad and id(p) not in selected]
if remaining:
    param_groups.append({"params": remaining})
```

- [ ] Construct an optional scheduler from the optimizer and return Lightning's native scheduler mapping with metadata fields copied verbatim.
- [ ] Set `auto_configure_optimizers=False` in `templates/common/main.py`. Add a CLI regression assertion by monkeypatching `GeosaveCLI` and executing the template entry point with `runpy.run_path`.
- [ ] Delete the generic registry base/loss/optimizer/scheduler files, the optimizer strategy package, and parameter-name utilities. Export only `StageSpec`, `register_model`, and `list_models`; remove stale imports and update callback task construction.
- [ ] Run the focused tests and Ruff.

```bash
uv run pytest tests/ml/tasks/test_semantic_segmentation.py tests/ml/callbacks/test_task_callbacks.py -q
uv run ruff check src/geosave_engine/ml/tasks/semantic_segmentation.py src/geosave_engine/ml/cli.py src/geosave_engine/templates/common/main.py tests/ml/tasks tests/ml/callbacks
```

- [ ] Commit the native task construction change.

```bash
git add src/geosave_engine/ml/tasks/semantic_segmentation.py src/geosave_engine/ml/cli.py src/geosave_engine/templates/common/main.py src/geosave_engine/ml/registry src/geosave_engine/ml/optimizer src/geosave_engine/ml/utils tests/ml/registry tests/ml/tasks/test_semantic_segmentation.py tests/ml/callbacks/test_task_callbacks.py
git commit -m "refactor: let segmentation task own training setup"
```

## Task 3: Adopt one model-ready tuple batch contract

**Files:**

- Modify: `src/geosave_engine/ml/tasks/semantic_segmentation.py`
- Modify: `src/geosave_engine/geodata/datasets/tiles.py`
- Modify: `tests/ml/tasks/test_semantic_segmentation.py`
- Modify: `tests/geodata/datasets/test_tiles.py`
- Modify: `tests/ml/models/encoder/test_model_context.py`

- [ ] Add tests for supervised `({"image": image}, target)`, contextual `({"image": image, "offset": offset}, target)`, prediction `({"image": image, "offset": offset}, index)`, and ordinary Python errors from incompatible batch values. Assert prediction returns `(logits, index)`.

```python
loss = task.training_step(({"image": image, "offset": offset}, target), 0)
logits, returned = task.predict_step(({"image": image, "offset": offset}, index), 0)
assert returned is index
torch.testing.assert_close(logits, image + offset)
```

- [ ] Update tile tests to unpack `(model_inputs, index)`, including stacks and model context merged into the inner dictionary.

```python
inputs, index = samples[3]
assert index == 3
assert set(inputs) == {"image", "temporal_coords"}
```

- [ ] Run these tests and record failures from dictionary indexing and old prediction output.

```bash
uv run pytest tests/ml/tasks/test_semantic_segmentation.py tests/geodata/datasets/test_tiles.py tests/ml/models/encoder/test_model_context.py -q
```

- [ ] Make `forward` accept only named chain inputs (`def forward(self, **model_inputs)`), and unpack every train/validation/test batch as `model_inputs, target = batch`. Squeeze only a singleton target channel; do not inspect keys or support a bare tensor alternative.
- [ ] Delete `image_key`, `label_key`, `class_map`, `band_map`, `augmentations`, `_extract_model_context`, and `ImageAugmenter` from the task. Construct metrics from `num_classes` without semantic labels.
- [ ] Change `TileDataset` to `Dataset[tuple[dict[str, Any], int]]`. Build `model_inputs = {"image": tile.gs.to_tensor(dtype=self.dtype)}` and update it with context. Return `(model_inputs, index)` and update examples/docstrings.
- [ ] Update tile merger loops and context tests to unpack the tuple while preserving lazy reads and index order.
- [ ] Run the focused tests and Ruff.

```bash
uv run pytest tests/ml/tasks/test_semantic_segmentation.py tests/geodata/datasets/test_tiles.py tests/ml/models/encoder/test_model_context.py -q
uv run ruff check src/geosave_engine/ml/tasks/semantic_segmentation.py src/geosave_engine/geodata/datasets/tiles.py tests/ml/tasks/test_semantic_segmentation.py tests/geodata/datasets/test_tiles.py tests/ml/models/encoder/test_model_context.py
```

- [ ] Commit the batch-contract change.

```bash
git add src/geosave_engine/ml/tasks/semantic_segmentation.py src/geosave_engine/geodata/datasets/tiles.py tests/ml/tasks/test_semantic_segmentation.py tests/geodata/datasets/test_tiles.py tests/ml/models/encoder/test_model_context.py
git commit -m "refactor: use model-ready tuple batches"
```

## Task 4: Unify lazy raster selection and validation

**Files:**

- Modify: `src/geosave_engine/workflow/spec/requirements.py`
- Modify: `src/geosave_engine/workflow/processing.py`
- Modify: `src/geosave_engine/workflow/ingestion.py`
- Modify: `tests/workflow/spec/test_requirements.py`
- Modify: `tests/workflow/test_processing.py`
- Modify: `tests/workflow/test_ingestion.py`
- Modify: `tests/workflow/test_io.py`

- [ ] Add source-schema tests requiring exactly one of ordered `variables` or positive `channels`. Cover duplicate names, zero channels, positional metadata names other than `*`, first-N data variables, first-N positions on a sole `band` variable, too few channels, and mixed ambiguous layouts.

```python
named = RasterRequirement(variables=("B08", "B04"))
positional = RasterRequirement(channels=2)
assert list(named.select_raster(raster).data_vars) == ["B08", "B04"]
assert list(positional.select_raster(raster).data_vars) == ["B02", "B03"]
assert positional.select_raster(cube).sizes["band"] == 2
```

- [ ] Add Dask callback tests proving selection/validation schedules no tasks and preserves the selected backing arrays. Exercise caller-owned xarray input and GeoTIFF/Zarr paths through `open_rasters` plus `Processor`.
- [ ] Add acquisition tests showing named requirements set STAC `bands`, positional requirements retain runtime band loading, and both pass through the same selector/validator after loading.
- [ ] Run the tests and record schema and missing-method failures.

```bash
uv run pytest tests/workflow/spec/test_requirements.py tests/workflow/test_processing.py tests/workflow/test_ingestion.py tests/workflow/test_io.py -q
```

- [ ] Change `RasterRequirement` to optional mutually exclusive selectors and add a method that returns the selected lazy `xr.Dataset`, then validates that selected view.

```python
variables: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
channels: Annotated[int, Field(gt=0)] | None = None

def select_raster(self, raster: xr.Dataset) -> xr.Dataset:
    selected = self._select_variables(raster) if self.variables is not None else self._select_channels(raster)
    self._validate_grid(selected)
    self._validate_variables(selected)
    self._validate_attrs(selected)
    return selected
```

- [ ] Named selection uses `raster[list(self.variables)]`. Positional selection uses the first N data variables unless the Dataset has exactly one data variable with a `band` dimension, where it uses `.isel(band=slice(self.channels))`. Reject a multi-variable Dataset containing band-dimensional variables as ambiguous.
- [ ] Make `validate_raster` call `select_raster` and return `None` for its existing validation-only contract. In `Processor`, replace each active source binding with `requirement.select_raster(values[name])` before any operation.
- [ ] In `acquire`, use `select_raster` after loading. `stac_config` replaces `bands` only for named variables; positional selection leaves the copied runtime config unchanged.
- [ ] Run the focused tests, including persistence cases outside the restricted sandbox if Zarr's async writer stalls.

```bash
uv run pytest tests/workflow/spec/test_requirements.py tests/workflow/test_processing.py tests/workflow/test_ingestion.py tests/workflow/test_io.py -q
uv run ruff check src/geosave_engine/workflow/spec/requirements.py src/geosave_engine/workflow/processing.py src/geosave_engine/workflow/ingestion.py tests/workflow/spec/test_requirements.py tests/workflow/test_processing.py tests/workflow/test_ingestion.py tests/workflow/test_io.py
```

- [ ] Commit the unified selector/validation path.

```bash
git add src/geosave_engine/workflow/spec/requirements.py src/geosave_engine/workflow/processing.py src/geosave_engine/workflow/ingestion.py tests/workflow/spec/test_requirements.py tests/workflow/test_processing.py tests/workflow/test_ingestion.py tests/workflow/test_io.py
git commit -m "feat: select and validate every raster source"
```

## Task 5: Resolve ordered STAC endpoints from the model spec

**Files:**

- Modify: `src/geosave_engine/workflow/spec/requirements.py`
- Modify: `src/geosave_engine/workflow/runtime.py`
- Modify: `src/geosave_engine/workflow/flows.py`
- Modify: `src/geosave_engine/workflow/README.md`
- Modify: `src/geosave_engine/workflow/spec/README.md`
- Modify: `tests/workflow/spec/test_requirements.py`
- Modify: `tests/workflow/test_runtime.py`
- Modify: `tests/workflow/test_prefect.py`

- [ ] Add Pydantic tests for the optional `collection`/`endpoints` pair, nonempty unique HTTP(S) endpoints, preserved endpoint order, and persisted-only requirements with neither field.
- [ ] Add runtime tests with fake `Client.open`/`get_collection` behavior. Prove fallback on `requests.ConnectionError`, API status 404, and 5xx; immediate failure on 401/403, malformed STAC/JSON, invalid query/load, and a successful first endpoint. Assert exhaustion reports each endpoint and cause.

```python
requirement = RasterRequirement(
    channels=3,
    collection="sentinel-2-l2a",
    endpoints=("https://primary.test/stac", "https://backup.test/stac"),
)
sources = open_sources({"imagery": {"query": {}, "load": {}}}, {"imagery": requirement})
assert opened == ["https://primary.test/stac", "https://backup.test/stac"]
assert sources["imagery"].collection == "sentinel-2-l2a"
```

- [ ] Add flow tests requiring `spec`, rejecting source settings not present in the spec or missing required runtime bindings, and confirming runtime settings contain no `url`/`collection`.
- [ ] Run tests and record failures from the old URL/collection runtime schema.

```bash
uv run pytest tests/workflow/spec/test_requirements.py tests/workflow/test_runtime.py tests/workflow/test_prefect.py -q
```

- [ ] Add `collection: Text | None` and `endpoints: tuple[HttpUrl, ...] | None` (serialized as strings) to `RasterRequirement`, with an after-validator enforcing the pair and uniqueness.
- [ ] Change `open_sources(settings, requirements)` so primitive validation of every `query`/`load` mapping happens before any network call. For each source, call `Client.open(endpoint)` then `get_collection(collection)` to establish availability before wrapping it in `StacClient`.
- [ ] Classify fallback narrowly: request transport exceptions; `pystac_client.exceptions.APIError` with `status_code == 404` or `>= 500`. Re-raise other `APIError` values and document/parser errors immediately. When all endpoints fail, raise one `ConnectionError` whose message lists endpoint plus cause, chaining the last failure.
- [ ] Bind `StacQuery(collections=[requirement.collection], **query)` and `StacSourceConfig.model_validate(load)` only after primitive settings have validated. Do not search or load here.
- [ ] Require `spec` in `ingest`; load its source requirements, require matching runtime source keys, call `open_sources(selected_settings, requirements)`, and keep empty search/asset failures in `acquire` with no retry path.
- [ ] Update live workflow docs and examples to show model-owned acquisition identity and runtime-owned query/load.
- [ ] Run focused tests and Ruff.

```bash
uv run pytest tests/workflow/spec/test_requirements.py tests/workflow/test_runtime.py tests/workflow/test_prefect.py tests/workflow/test_ingestion.py -q
uv run ruff check src/geosave_engine/workflow/spec/requirements.py src/geosave_engine/workflow/runtime.py src/geosave_engine/workflow/flows.py tests/workflow/spec/test_requirements.py tests/workflow/test_runtime.py tests/workflow/test_prefect.py
```

- [ ] Commit endpoint resolution.

```bash
git add src/geosave_engine/workflow/spec/requirements.py src/geosave_engine/workflow/runtime.py src/geosave_engine/workflow/flows.py src/geosave_engine/workflow/README.md src/geosave_engine/workflow/spec/README.md tests/workflow/spec/test_requirements.py tests/workflow/test_runtime.py tests/workflow/test_prefect.py
git commit -m "feat: resolve model-owned STAC endpoints"
```

## Task 6: Attach typed output legends after postprocessing

**Files:**

- Modify: `src/geosave_engine/workflow/spec/model.py`
- Modify: `src/geosave_engine/workflow/spec/__init__.py`
- Modify: `src/geosave_engine/workflow/processing.py`
- Modify: `tests/workflow/spec/test_model.py`
- Modify: `tests/workflow/test_processing.py`

- [ ] Add model-spec Python/YAML round-trip tests for named `outputs` containing `OutputSpec(legend=Legend(class_map={0: "background"}))`, including class-only and class/color legends.
- [ ] Add postprocessing tests for a matching `DataArray`, a matching one-variable `Dataset`, an output already supplied to the processor, a missing output name, a multi-variable Dataset, and a non-xarray value. Assert the resulting variable's parsed attrs contain the same `Legend`.

```python
result = finish({"prediction": labels})["prediction"]
assert attrs.create_header(result).root.get(attrs.Legend) == legend

dataset = finish({"prediction": labels.to_dataset(name="class")})["prediction"]
assert attrs.create_header(dataset).data_vars["class"].get(attrs.Legend) == legend
```

- [ ] Run tests and record missing `outputs`/legend-attachment failures.

```bash
uv run pytest tests/workflow/spec/test_model.py tests/workflow/test_processing.py -q
```

- [ ] Add the typed output declaration and mapping.

```python
class OutputSpec(SpecModel):
    legend: attrs.Legend | None = None


class ModelSpec(SpecModel):
    schema_version: Literal[2]
    sources: dict[Name, RasterRequirement]
    preprocessing: dict[Name, OperationSpec] = Field(default_factory=dict)
    inference: dict[Name, OperationSpec] = Field(default_factory=dict)
    postprocessing: dict[Name, OperationSpec] = Field(default_factory=dict)
    outputs: dict[Name, OutputSpec] = Field(default_factory=dict)
```

- [ ] Retain the validated output declarations only for a postprocessing `Processor`. After operations run, look up every declared output. For a `DataArray`, call `attrs.rebase(value, legend)`; for a one-variable `Dataset`, call `attrs.rebase(value, legend, target=variable_name)`. Replace the binding with the returned xarray object.
- [ ] Raise `ValueError` naming a missing output, and `TypeError` naming non-xarray/multi-variable output values. Do not apply outputs during preprocessing.
- [ ] Export `OutputSpec`, run focused tests and Ruff.

```bash
uv run pytest tests/workflow/spec/test_model.py tests/workflow/test_processing.py tests/geodata/attrs/test_legend.py -q
uv run ruff check src/geosave_engine/workflow/spec/model.py src/geosave_engine/workflow/spec/__init__.py src/geosave_engine/workflow/processing.py tests/workflow/spec/test_model.py tests/workflow/test_processing.py
```

- [ ] Commit output semantics.

```bash
git add src/geosave_engine/workflow/spec/model.py src/geosave_engine/workflow/spec/__init__.py src/geosave_engine/workflow/processing.py tests/workflow/spec/test_model.py tests/workflow/test_processing.py
git commit -m "feat: declare semantic workflow outputs"
```

## Task 7: Harmonize the semantic-segmentation template

**Files:**

- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model.yaml`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/augmentation.yaml`
- Add: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml`
- Add: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/modules/data.py`
- Delete: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/metadata.yaml`
- Delete after migrating useful prose/code: `src/geosave_engine/workflow/examples/model_spec.yaml`
- Modify or delete after migration: `src/geosave_engine/workflow/examples/README.md`
- Modify or delete after migration: `src/geosave_engine/workflow/examples/run.py`
- Modify: `tests/workflow/spec/test_examples.py`
- Add: `tests/cli/test_workspace.py`

- [ ] Add a template test that loads both YAML documents, checks `sentinel_2_l2a`, ordered `B02/B03/B04/B08`, both approved L2A endpoints, source count equals model `in_channels`, dense legend size equals model `num_classes`, and augmentation overlays `data.init_args.augmentations`.
- [ ] Add workspace-generation tests using `create_workspace(tmp_path, "semantic_segmentation", "supervised")`; import the copied `modules.data`, instantiate its DataModule against a small deterministic prepared fixture, and assert one batch has the tuple contract.
- [ ] Add a generated-entry-point test that observes `auto_configure_optimizers=False` without starting training.
- [ ] Run tests and record failures from absent `model_spec.yaml`/DataModule and stale model fields.

```bash
uv run pytest tests/workflow/spec/test_examples.py tests/cli/test_workspace.py -q
```

- [ ] Rewrite `model.yaml` with the complete native task config: explicit counts, `model_chain`, importable criterion/optimizer/scheduler, metrics, and `data.class_path: modules.data.SegmentationDataModule`. Remove maps, source names, dataset keys, and model augmentations.
- [ ] Move the canonical workflow document into the template with this source and preprocessing:

```yaml
schema_version: 2
sources:
  sentinel_2_l2a:
    collection: sentinel-2-l2a
    endpoints:
      - https://planetarycomputer.microsoft.com/api/stac/v1/
      - https://stac.dataspace.copernicus.eu/v1/
    variables: [B02, B03, B04, B08]
    require_crs: true
preprocessing:
  valid_pixels:
    call: !ref sentinel_2_l2a.gs.to_nan
  image:
    call: !ref valid_pixels.gs.unpack
outputs:
  prediction:
    legend:
      class_map: {0: background, 1: vegetation}
      color_map: {0: "#000000", 1: "#00ff00"}
```

- [ ] Implement the workspace-owned LitData adapter. Its constructor accepts train/validation paths, input size, batch size, workers, and augmentation declarations. `setup` creates `StreamingDataset` instances whose prepared items are `(image, target)`. A custom collate function first uses PyTorch's `default_collate`, applies `ImageAugmenter(augmentations, size=input_size, data_keys=["image", "mask"])` to the batched tensors, then returns `({"image": image}, target)`. The DataModule returns `StreamingDataLoader` instances with that collate function.
- [ ] Point `augmentation.yaml` at `data.init_args.augmentations`. Delete `metadata.yaml`.
- [ ] Migrate any useful workflow-example instructions to the template/readme docs, then remove the duplicate example document and update tests to use the template-owned path. Preserve the user's current workflow-example changes by translating them, not by restoring old content.
- [ ] Run template/workspace tests and Ruff.

```bash
uv run pytest tests/workflow/spec/test_examples.py tests/cli/test_workspace.py tests/ml/tasks/test_semantic_segmentation.py -q
uv run ruff check src/geosave_engine/templates/tasks/semantic_segmentation/supervised/modules/data.py tests/cli/test_workspace.py tests/workflow/spec/test_examples.py
```

- [ ] Commit the semantic template migration.

```bash
git add src/geosave_engine/templates/tasks/semantic_segmentation/supervised src/geosave_engine/workflow/examples tests/workflow/spec/test_examples.py tests/cli/test_workspace.py
git commit -m "feat: ship aligned segmentation workspace templates"
```

## Task 8: Add the ordinary-Lightning custom task template

**Files:**

- Add: `src/geosave_engine/templates/tasks/custom/lightning/description.txt`
- Add: `src/geosave_engine/templates/tasks/custom/lightning/configs/model.yaml`
- Add: `src/geosave_engine/templates/tasks/custom/lightning/modules/task.py`
- Add: `src/geosave_engine/templates/tasks/custom/lightning/modules/data.py`
- Modify: `tests/cli/test_workspace.py`

- [ ] Add a generated-workspace test that checks `get_tasks()["custom"]` includes `lightning`, creates the workspace, imports both modules in a fresh process, instantiates the configured model/data classes through LightningCLI with `run=False`, and executes one train and validation batch.

```python
create_workspace(root, "custom", "lightning")
result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=root,
    check=False,
    capture_output=True,
    text=True,
)
assert result.returncode == 0, result.stderr
```

- [ ] Run the test and record failure because the custom template is not yet listed.

```bash
uv run pytest tests/cli/test_workspace.py -q
```

- [ ] Add `modules/data.py` with a deterministic `Dataset` that derives fixed input and target tensors from the index, plus a `LightningDataModule` returning ordinary PyTorch DataLoaders. Keep the replacement seam explicit in class/method docstrings.

```python
class ExampleDataset(Dataset[tuple[dict[str, Tensor], Tensor]]):
    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        generator = torch.Generator().manual_seed(index)
        inputs = torch.rand(4, generator=generator)
        target = (inputs.sum() > 2).long()
        return {"features": inputs}, target
```

- [ ] Add `modules/task.py` with an ordinary `LightningModule`: a small network, `forward(features=features)`, shared loss calculation, `training_step`, `validation_step`, and native `torch.optim.AdamW` in `configure_optimizers`. It must not import GeoSave model-chain/task abstractions.
- [ ] Add `configs/model.yaml` that selects `modules.task.CustomTask` and `modules.data.CustomDataModule`, with a short CPU trainer configuration. The copied common entry point provides LightningCLI.
- [ ] Run workspace tests, one actual `fast_dev_run` smoke from the generated template, and Ruff.

```bash
uv run pytest tests/cli/test_workspace.py -q
uv run ruff check src/geosave_engine/templates/tasks/custom/lightning tests/cli/test_workspace.py
```

- [ ] Commit the custom template.

```bash
git add src/geosave_engine/templates/tasks/custom/lightning tests/cli/test_workspace.py
git commit -m "feat: add custom Lightning task template"
```

## Task 9: Audit the breaking change and verify the affected system

**Files:**

- Modify as search reveals: live `src/`, `tests/`, and `docs/` files outside notebooks
- Modify: `docs/superpowers/plans/2026-09-25-ml-construction-and-training-config.md` execution record only after implementation

- [ ] Search for every removed public name and old data shape. Expected result: no live matches except explicit migration prose in the approved spec/plan.

```bash
rg -n "BuildSpec|build_model|build_loss|build_optimizer|build_scheduler|AdamW\.split|image_key|label_key|model_context|\[\"layers" src tests docs --glob '!*.ipynb' --glob '!docs/superpowers/specs/2026-09-25-ml-construction-and-training-config-design.md' --glob '!docs/superpowers/plans/2026-09-25-ml-construction-and-training-config.md'
rg -n "init_args:\s*$|stages:|loss:|scheduler:" src/geosave_engine/templates tests --glob '*.yaml' --glob '*.py'
```

- [ ] Run the focused ML suite.

```bash
uv run pytest tests/ml/registry tests/ml/models/contract tests/ml/models/encoder/test_model_context.py tests/ml/tasks tests/ml/callbacks tests/geodata/datasets/test_tiles.py -q
```

- [ ] Run the focused workflow/template suite.

```bash
uv run pytest tests/workflow tests/cli/test_workspace.py -q
```

- [ ] Run scoped lint and whitespace checks.

```bash
uv run ruff check src/geosave_engine/ml src/geosave_engine/workflow src/geosave_engine/geodata/datasets/tiles.py src/geosave_engine/templates/tasks tests/ml tests/workflow tests/geodata/datasets/test_tiles.py tests/cli/test_workspace.py
git diff --check
```

- [ ] Run the full suite if the focused suites pass; report existing unrelated failures separately with evidence rather than changing unrelated code.

```bash
uv run pytest -q
```

- [ ] Review the final diff against the approved spec, specifically confirming no compatibility paths, no pixel compute during selection, no unintended changes to the user's pre-existing workflow edits, and no generated/cache files staged.

```bash
git status --short
git diff --stat
git diff -- src/geosave_engine/ml src/geosave_engine/workflow src/geosave_engine/geodata/datasets/tiles.py src/geosave_engine/templates tests/ml tests/workflow tests/geodata/datasets tests/cli
```

- [ ] Append an evidence-based execution record to this plan and commit only any final audit fixes plus the record.

```bash
git add docs/superpowers/plans/2026-09-25-ml-construction-and-training-config.md
git commit -m "docs: record ML configuration verification"
```

# Modular ML Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize configured PyTorch construction in `ml.registry`, make `ModelChain` a readable execution-only module, make `GeoSaveModel` the sole Hugging Face adapter, and provide a modular built-in segmentation Lightning pair.

**Architecture:** `ml.registry` resolves and builds configured criteria, models, optimizers, and schedulers. `ml.model_chain` composes actual modules and routes named values without persistence behavior. `ml.huggingface` adapts configured chains to Transformers, while built-in Lightning modules consume the same public builders available to user code.

**Tech Stack:** Python 3.12, PyTorch, Lightning, jsonargparse through LightningCLI, Transformers, xarray, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-ml-registry-builders-design.md`

## Global Constraints

- Do not restore compatibility aliases for `geosave_engine.ml.models.contract` or ModelChain Hub methods.
- Keep notebooks out of migration, lint, and test scope.
- Keep model-chain routing behavior unchanged while improving names and layout.
- Use exact direct-child names for optimizer groups; never infer groups from parameter-name substrings.
- Keep native constructor exceptions intact after registry validation.
- Do not implement model-spec inference inputs, postprocessing, dense merging, threshold export, or the concrete supervised dataset in this phase.
- Preserve unrelated workflow/spec changes already in the worktree.
- Reuse the user's in-progress move of OHEM into `ml/registry/criterion/ohem.py`; remove only the conflicting empty `criterion.py` and misspelled empty `scheduier.py` files.

## Review Focus

- A factory spec containing neither or both selectors must fail before import or construction.
- Registered names are case-insensitive, but unknown names and wrong imported base classes fail clearly.
- Optimizer groups cannot duplicate shared parameters, omit unselected trainable parameters, or include frozen parameters.
- A directly composed `ModelChain` runs normally but cannot be exported without a registry recipe.
- Transformer reload must reproduce stage order, parameters, outputs, and evaluation mode in a fresh offline process.

---

### Task 1: Shared factory, criterion, optimizer, and scheduler builders

**Files:**
- Create: `src/geosave_engine/ml/registry/factory.py`
- Modify: `src/geosave_engine/ml/registry/criterion/__init__.py`
- Keep: `src/geosave_engine/ml/registry/criterion/ohem.py`
- Modify: `src/geosave_engine/ml/registry/optimizer.py`
- Create: `src/geosave_engine/ml/registry/scheduler.py`
- Delete: `src/geosave_engine/ml/registry/criterion.py`
- Delete: `src/geosave_engine/ml/registry/scheduier.py`
- Modify: `src/geosave_engine/ml/registry/__init__.py`
- Create: `tests/ml/registry/test_builders.py`

**Interfaces:**
- Produces: `BuildSpec`, `CriterionSpec`, `OptimizerSpec`, `SchedulerSpec`, `build_criterion`, `build_optimizer`, and `build_scheduler`.
- Consumes: registered callables, importable PyTorch classes, `nn.Module`, and `Optimizer` instances.

- [ ] **Step 1: Write failing shared resolver and criterion tests**

```python
@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"name": "cross_entropy", "class_path": "torch.nn.CrossEntropyLoss"},
        {"name": "missing"},
        {"class_path": "CrossEntropyLoss"},
        {"class_path": "torch.nn.CrossEntropyLoss", "unknown": True},
        {"class_path": "torch.nn.CrossEntropyLoss", "init_args": []},
    ],
)
def test_build_spec_rejects_invalid_specs(spec):
    with pytest.raises(ValidationError):
        BuildSpec.model_validate(spec)


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "CrOsS_EnTrOpY", "init_args": {"ignore_index": 255}},
        {"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"ignore_index": 255}},
    ],
)
def test_build_criterion_supports_registered_names_and_class_paths(spec):
    criterion = build_criterion(spec)
    logits = torch.zeros(2, 3)
    target = torch.tensor([1, 255])
    assert criterion(logits, target).item() == pytest.approx(torch.log(torch.tensor(3.0)).item())


def test_build_criterion_exposes_ohem():
    criterion = build_criterion({"name": "ohem", "init_args": {"ignore_index": 255}})
    assert isinstance(criterion, ProbOhemCrossEntropy2d)


def test_build_criterion_rejects_optimizer_classes():
    with pytest.raises(TypeError, match="Module subclass"):
        build_criterion({"class_path": "torch.optim.AdamW"})
```

- [ ] **Step 2: Run resolver and criterion tests to verify RED**

Run: `uv run pytest tests/ml/registry/test_builders.py -q`

Expected: collection fails because the public builders do not exist.

- [ ] **Step 3: Implement Pydantic `BuildSpec` and the criterion package**

```python
class BuildSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1)
    class_path: str | None = None
    init_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        if (self.name is None) == (self.class_path is None):
            raise ValueError("Specify exactly one of name or class_path")
        return self

    def resolve[T](self, registry, base: type[T]) -> Callable[..., T]:
        if self.name is not None:
            factories = {key.casefold(): value for key, value in registry.items()}
            if self.name.casefold() not in factories:
                raise ValueError(f"Unknown name {self.name!r}")
            return factories[self.name.casefold()]
        module, _, attribute = self.class_path.rpartition(".")
        factory = getattr(import_module(module), attribute, None)
        if not isinstance(factory, type) or not issubclass(factory, base):
            raise TypeError(f"{self.class_path!r} must name a {base.__name__} subclass")
        return factory
```

Define `CriterionSpec(BuildSpec)` and register `CROSS_ENTROPY` plus `OHEM` in `registry/criterion/__init__.py`. Builders accept models or mappings, normalize with `model_validate`, call the spec's `resolve`, and forward `init_args` unchanged.

- [ ] **Step 4: Write failing optimizer and scheduler tests**

```python
def test_optimizer_groups_exact_children_and_remaining_parameters():
    model = nn.ModuleDict({"encoder": nn.Linear(2, 2), "head": nn.Linear(2, 1)})
    model["encoder"].bias.requires_grad_(False)
    optimizer = build_optimizer(
        {
            "name": "adamw",
            "init_args": {"lr": 1e-3, "weight_decay": 1e-2},
            "groups": {"head": {"lr": 1e-4, "weight_decay": 0.0}},
        },
        model,
    )
    head, remaining = optimizer.param_groups
    assert head["params"] == [model["head"].weight, model["head"].bias]
    assert remaining["params"] == [model["encoder"].weight]
    assert (head["lr"], head["weight_decay"]) == (1e-4, 0.0)


def test_optimizer_rejects_unknown_and_overlapping_groups():
    first = nn.Linear(1, 1)
    second = nn.Linear(1, 1)
    second.weight = first.weight
    model = nn.ModuleDict({"first": first, "second": second})
    with pytest.raises(ValueError, match="share parameters"):
        build_optimizer({"name": "adamw", "groups": {"first": {}, "second": {}}}, model)
    with pytest.raises(ValueError, match="Unknown model groups"):
        build_optimizer({"name": "adamw", "groups": {"missing": {}}}, model)


def test_optimizer_rejects_a_model_without_trainable_parameters():
    model = nn.Linear(1, 1).requires_grad_(False)
    with pytest.raises(ValueError, match="trainable parameters"):
        build_optimizer({"name": "adamw"}, model)


def test_scheduler_returns_lightning_metadata():
    optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)
    configured = build_scheduler(
        {
            "name": "reduce_on_plateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
            "interval": "epoch",
            "strict": False,
        },
        optimizer,
    )
    assert isinstance(configured["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert configured["scheduler"].optimizer is optimizer
    assert {key: configured[key] for key in ("monitor", "interval", "strict")} == {
        "monitor": "val_loss", "interval": "epoch", "strict": False
    }
```

- [ ] **Step 5: Run optimizer and scheduler tests to verify RED**

Run: `uv run pytest tests/ml/registry/test_builders.py -q`

Expected: optimizer and scheduler imports or behavior fail.

- [ ] **Step 6: Implement optimizer and scheduler builders**

Register `ADAMW`, `ADAM`, `SGD`, `RMSPROP`, and `ADAGRAD`. Validate `groups` as mappings, resolve selector fields separately, reject duplicate parameter identities across explicit groups, exclude frozen parameters, append remaining trainable parameters, and reject an empty parameter list.

Register `COSINE_ANNEALING`, `REDUCE_ON_PLATEAU`, and `STEP`. `SchedulerSpec` adds Lightning's `interval`, `frequency`, `monitor`, `strict`, and `name` keys. Return `{"scheduler": scheduler, **metadata}`.

- [ ] **Step 7: Run builder tests to verify GREEN**

Run: `uv run pytest tests/ml/registry/test_builders.py -q`

Expected: all builder tests pass.

- [ ] **Step 8: Commit the general builders**

```bash
git add src/geosave_engine/ml/registry tests/ml/registry/test_builders.py src/geosave_engine/ml/loss
git commit -m "feat: restore configurable ML builders"
```

### Task 2: Readable model-chain package and configured model builder

**Files:**
- Create: `src/geosave_engine/ml/model_chain/__init__.py`
- Create: `src/geosave_engine/ml/model_chain/chain.py`
- Create: `src/geosave_engine/ml/model_chain/routing.py`
- Create: `src/geosave_engine/ml/model_chain/step.py`
- Create: `src/geosave_engine/ml/model_chain/published.py`
- Delete: `src/geosave_engine/ml/models/contract/`
- Modify: `src/geosave_engine/ml/registry/model.py`
- Modify: `src/geosave_engine/ml/registry/__init__.py`
- Move: `tests/ml/models/contract/test_chain.py` to `tests/ml/model_chain/test_chain.py`
- Move: `tests/ml/models/contract/test_step.py` to `tests/ml/model_chain/test_step.py`
- Move: `tests/ml/models/contract/test_published.py` to `tests/ml/model_chain/test_published.py`
- Move: `tests/ml/registry/test_model.py` remains in place and covers `build_model`.
- Modify: `tests/ml/callbacks/test_task_callbacks.py`
- Modify: all live Python imports under `src/` and `tests/`, excluding notebooks.

**Interfaces:**
- Produces: `geosave_engine.ml.model_chain.{ModelChain, Published, chain_step}` and `registry.build_model(stage_specs) -> ModelChain`.
- Consumes: shared `BuildSpec`/`resolve` and registered model factories.

- [ ] **Step 1: Change tests to the desired public imports and construction path**

```python
from geosave_engine.ml.model_chain import ModelChain, Published, chain_step
from geosave_engine.ml.registry import StageSpec, build_model


def test_build_model_records_an_independent_resolved_recipe(model_factories):
    specs = {"encoder": {"name": "test"}, "head": {"name": "test"}}
    model = build_model(specs)
    specs.clear()
    assert list(model.stage_specs) == ["encoder", "head"]
    copied = model.stage_specs
    copied.clear()
    assert list(model.stage_specs) == ["encoder", "head"]


def test_direct_chain_has_no_construction_recipe():
    chain = ModelChain(encoder=Encoder(), head=Head(2))
    with pytest.raises(ValueError, match="stage specifications"):
        _ = chain.stage_specs


def test_model_chain_has_no_huggingface_mixin_methods():
    chain = ModelChain(encoder=Encoder(), head=Head(2))
    assert not hasattr(chain, "save_pretrained")
    assert not hasattr(chain, "from_pretrained")
    assert not hasattr(chain, "push_to_hub")
```

Retain the existing routing, ambiguity, cycle, annotation, type-validation, and published-value tests while changing their import paths and configured construction calls.

- [ ] **Step 2: Run chain and model-registry tests to verify RED**

Run: `uv run pytest tests/ml/model_chain tests/ml/registry/test_model.py -q`

Expected: the new package and `build_model` are absent.

- [ ] **Step 3: Move and clean up the model-chain implementation**

Make `ModelChain` inherit only `nn.Module`. Its constructor accepts module instances only, initializes `_stage_specs` to `None`, registers modules, asks `routing.py` for ordered steps and required external inputs, and executes those steps. Remove JSON, path, Hub mixin, and persistence methods.

Move `graph.py` to `routing.py`; use names that state effects (`ordered_steps`, `required_inputs`, `dependencies`) and keep bound-step details internal. In `step.py`, rename `_check_value` to `_validate_value` and `_return_names` to `_infer_output_names`, and keep decorator parsing separate from invocation. In `published.py`, keep only `Published` public.

- [ ] **Step 4: Refactor the model registry around `build_model`**

Make `StageSpec` extend `BuildSpec`, delegate selector validation to `resolve`, retain published-value/default snapshot logic, and construct the final chain from built modules. Set the chain's internal recipe to an independent copy of resolved stage specifications before returning it. `model_chain` must not import `registry`, avoiding a construction/execution cycle.

- [ ] **Step 5: Migrate every live import and verify no old path remains**

Run: `rg -n "ml\.models\.contract|models\.contract|ModelChain\(stages=" src tests src/geosave_engine/templates -g '*.py' -g '*.md' -g '*.yaml' -g '!*.ipynb'`

Expected: no live matches after migrating model implementations, callbacks, tasks, Hugging Face adapter, tests, and user-facing documentation. Historical specs and superseded plans are not part of this live-path audit.

- [ ] **Step 6: Run model-chain and registry tests to verify GREEN**

Run: `uv run pytest tests/ml/model_chain tests/ml/registry -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the model-chain move**

```bash
git add src/geosave_engine/ml/model_chain src/geosave_engine/ml/models src/geosave_engine/ml/registry tests/ml/model_chain tests/ml/registry docs
git commit -m "refactor: separate model construction from execution"
```

### Task 3: Transformers-only Hugging Face publication

**Files:**
- Modify: `src/geosave_engine/ml/huggingface.py`
- Delete: `tests/ml/models/contract/test_hub.py`
- Delete: `tests/ml/models/contract/test_push_to_hub.py`
- Move and rewrite: `tests/ml/models/contract/test_automodel.py` to `tests/ml/test_huggingface.py`
- Move reusable test models into: `tests/ml/model_chain/conftest.py`
- Modify: `tests/ml/models/encoder/test_model_context.py`
- Modify: `src/geosave_engine/ml/models/README.md`
- Modify: `pyproject.toml`
- Modify mechanically: `uv.lock`

**Interfaces:**
- Consumes: configured `ModelChain.stage_specs` and `registry.build_model`.
- Produces: `GeoSaveConfig` and `GeoSaveModel` as the only save/load/upload interface.

- [ ] **Step 1: Rewrite publication tests around `GeoSaveModel`**

```python
def test_transformers_roundtrip_preserves_chain_recipe_weights_and_outputs(tmp_path, stages):
    chain = build_model(stages)
    encoder = chain.get_submodule("z_encoder")
    with torch.no_grad():
        encoder.scale.fill_(7.0)
    published = GeoSaveModel.from_chain(chain)
    published.eval()
    published.save_pretrained(tmp_path)
    restored = AutoModel.from_pretrained(tmp_path, local_files_only=True)
    assert list(restored.chain.stage_specs) == ["z_encoder", "a_head"]
    torch.testing.assert_close(restored(image=torch.tensor(3.0)), torch.tensor(23.0))
    for name, value in chain.state_dict().items():
        torch.testing.assert_close(restored.chain.state_dict()[name], value)


def test_export_rejects_a_directly_composed_chain():
    with pytest.raises(ValueError, match="stage specifications"):
        GeoSaveModel.from_chain(ModelChain(encoder=Encoder(), head=Head(2)))
```

Retain the existing fresh-process `trust_remote_code=True` test, but build its source chain through `build_model`. Change the upload fixture to invoke `GeoSaveModel.push_to_hub`, proving the adapter—not `ModelChain`—owns upload.

- [ ] **Step 2: Run Hugging Face tests to verify RED**

Run: `uv run pytest tests/ml/test_huggingface.py -q`

Expected: adapter reconstruction still calls `ModelChain(stages=...)` or tests reference removed mixin behavior.

- [ ] **Step 3: Make `GeoSaveModel` reconstruct through the model registry**

Import `ModelChain` from `ml.model_chain` and `build_model` from `ml.registry`. In `GeoSaveModel.__init__`, reconstruct omitted chains with `build_model(specs)`. Keep `from_chain` sharing the supplied modules and state. Preserve `no_init_weights()` around `post_init()`.

- [ ] **Step 4: Remove the direct Hub dependency and obsolete documentation**

Remove `huggingface-hub` from base dependencies, run `uv lock`, and remove every claim that a plain `ModelChain` saves or uploads. Document `GeoSaveModel.from_chain(...).save_pretrained(...)` as the only publication path.

- [ ] **Step 5: Run adapter tests to verify GREEN**

Run: `uv run pytest tests/ml/test_huggingface.py tests/ml/model_chain tests/ml/models/encoder/test_model_context.py -q`

Expected: all tests pass, including the fresh offline process.

- [ ] **Step 6: Commit Hugging Face separation**

```bash
git add src/geosave_engine/ml/huggingface.py tests/ml pyproject.toml uv.lock src/geosave_engine/ml/models/README.md
git commit -m "refactor: isolate Hugging Face publication"
```

### Task 4: Built-in segmentation task and data-module skeleton

**Files:**
- Modify: `src/geosave_engine/ml/tasks/semantic_segmentation.py`
- Create: `src/geosave_engine/ml/data/__init__.py`
- Create: `src/geosave_engine/ml/data/semantic_segmentation.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model.yaml`
- Delete: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/modules/data.py`
- Modify: `src/geosave_engine/templates/tasks/semantic_segmentation/supervised/README.md`
- Modify: `tests/ml/tasks/test_semantic_segmentation.py`
- Create: `tests/ml/data/test_semantic_segmentation.py`
- Modify: `tests/cli/test_workspace.py`

**Interfaces:**
- Consumes: all public builders and `ModelChain`.
- Produces: a modular `SemanticSegmentationTask` and matching `SemanticSegmentationDataModule` top-level LightningCLI classes.

- [ ] **Step 1: Write failing task-delegation tests**

```python
def test_task_accepts_registered_training_builders(stages):
    task = SemanticSegmentationTask(
        in_channels=2,
        num_classes=2,
        model_chain=stages,
        criterion={"name": "cross_entropy", "init_args": {"ignore_index": -1}},
        optimizer={"name": "sgd", "init_args": {"lr": 0.1}},
        lr_scheduler={
            "name": "reduce_on_plateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
        },
    )
    task.configure_model()
    configured = task.configure_optimizers()
    assert isinstance(task.criterion, nn.CrossEntropyLoss)
    assert task.criterion.ignore_index == -1
    assert isinstance(configured["optimizer"], torch.optim.SGD)
    assert configured["lr_scheduler"]["monitor"] == "val_loss"
```

Keep tests proving task dimensions override first/last model-stage arguments and input specifications remain unmutated.

- [ ] **Step 2: Write failing data-module and workspace tests**

```python
def test_segmentation_data_module_exposes_normal_lightning_lifecycle():
    data = SemanticSegmentationDataModule(
        train_path="train", val_path="val", input_size=32, batch_size=4, num_workers=0
    )
    assert isinstance(data, LightningDataModule)
    assert data.input_size == (32, 32)
    with pytest.raises(NotImplementedError, match="supervised Dataset"):
        data.setup("fit")


def test_generated_workspace_uses_library_lightning_pair(workspace):
    config = yaml.safe_load((workspace / "configs" / "model.yaml").read_text())
    assert config["model"]["class_path"] == "geosave_engine.ml.tasks.SemanticSegmentationTask"
    assert config["data"]["class_path"] == "geosave_engine.ml.data.SemanticSegmentationDataModule"
    assert not (workspace / "modules" / "data.py").exists()
```

- [ ] **Step 3: Run task/data/workspace tests to verify RED**

Run: `uv run pytest tests/ml/tasks/test_semantic_segmentation.py tests/ml/data/test_semantic_segmentation.py tests/cli/test_workspace.py -q`

Expected: named builders and library data module are absent.

- [ ] **Step 4: Delegate all task construction**

Remove task-local construction TypedDicts, class importing, optimizer grouping, and scheduler metadata handling. Merge task `ignore_index` into a copied criterion spec before `build_criterion`; use `build_model`, `build_optimizer`, and `build_scheduler` directly. Keep task-specific model dimension injection, training steps, metrics, and threshold state local.

- [ ] **Step 5: Add the matching data-module skeleton**

Implement constructor storage for `train_path`, `val_path`, normalized `input_size`, `batch_size`, and `num_workers`. Provide `_dataset(path, training)` that raises `NotImplementedError("SemanticSegmentationDataModule supervised Dataset is not implemented yet")`; make `setup`, `_loader`, `train_dataloader`, and `val_dataloader` follow normal Lightning lifecycle around that seam.

Point generated YAML at the library class, remove the unsupported augmentation field, delete the workspace-local module, and update README instructions.

- [ ] **Step 6: Run task/data/workspace tests to verify GREEN**

Run: `uv run pytest tests/ml/tasks tests/ml/data tests/cli/test_workspace.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the built-in Lightning pair**

```bash
git add src/geosave_engine/ml/tasks src/geosave_engine/ml/data src/geosave_engine/templates/tasks/semantic_segmentation tests/ml/tasks tests/ml/data tests/cli/test_workspace.py
git commit -m "refactor: modularize segmentation training"
```

### Task 5: Model-input encoding placement and final migration checks

**Files:**
- Create: `src/geosave_engine/ml/encoding/__init__.py`
- Create: `src/geosave_engine/ml/encoding/time.py`
- Delete: `src/geosave_engine/ml/models/context/`
- Modify: `src/geosave_engine/ml/models/encoder/clay.py`
- Modify: `src/geosave_engine/ml/models/encoder/prithvi.py`
- Move or add: focused datetime-label tests under `tests/ml/encoding/test_time.py`
- Modify: live documentation imports and package-layout text.

**Interfaces:**
- Produces: shared native-data-to-model-input encoding ownership under `ml.encoding`.
- Consumes: xarray `Dataset`/`DataArray` and registered geodata accessors.

- [ ] **Step 1: Move the time-label tests to the desired import**

```python
from geosave_engine.ml.encoding.time import time_labels


def test_time_labels_preserve_frame_order(raster):
    actual = time_labels(raster.assign_coords(time=["2024-02-01", "2023-01-01"]))
    assert [value.isoformat() for value in actual] == [
        "2024-02-01T00:00:00", "2023-01-01T00:00:00"
    ]
```

Retain missing, empty, multidimensional, non-datetime, and `NaT` validation cases.

- [ ] **Step 2: Run encoding and encoder-context tests to verify RED**

Run: `uv run pytest tests/ml/encoding/test_time.py tests/ml/models/encoder/test_model_context.py -q`

Expected: `ml.encoding.time` is absent.

- [ ] **Step 3: Move time encoding and update encoder imports**

Move the existing implementation unchanged into `ml/encoding/time.py`, export nothing broader from `ml.encoding`, update Clay and Prithvi imports, and delete the empty old context package. Keep encoder `model_context` methods for the current TileDataset path; the spec explicitly defers their replacement by model-spec inference inputs.

- [ ] **Step 4: Run all focused ML and workspace tests**

Run: `uv run pytest tests/ml tests/cli/test_workspace.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Audit removed paths and methods**

Run:

```bash
rg -n "ml\.models\.contract|models\.contract|models\.context|PyTorchModelHubMixin|ModelChain\.(from_pretrained|save_pretrained|push_to_hub)|ModelChain\(stages=" src tests src/geosave_engine/templates -g '*.py' -g '*.md' -g '*.yaml' -g '!*.ipynb'
```

Expected: no matches in live source, tests, templates, or user-facing ML documentation. Historical design documents remain unchanged.

- [ ] **Step 6: Run full verification**

Run:

```bash
uv run pytest
uv run ruff check src/geosave_engine/ml src/geosave_engine/templates/tasks/semantic_segmentation tests/ml tests/cli/test_workspace.py
git diff --check
```

Expected: the full suite and Ruff pass. Report every unrelated failure by test name.

- [ ] **Step 7: Commit encoding placement and final migration**

```bash
git add src/geosave_engine/ml/encoding src/geosave_engine/ml/models tests/ml/encoding docs
git commit -m "refactor: clarify model input encoding ownership"
```

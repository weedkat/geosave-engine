# Workflow Layering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the job-domain workflow packages with stable `configs`, `flows`, and `tasks` layers, removing the duplicated recipe checks, Zarr publication, and submission code found during the gap review, while preserving current ingestion and dense-preparation behavior.

**Architecture:** Runtime schemas live in `workflow.configs`, independently runnable Prefect entry points live in `workflow.flows`, and reusable work plus its cohesive storage operations live in `workflow.tasks`. `ModelSpec` continues to own raster acquisition and gains the single recipe check that flows call before I/O. One atomic Zarr writer serves both stacks and Zarr samples.

**Tech Stack:** Python 3.12, Pydantic 2, Prefect 3, xarray/DataTree, GeoPandas, pytest, Ruff, BasedPyright, Hatchling

**Spec:** `docs/superpowers/specs/2026-09-30-workflow-layering-design.md` (see "Consolidation during the move")

## Global Constraints

- Preserve the current ingestion, dense preparation, metadata, resume, bounded-concurrency, laziness, and atomic-publication behavior.
- Keep `ModelSpec` as the owner of raster acquisition and preprocessing declarations.
- Do not restore removed prediction, postprocessing, or workflow source-configuration APIs.
- Do not add compatibility aliases for `workflow.ingestion` or `workflow.training_data`.
- Keep only independently runnable jobs as flows.
- Keep `prepare_dense_sample` as the only submitted concurrency-bound task; do not decorate storage helpers merely because they live in `workflow.tasks`.
- Do not add `ModelSpec.preprocess()` to dense preparation and do not move the GeoTIFF time re-expansion out of `tasks.dense`.
- Use `git mv` for moved files; edit with the Edit tool, never `sed -i` or blind regex sweeps.
- Preserve unrelated dirty-worktree changes and stage only task-owned paths.

## Review Focus

- A missing raster recipe must fail before opening a remote raster anchor or contacting STAC, now through `ModelSpec.require_recipes()`; Task 2 keeps the ingest no-I/O assertion and Task 3 keeps the no-submission assertion.
- Invalid or mismatched metadata must fail before the first `prepare_dense_sample.submit()` and must not replace an existing manifest; Task 3 retains both assertions.
- A failed sample must stop further submission after the loop rewrite; Task 3 retains `test_prepare_dense_data_stops_submitting_after_failure` and must pass unmodified apart from imports.
- Zarr samples written through the shared `write_stack` must still forward writer options and keep atomic, non-overwriting publication; Task 3 adds a forwarding test.
- A generated workspace and built wheel must import only the layered paths; Task 4 adds stale-import and clean-artifact checks.

---

### Task 1: One recipe check on ModelSpec

**Files:**
- Modify: `src/geosave_engine/model_spec/model.py:77-93`
- Test: `tests/model_spec/test_acquisition.py`

**Interfaces:**
- Produces: `ModelSpec.require_recipes(self) -> None`, raising `ValueError("Rasters need STAC recipes: [...]")` listing every raster without a `stac` recipe in declaration order.

- [ ] **Step 1: Write the failing test**

Add beside `test_load_rasters_reports_all_missing_recipes_before_http`:

```python
def test_require_recipes_lists_every_raster_without_a_recipe():
    model = ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(channels=3),
            "elevation": RasterRequirement(channels=1),
        },
    )

    with pytest.raises(ValueError, match=r"\['optical', 'elevation'\]"):
        model.require_recipes()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/model_spec/test_acquisition.py -k recipes`
Expected: FAIL with `AttributeError: 'ModelSpec' object has no attribute 'require_recipes'`.

- [ ] **Step 3: Implement and reuse it in `load_rasters`**

```python
    def require_recipes(self) -> None:
        """Require a STAC recipe for every declared raster.

        Raises:
            ValueError: If any raster has no STAC recipe.
        """
        missing = [name for name, requirement in self.rasters.items() if requirement.stac is None]
        if missing:
            raise ValueError(f"Rasters need STAC recipes: {missing}")

    def load_rasters(self, anchor: GeoAnchor, /) -> dict[str, xr.Dataset]:
        """Load and validate every declared raster on an anchor."""
        model = self._validated()
        model.require_recipes()

        rasters = {}
        ...  # loop unchanged
```

- [ ] **Step 4: Run model-spec tests**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/model_spec`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/geosave_engine/model_spec/model.py tests/model_spec/test_acquisition.py
git commit -m "refactor: let model spec own the recipe check"
```

### Task 2: Configs, ingest flow, and shared stack writer

**Files:**
- Create: `src/geosave_engine/workflow/configs/__init__.py`, `configs/base.py`, `configs/anchor.py`
- Create: `src/geosave_engine/workflow/flows/__init__.py`, `flows/ingest.py`
- Create: `src/geosave_engine/workflow/tasks/__init__.py`, `tasks/stack.py`
- Modify: `src/geosave_engine/cli/commands/workflow/ingest.py`
- Delete: `src/geosave_engine/workflow/ingestion/` (all three files)
- Move: `tests/workflow/ingestion/test_anchor.py` → `tests/workflow/configs/test_anchor.py`
- Move: `tests/workflow/ingestion/test_flow.py` → `tests/workflow/flows/test_ingest.py`
- Move: `tests/workflow/ingestion/test_storage.py` → `tests/workflow/tasks/test_stack.py`
- Modify: `tests/cli/commands/test_workflow.py`

**Interfaces:**
- Consumes: `ModelSpec.require_recipes()` (Task 1), `ModelSpec.load(path)`, `ModelSpec.load_rasters(anchor)`.
- Produces: `ConfigModel`; `AnchorConfig`, `CoordinateAnchorConfig`, `GeoJSONAnchorConfig`, `RasterAnchorConfig`; `write_stack(rasters: Mapping[str, xr.Dataset], output: str | Path, **options: Any) -> str`; Prefect flow `ingest(anchor: dict[str, JsonValue], *, output: str, spec: str) -> str`.

- [ ] **Step 1: Move tests and point them at layered paths**

Before moving, remove untracked leftovers so `git mv` targets are clean:

```bash
find tests/workflow -name __pycache__ -type d -prune -exec rm -rf {} +
git mv tests/workflow/ingestion/test_anchor.py tests/workflow/configs/test_anchor.py
git mv tests/workflow/ingestion/test_flow.py tests/workflow/flows/test_ingest.py
git mv tests/workflow/ingestion/test_storage.py tests/workflow/tasks/test_stack.py
```

Update imports:

```python
# tests/workflow/configs/test_anchor.py
from geosave_engine.workflow.configs import AnchorConfig

# tests/workflow/flows/test_ingest.py
from geosave_engine.workflow.flows import ingest
ingest_module = import_module("geosave_engine.workflow.flows.ingest")
# and: monkeypatch.setattr(ingest_module, "write_stack", write)

# tests/workflow/tasks/test_stack.py
from geosave_engine.workflow.tasks.stack import write_stack
```

Rename every `_write_stack` call in `test_stack.py` to `write_stack`. Add an options-forwarding test:

```python
def test_write_stack_forwards_native_writer_options(raw, tmp_path, monkeypatch):
    captured = {}
    original = io.zarr.write

    def write(tree, path, **options):
        captured.update(options)
        return original(tree, path, **options)

    monkeypatch.setattr(io.zarr, "write", write)

    write_stack(raw, tmp_path / "raw.zarr", consolidated=True)

    assert captured == {"compute": True, "overwrite": False, "consolidated": True}
```

In `tests/cli/commands/test_workflow.py`, import `from geosave_engine.workflow import flows` and `from geosave_engine.workflow.configs import AnchorConfig`, monkeypatch `flows.ingest`, and keep the malformed-anchor assertions.

- [ ] **Step 2: Run the moved tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/configs tests/workflow/flows/test_ingest.py tests/workflow/tasks/test_stack.py tests/cli/commands/test_workflow.py -k ingest`
Expected: collection errors, `ModuleNotFoundError: No module named 'geosave_engine.workflow.configs'`.

- [ ] **Step 3: Create configs**

`configs/base.py` holds `ConfigModel` verbatim from `ingestion/anchor.py:12-21`. `configs/anchor.py` holds the three concrete configs and the `AnchorConfig` union verbatim, importing `ConfigModel` from `.base`.

```python
# configs/__init__.py
"""Deployment-safe invocation models for workflow flows."""

from .anchor import (
    AnchorConfig,
    CoordinateAnchorConfig,
    GeoJSONAnchorConfig,
    RasterAnchorConfig,
)

__all__ = [
    "AnchorConfig",
    "CoordinateAnchorConfig",
    "GeoJSONAnchorConfig",
    "RasterAnchorConfig",
]
```

- [ ] **Step 4: Create the shared stack writer**

```python
# tasks/stack.py
"""Atomically publish one named raster stack as a local Zarr store."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io


def write_stack(
    rasters: Mapping[str, xr.Dataset], output: str | Path, **options: Any
) -> str:
    """Write rasters fully before publishing a new local Zarr store.

    Args:
        rasters: Named, co-registered rasters.
        output: New local ``.zarr`` path.
        **options: Native Zarr writer options.

    Returns:
        Completed store path.

    Raises:
        ValueError: If the output is remote or not a ``.zarr`` path.
        FileExistsError: If the output exists before or after writing.
    """
    destination = Path(output)
    if "://" in str(output) or destination.suffix != ".zarr":
        raise ValueError("Output must be a local .zarr path")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = stack(rasters)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        io.zarr.write(tree, staged, compute=True, overwrite=False, **options)
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        staged.rename(destination)
    return str(destination)
```

`tasks/__init__.py` for now:

```python
"""Reusable workflow work units."""

__all__: list[str] = []
```

- [ ] **Step 5: Create the ingest flow and update the CLI**

```python
# flows/ingest.py
"""Ingest one model-ready raster stack on an explicit anchor."""

from prefect import flow
from pydantic import JsonValue, TypeAdapter

from geosave_engine.model_spec import ModelSpec
from geosave_engine.workflow.configs import AnchorConfig
from geosave_engine.workflow.tasks.stack import write_stack


@flow(name="ingest", persist_result=False)
def ingest(anchor: dict[str, JsonValue], *, output: str, spec: str) -> str:
    """<docstring verbatim from ingestion/flow.py:44-56>"""
    model = ModelSpec.load(spec)
    model.require_recipes()
    native_anchor = TypeAdapter(AnchorConfig).validate_python(anchor).open()
    return write_stack(model.load_rasters(native_anchor), output)
```

```python
# flows/__init__.py
"""Independently runnable Prefect flows."""

from .ingest import ingest

__all__ = ["ingest"]
```

CLI `ingest.py`: replace the two `workflow.ingestion` imports with `from geosave_engine.workflow import flows` and `from geosave_engine.workflow.configs import AnchorConfig`; call `flows.ingest(...)`. Then `git rm -r src/geosave_engine/workflow/ingestion`.

- [ ] **Step 6: Run the ingest slice**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/configs tests/workflow/flows/test_ingest.py tests/workflow/tasks/test_stack.py tests/cli/commands/test_workflow.py -k ingest`
Expected: PASS, including `test_ingest_requires_all_stac_recipes_before_opening_anchor` and every stack atomicity test.

- [ ] **Step 7: Lint and commit**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow tests/workflow tests/cli/commands/test_workflow.py && git diff --check`
Expected: PASS. (`workflow.training_data` still imports fine; it does not depend on `ingestion`.)

```bash
git add src/geosave_engine/workflow/configs src/geosave_engine/workflow/flows src/geosave_engine/workflow/tasks src/geosave_engine/workflow/ingestion src/geosave_engine/cli/commands/workflow/ingest.py tests/workflow/configs tests/workflow/flows tests/workflow/tasks tests/workflow/ingestion tests/cli/commands/test_workflow.py
git commit -m "refactor: restore layered ingest workflow"
```

### Task 3: Dense flow and preparation tasks

**Files:**
- Create: `src/geosave_engine/workflow/flows/prepare_dense_data.py`
- Create: `src/geosave_engine/workflow/tasks/dense.py`, `tasks/manifest.py`, `tasks/sample.py`
- Modify: `src/geosave_engine/workflow/flows/__init__.py`, `tasks/__init__.py`
- Modify: `src/geosave_engine/cli/commands/workflow/prepare_dense_data.py`
- Modify: `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py:11`
- Delete: `src/geosave_engine/workflow/training_data/` (all four files)
- Move: `tests/workflow/training_data/test_dense.py` → `tests/workflow/flows/test_prepare_dense_data.py`
- Move: `tests/workflow/training_data/test_sample_preparation.py` → `tests/workflow/tasks/test_dense.py`
- Move: `tests/workflow/training_data/test_manifest.py` → `tests/workflow/tasks/test_manifest.py`
- Move: `tests/workflow/training_data/test_sample.py` → `tests/workflow/tasks/test_sample.py`
- Merge then delete: `tests/workflow/training_data/test_metadata.py` → append into `tests/workflow/tasks/test_manifest.py`
- Modify: `tests/cli/commands/test_workflow.py`, `tests/workflow/test_ingest_script.py`, `tests/cli/core/test_workspace.py`

**Interfaces:**
- Consumes: `ModelSpec.require_recipes()` (Task 1); `write_stack(rasters, output, **options)` (Task 2); `workflow.flows`, `workflow.tasks` packages (Task 2).
- Produces: Prefect task `prepare_dense_sample(label: str | Path, model: ModelSpec, output: str | Path, *, format: SampleFormat = "geotiff", write_options: Mapping[str, JsonValue] | None = None) -> str`; Prefect flow `prepare_dense_data(labels: str, *, output: str, spec: str, pattern: str = "**/*.tif", max_concurrency: PositiveInt = 1, format: Literal["geotiff", "zarr"] = "geotiff", write_options: dict[str, JsonValue] | None = None, metadata: str | None = None) -> str`; `tasks.sample`: `SampleFormat`, `open_sample`, `write_sample`; `tasks.manifest`: `find_labels`, `sample_path(root: Path, sample_id: str, format: SampleFormat) -> Path`, `read_sample_metadata`, `write_manifest`.

- [ ] **Step 1: Move and merge tests**

```bash
git mv tests/workflow/training_data/test_dense.py tests/workflow/flows/test_prepare_dense_data.py
git mv tests/workflow/training_data/test_sample_preparation.py tests/workflow/tasks/test_dense.py
git mv tests/workflow/training_data/test_manifest.py tests/workflow/tasks/test_manifest.py
git mv tests/workflow/training_data/test_sample.py tests/workflow/tasks/test_sample.py
```

Append the body of `test_metadata.py` (helpers `_labels`, `_table`, and all `test_read_sample_metadata_*` tests) to `tests/workflow/tasks/test_manifest.py`, merging its imports (`pandas as pd`, `read_sample_metadata`) into the top import block; then `git rm tests/workflow/training_data/test_metadata.py`.

Move `test_find_labels_preserves_tree_and_removes_only_the_final_suffix` and `test_sample_path_preserves_the_suffix_free_identity` from `test_prepare_dense_data.py` into `test_manifest.py`, calling public `find_labels` and `sample_path` imported from `geosave_engine.workflow.tasks.manifest`.

Update remaining imports:

```python
# tests/workflow/flows/test_prepare_dense_data.py
from geosave_engine.workflow import flows, tasks
flow_module = import_module("geosave_engine.workflow.flows.prepare_dense_data")

# tests/workflow/tasks/test_dense.py
from geosave_engine.workflow.tasks.sample import open_sample, write_sample
dense_module = import_module("geosave_engine.workflow.tasks.dense")

# tests/workflow/tasks/test_manifest.py
from geosave_engine.workflow.tasks.manifest import (
    find_labels,
    read_sample_metadata,
    sample_path,
    write_manifest,
)
from geosave_engine.workflow.tasks.sample import SampleFormat, write_sample

# tests/workflow/tasks/test_sample.py
import geosave_engine.workflow.tasks.sample as save_module
```

In `test_prepare_dense_data.py`, `flow_module.prepare_dense_sample` still resolves because the flow module imports the task by name; keep those monkeypatches.

Replace `test_training_data_exports_only_public_operations` with:

```python
def test_layers_export_only_supported_operations() -> None:
    from prefect import Flow, Task

    assert flows.__all__ == ["ingest", "prepare_dense_data"]
    assert tasks.__all__ == ["prepare_dense_sample"]
    assert isinstance(flows.ingest, Flow)
    assert isinstance(flows.prepare_dense_data, Flow)
    assert isinstance(tasks.prepare_dense_sample, Task)
    assert not hasattr(tasks, "write_manifest")
    assert not hasattr(tasks, "write_sample")
```

Add to `tests/workflow/tasks/test_sample.py` a Zarr option-forwarding test:

```python
def test_write_sample_forwards_zarr_options_to_the_stack_writer(raw, tmp_path, monkeypatch):
    captured = {}

    def write(rasters, output, **options):
        captured.update(options)
        return str(output)

    monkeypatch.setattr(save_module, "write_stack", write)

    save_module.write_sample(
        raw, tmp_path / "a.zarr", format="zarr", write_options={"consolidated": True}
    )

    assert captured == {"consolidated": True}
```

CLI test: monkeypatch `flows.prepare_dense_data`. `test_ingest_script.py` and `test_workspace.py`: require `from geosave_engine.workflow.flows import prepare_dense_data`.

- [ ] **Step 2: Run to verify collection fails**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow tests/cli/commands/test_workflow.py tests/cli/core/test_workspace.py`
Expected: collection errors for `geosave_engine.workflow.tasks.sample` / `.manifest` / `.dense` / `flows.prepare_dense_data`.

- [ ] **Step 3: Move sample storage onto the shared writer**

`git mv src/geosave_engine/workflow/training_data/sample.py src/geosave_engine/workflow/tasks/sample.py`. Delete `_write_zarr_sample` and its now-unused imports (`TemporaryDirectory` stays for GeoTIFF). Add `from .stack import write_stack` and change the Zarr branch of `write_sample`:

```python
    if format == "zarr":
        return write_stack(rasters, output, **options)
```

Everything else (`SampleFormat`, `_PUBLICATION_OPTIONS`, GeoTIFF path, `open_sample`, `_geotiff_scene`) stays verbatim.

- [ ] **Step 4: Move manifest and tighten `sample_path`**

`git mv src/geosave_engine/workflow/training_data/manifest.py src/geosave_engine/workflow/tasks/manifest.py`. Import stays `from .sample import SampleFormat, open_sample`. Change only:

```python
def sample_path(root: Path, sample_id: str, format: SampleFormat) -> Path:
```

and drop `Literal` from the `typing` import.

- [ ] **Step 5: Create the dense task without the redundant copy**

`tasks/dense.py`: move `_validate_dense_sample` and `prepare_dense_sample` verbatim from `training_data/dense.py:24-79`, importing `open_sample`, `write_sample`, `SampleFormat` from `.sample`, and delete this line:

```python
    model = ModelSpec.model_validate(model.model_dump())
```

`load_rasters` already validates its own copy, and Prefect passes the submitted object unchanged.

```python
# tasks/__init__.py
"""Reusable workflow work units."""

from .dense import prepare_dense_sample

__all__ = ["prepare_dense_sample"]
```

- [ ] **Step 6: Create the dense flow with one submission loop**

```python
# flows/prepare_dense_data.py
"""Prepare bounded, label-aligned dense training samples."""

from pathlib import Path
from typing import Literal

from prefect import flow
from prefect.futures import as_completed
from pydantic import JsonValue, PositiveInt, TypeAdapter

from geosave_engine.model_spec import ModelSpec
from geosave_engine.workflow.tasks import prepare_dense_sample
from geosave_engine.workflow.tasks.manifest import (
    find_labels,
    read_sample_metadata,
    sample_path,
    write_manifest,
)


@flow(name="prepare-dense-data", persist_result=False)
def prepare_dense_data(
    labels: str,
    *,
    output: str,
    spec: str,
    pattern: str = "**/*.tif",
    max_concurrency: PositiveInt = 1,
    format: Literal["geotiff", "zarr"] = "geotiff",
    write_options: dict[str, JsonValue] | None = None,
    metadata: str | None = None,
) -> str:
    """<docstring verbatim from training_data/dense.py:94-113>"""
    limit = TypeAdapter(PositiveInt).validate_python(max_concurrency)
    model = ModelSpec.load(spec)
    if "label" in model.rasters:
        raise ValueError("Model raster name 'label' is reserved")
    model.require_recipes()

    destination = Path(output)
    discovered = find_labels(Path(labels), pattern)
    properties = read_sample_metadata(metadata, discovered)
    pending = {}
    completed = {}
    for sample_id, label in discovered.items():
        if len(pending) == limit:
            # A failed result raises here, so no further samples are submitted.
            future = next(as_completed(list(pending)))
            completed[pending.pop(future)] = future.result()
        future = prepare_dense_sample.submit(
            label,
            model,
            sample_path(destination, sample_id, format),
            format=format,
            write_options=write_options,
        )
        pending[future] = sample_id
    for future in as_completed(list(pending)):
        completed[pending[future]] = future.result()

    ordered = {sample_id: completed[sample_id] for sample_id in discovered}
    return write_manifest(
        ordered,
        destination / "manifest.parquet",
        format=format,
        metadata=properties,
    )
```

```python
# flows/__init__.py
"""Independently runnable Prefect flows."""

from .ingest import ingest
from .prepare_dense_data import prepare_dense_data

__all__ = ["ingest", "prepare_dense_data"]
```

Then `git rm -r src/geosave_engine/workflow/training_data`.

- [ ] **Step 7: Update CLI and template consumers**

CLI `prepare_dense_data.py`: `from geosave_engine.workflow import flows`; call `flows.prepare_dense_data(...)`. Template `ingest_imagery.py:11`: `from geosave_engine.workflow.flows import prepare_dense_data`. No command name, option, default, or output change.

- [ ] **Step 8: Run the dense and consumer suite**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow tests/model_spec tests/cli/commands/test_workflow.py tests/cli/core/test_workspace.py`
Expected: PASS, including `test_prepare_dense_data_stops_submitting_after_failure`, bounded concurrency, metadata-before-submit, unchanged-manifest failure, resume validation, and both persistence formats.

- [ ] **Step 9: Lint and commit**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow src/geosave_engine/templates tests/workflow tests/cli && git diff --check`
Expected: PASS.

```bash
git add src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow/prepare_dense_data.py src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py tests/workflow tests/cli/commands/test_workflow.py tests/cli/core/test_workspace.py
git commit -m "refactor: restore layered dense preparation"
```

### Task 4: Documentation, stale-path guard, and clean package

**Files:**
- Modify: `src/geosave_engine/workflow/__init__.py`
- Modify: `docs/guides/workflows.md:231-250`
- Test: `tests/workflow/flows/test_prepare_dense_data.py`

**Interfaces:**
- Consumes: final layered paths from Tasks 2–3.
- Produces: active docs and packaged source naming only `workflow.configs`, `workflow.flows`, `workflow.tasks`.

- [ ] **Step 1: Add the stale-import guard**

```python
def test_active_files_do_not_reference_removed_workflow_packages() -> None:
    root = Path(__file__).parents[3]
    sources = [
        *root.joinpath("src/geosave_engine").rglob("*.py"),
        *root.joinpath("tests").rglob("*.py"),
        *root.joinpath("docs/guides").rglob("*.md"),
        root / "README.md",
    ]
    removed = ("geosave_engine.workflow.ingestion", "geosave_engine.workflow.training_data")
    stale = [
        str(path.relative_to(root))
        for path in sources
        if "__pycache__" not in path.parts
        and path != Path(__file__)
        and any(name in path.read_text() for name in removed)
    ]
    assert stale == []
```

- [ ] **Step 2: Run it to verify it fails on the guide**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q tests/workflow/flows/test_prepare_dense_data.py::test_active_files_do_not_reference_removed_workflow_packages`
Expected: FAIL listing `docs/guides/workflows.md`.

- [ ] **Step 3: Update docs and package docstring**

`workflow/__init__.py`: `"""Deployment configs, reusable tasks, and runnable Prefect flows."""`

`docs/guides/workflows.md` "Python equivalent": import both flows from `geosave_engine.workflow.flows`; mention `geosave_engine.workflow.configs.AnchorConfig` for anchor mappings. Do not edit historical `docs/superpowers` records.

- [ ] **Step 4: Stale-path, lint, and type gates**

Run: `rg -n "geosave_engine\.workflow\.(ingestion|training_data)" README.md docs/guides src tests`
Expected: only the guard test's own string literals.

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run ruff check src tests && UV_CACHE_DIR=/tmp/geosave-uv-cache uv run basedpyright src/geosave_engine/workflow src/geosave_engine/cli/commands/workflow src/geosave_engine/model_spec/model.py`
Expected: Ruff passes; BasedPyright 0 errors.

- [ ] **Step 5: Full verification**

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv run pytest -q`
Expected: complete non-slow suite passes.

Run: `UV_CACHE_DIR=/tmp/geosave-uv-cache uv build`
Expected: sdist and wheel build. If sandbox DNS blocks Hatchling, rerun with network permission; do not change dependencies.

Clean-export smoke:

```bash
tmp=$(mktemp -d) && git archive HEAD | tar -x -C "$tmp"
PYTHONPATH="$tmp/src" .venv/bin/python -c "from geosave_engine.workflow.configs import AnchorConfig; from geosave_engine.workflow.flows import ingest, prepare_dense_data; from geosave_engine.workflow.tasks import prepare_dense_sample; print('workflow imports ok')"
unzip -l dist/geosave_engine-*.whl | grep -E "workflow/(ingestion|training_data)/" && echo STALE || echo clean
```

Expected: `workflow imports ok` and `clean`.

- [ ] **Step 6: Commit**

```bash
git add src/geosave_engine/workflow/__init__.py docs/guides/workflows.md tests/workflow/flows/test_prepare_dense_data.py
git commit -m "docs: align layered workflow API"
```

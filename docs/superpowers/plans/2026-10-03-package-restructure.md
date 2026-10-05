# Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `geosave_engine` into `model/` (what a release contains) and `ml/` (Lightning training), with one package per head type and training method, without changing behaviour.

**Architecture:** Files move with `git mv` and only their import lines change. `model/` gathers the chain, registry, stage modules, model spec, and release code. `ml/` keeps training: builders, callbacks, datasets, and `segmentation/supervised/` holding the matched `Module` and `DataModule`. A new `tests/test_layering.py` pins the dependency direction and proves no old import path survives.

**Tech Stack:** Python 3.12, PyTorch, Lightning 2.6.1, Pydantic 2, pytest, ruff, uv

**Spec:** `docs/superpowers/specs/2026-10-03-package-restructure-design.md`

## Global Constraints

- No behaviour change. Besides import lines, the only edits are the three listed in the spec under "Changes that are not pure moves".
- No compatibility alias for any old import path.
- `src/geosave_engine/model/__init__.py` stays empty.
- `geosave_engine.geodata` loads no `torch`. `geosave_engine.model.spec` and `geosave_engine.workflow.flows` load no `torch`. `geosave_engine.model.chain`, `.registry`, `.head`, and `.release` load no `lightning`.
- Left untouched: `geodata/sensors/`, `geodata/viz/`, `utils/colorize.py`, `workspace/`, `AGENTS.md`, every `.ipynb`.
- The working tree holds the user's uncommitted work, including files this plan moves. Never run `git checkout --`, `git restore`, `git stash`, `git clean`, or `git add -A`. Undo your own edit by editing it back.
- Use `git mv` for tracked paths and plain `mv` for untracked ones. `git mv` on a directory carries its untracked files along.
- Rename identifiers one file at a time: list the sites with `grep -n`, replace inside that file only, then confirm the count is zero. No multi-file `sed`.
- Remove a leftover source directory only after `find <dir> -type f -not -name '*.pyc'` prints nothing.
- Run tests with `uv run pytest`. Tests marked `slow` are deselected by default; run them with `-m slow` where a step says so.
- Do not commit. The user decided on 2026-10-03 that the moves stay in the working tree on top of their uncommitted work. Every Commit step is skipped.

## Review Focus

- A leftover directory holding only `__pycache__` makes an old path importable as a namespace package. `test_old_path_is_gone` in every task fails until the directory is removed.
- Registration used to rely on a package walk in `ml/models/__init__.py`. Task 2 pins, in a fresh interpreter, that every registered name is still listed.
- An import added to `model/__init__.py` would pull torch into `geosave workflow ingest`. Task 3 pins that `model.spec` and `workflow.flows` load no torch.
- A generated workspace must name the new class paths in `train.yaml` and `main.py`. Task 4 runs `tests/cli/core/test_workspace.py`, including its `slow` tests.
- `tests/model_spec/test_examples.py` finds the repo root with `Path(__file__).parents[2]`. One directory deeper it must be `parents[3]`. Task 3 fixes both sites and runs the file.

## File Structure

| New path | Was |
| --- | --- |
| `src/geosave_engine/ml/datasets/` | `geodata/datasets/` |
| `src/geosave_engine/model/__init__.py` | new, empty |
| `src/geosave_engine/model/chain/` | `ml/model_chain/` |
| `src/geosave_engine/model/factory.py` | `ml/registry/factory.py` |
| `src/geosave_engine/model/registry.py` | `ml/registry/model.py` |
| `src/geosave_engine/model/{encoder,decoder,head,monolith}/` | `ml/models/{...}/` |
| `src/geosave_engine/model/encoder/time.py` | `ml/encoding/time.py` |
| `src/geosave_engine/model/README.md` | `ml/models/README.md` |
| `src/geosave_engine/model/spec/` | `model_spec/` |
| `src/geosave_engine/model/release/` | `release/` |
| `src/geosave_engine/ml/builders/` | `ml/registry/` (criterion, optimizer, scheduler) |
| `src/geosave_engine/ml/cli.py` | `ml/lightning/cli.py` |
| `src/geosave_engine/ml/segmentation/metrics.py` | `ml/metrics/semantic_segmentation.py` |
| `src/geosave_engine/ml/segmentation/transforms.py` | `ml/transforms/semantic_segmentation.py` |
| `src/geosave_engine/ml/segmentation/calibrate.py` | `ml/callbacks/threshold_calibrator.py` |
| `src/geosave_engine/ml/segmentation/supervised/module.py` | `ml/lightning/tasks/semantic_segmentation.py` |
| `src/geosave_engine/ml/segmentation/supervised/data.py` | `ml/lightning/data/semantic_segmentation.py` |
| `tests/test_layering.py` | new |

Tests move to mirror their sources. `tests/ml/callbacks/test_task_callbacks.py` stays where it is; it tests both callbacks together and is not split here.

---

### Task 0: Baseline and precondition

**Files:** none changed.

- [ ] **Step 1: Note the commit decision**

Decided 2026-10-03: the user is still designing, so the moves are made on top of the uncommitted work and nothing is committed. Skip every Commit step in this plan.

- [ ] **Step 2: Record the baseline**

Run: `uv run pytest tests/ml tests/model_spec tests/release tests/cli tests/workflow tests/geodata/datasets -q`
Expected: `507 passed, 9 deselected` (the count on 2026-10-03; if the user committed new work since, record the new count).

Run: `uv run pytest -q`
Record the pass and fail counts. Any failure here predates this plan and must be unchanged at the end.

---

### Task 1: Move `TileDataset` into `ml`

**Files:**
- Create: `tests/test_layering.py`
- Move: `src/geosave_engine/geodata/datasets/` to `src/geosave_engine/ml/datasets/`
- Move: `tests/geodata/datasets/` to `tests/ml/datasets/`
- Modify: `tests/ml/datasets/test_tiles.py`, `tests/ml/lightning/tasks/test_semantic_segmentation.py`, `tests/ml/models/encoder/test_model_context.py`, `src/geosave_engine/ml/models/README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `geosave_engine.ml.datasets.TileDataset`, unchanged signature. `tests/test_layering.py` with helpers `_run(code: str) -> str` and `_loads(imports: str, module: str) -> bool`, and the parametrized `test_old_path_is_gone(path)` that later tasks extend.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layering.py`:

```python
"""Package dependency direction and removed import paths."""

import importlib.util
import subprocess
import sys

import pytest


def _run(code: str) -> str:
    """Return the last line a fresh interpreter prints for `code`."""
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def _loads(imports: str, module: str) -> bool:
    """Report whether importing `imports` also loads `module`."""
    return _run(f"import sys\nimport {imports}\nprint({module!r} in sys.modules)") == "True"


@pytest.mark.parametrize(
    "path",
    [
        "geosave_engine.geodata.datasets",
    ],
)
def test_old_path_is_gone(path):
    assert importlib.util.find_spec(path) is None


@pytest.mark.slow
def test_geodata_loads_no_torch():
    assert not _loads("geosave_engine.geodata", "torch")
```

- [ ] **Step 2: Run them to verify the state before the move**

Run: `uv run pytest tests/test_layering.py -m "slow or not slow" -v`
Expected: `test_old_path_is_gone[geosave_engine.geodata.datasets]` FAILS (the package still exists). `test_geodata_loads_no_torch` PASSES; it pins a rule that already holds.

- [ ] **Step 3: Move the files**

```bash
git mv src/geosave_engine/geodata/datasets src/geosave_engine/ml/datasets
mkdir -p tests/ml/datasets
git mv tests/geodata/datasets/test_tiles.py tests/ml/datasets/test_tiles.py
find src/geosave_engine/geodata/datasets tests/geodata/datasets -type f -not -name '*.pyc' 2>/dev/null
```

The `find` must print nothing. Then remove any leftover directory it searched: `rm -r src/geosave_engine/geodata/datasets tests/geodata/datasets 2>/dev/null`.

- [ ] **Step 4: Update the imports**

| File | Old | New |
| --- | --- | --- |
| `tests/ml/datasets/test_tiles.py` | `from geosave_engine.geodata.datasets import TileDataset` | `from geosave_engine.ml.datasets import TileDataset` |
| `tests/ml/lightning/tasks/test_semantic_segmentation.py` (line 571, indented) | `from geosave_engine.geodata.datasets import TileDataset` | `from geosave_engine.ml.datasets import TileDataset` |
| `tests/ml/models/encoder/test_model_context.py` | `from geosave_engine.geodata.datasets import TileDataset` | `from geosave_engine.ml.datasets import TileDataset` |
| `src/geosave_engine/ml/models/README.md` (line 165) | `from geosave_engine.geodata.datasets import TileDataset` | `from geosave_engine.ml.datasets import TileDataset` |

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_layering.py tests/ml tests/geodata -m "slow or not slow" -q -k "layering or tiles or model_context or semantic_segmentation"`
Expected: PASS, including `test_old_path_is_gone`.

Run: `grep -rn "geodata.datasets" src tests docs/guides`
Expected: no output.

- [ ] **Step 6: Commit** (only if Task 0 allows)

```bash
git add tests/test_layering.py
git commit -m "refactor: move TileDataset into ml" -- tests/test_layering.py src/geosave_engine/geodata/datasets src/geosave_engine/ml/datasets tests/geodata/datasets tests/ml/datasets tests/ml/lightning/tasks/test_semantic_segmentation.py tests/ml/models/encoder/test_model_context.py src/geosave_engine/ml/models/README.md
```

---

### Task 2: Create `model/` with the chain, registry, and stage modules

**Files:**
- Create: `src/geosave_engine/model/__init__.py` (empty)
- Move: `ml/model_chain/` to `model/chain/`; `ml/registry/factory.py` to `model/factory.py`; `ml/registry/model.py` to `model/registry.py`; `ml/models/{encoder,decoder,head,monolith}/` to `model/`; `ml/encoding/time.py` to `model/encoder/time.py`; `ml/models/README.md` to `model/README.md`
- Delete: `ml/models/__init__.py`, `ml/encoding/__init__.py`, `tests/ml/models/encoder/test_clay.py` (empty file)
- Modify: `ml/registry/__init__.py`, `ml/registry/{criterion,optimizer,scheduler}.py`, `ml/lightning/tasks/semantic_segmentation.py`, `release/artifact.py`, `release/huggingface.py`, every moved stage module
- Test: `tests/model/` (moved from `tests/ml/`), `tests/test_layering.py`

**Interfaces:**
- Consumes: `tests/test_layering.py` helpers from Task 1.
- Produces:
  - `geosave_engine.model.chain`: `ModelChain`, `Published`, `chain_step`
  - `geosave_engine.model.factory`: `BuildSpec`
  - `geosave_engine.model.registry`: `register_model`, `list_models`, `build_model`, `MODEL_REGISTRY`
  - `geosave_engine.model.{encoder,decoder,head,monolith}`: the same classes as before
  - `geosave_engine.model.encoder.time`: `time_labels`
  - `geosave_engine.ml.registry` now exports only `CriterionSpec`, `OptimizerSpec`, `SchedulerSpec`, `build_criterion`, `build_optimizer`, `build_scheduler` (renamed to `ml.builders` in Task 4).

- [ ] **Step 1: Write the failing tests**

In `tests/test_layering.py`, extend the `path` list and add two tests:

```python
@pytest.mark.parametrize(
    "path",
    [
        "geosave_engine.geodata.datasets",
        "geosave_engine.ml.model_chain",
        "geosave_engine.ml.models",
        "geosave_engine.ml.encoding",
    ],
)
def test_old_path_is_gone(path):
    assert importlib.util.find_spec(path) is None
```

```python
@pytest.mark.slow
def test_model_core_loads_no_lightning():
    imports = "geosave_engine.model.chain, geosave_engine.model.registry, geosave_engine.model.head"
    assert not _loads(imports, "lightning")


@pytest.mark.slow
def test_every_stage_package_registers_its_models():
    listed = _run(
        "from geosave_engine.model.registry import list_models\nprint(list_models())"
    )
    assert ast.literal_eval(listed) == {
        "decoder": ["DPT", "UNET"],
        "encoder": ["CLAY", "DINOV3", "PRITHVI", "PRITHVI_TL"],
        "head": ["CLASSIFICATION", "DETECTION", "REGRESSION", "SEGMENTATION"],
        "model": ["IBM_GRANITE_BIOMASS"],
    }
```

Add `import ast` at the top of the file.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_layering.py -m "slow or not slow" -v`
Expected: the three new `test_old_path_is_gone` cases FAIL. The two new slow tests FAIL with `CalledProcessError` (no module `geosave_engine.model`).

- [ ] **Step 3: Move the source files**

```bash
S=src/geosave_engine
mkdir -p $S/model
: > $S/model/__init__.py
git add $S/model/__init__.py
git mv $S/ml/model_chain $S/model/chain
git mv $S/ml/registry/factory.py $S/model/factory.py
git mv $S/ml/registry/model.py $S/model/registry.py
git mv $S/ml/models/encoder $S/model/encoder
git mv $S/ml/models/decoder $S/model/decoder
git mv $S/ml/models/head $S/model/head
git mv $S/ml/models/monolith $S/model/monolith
git mv $S/ml/encoding/time.py $S/model/encoder/time.py
git mv $S/ml/models/README.md $S/model/README.md
git rm -q $S/ml/models/__init__.py $S/ml/encoding/__init__.py
ls $S/model/head
find $S/ml/models $S/ml/encoding -type f -not -name '*.pyc' 2>/dev/null
```

`ls` must show `classification.py dense.py detection.py regression.py segmentation.py __init__.py`; the four untracked heads travel with the directory. The `find` must print nothing; then `rm -r $S/ml/models $S/ml/encoding 2>/dev/null`.

- [ ] **Step 4: Replace the registration walk**

In `src/geosave_engine/model/registry.py`, both `list_models` and `build_model` begin with:

```python
    import geosave_engine.ml.models  # noqa: F401
```

Replace each with:

```python
    from geosave_engine.model import decoder, encoder, head, monolith  # noqa: F401
```

Each stage package's `__init__.py` already imports every class it holds, so importing the four packages runs every `@register_model`.

- [ ] **Step 5: Update source imports**

Apply these replacements in each file that contains the old line:

| Old | New |
| --- | --- |
| `from geosave_engine.ml.model_chain import` | `from geosave_engine.model.chain import` |
| `from geosave_engine.ml.model_chain.published import published_kwargs` | `from geosave_engine.model.chain.published import published_kwargs` |
| `from geosave_engine.ml.registry import register_model` | `from geosave_engine.model.registry import register_model` |
| `from geosave_engine.ml.registry.factory import BuildSpec` | `from geosave_engine.model.factory import BuildSpec` |
| `from geosave_engine.ml.encoding.time import time_labels` | `from geosave_engine.model.encoder.time import time_labels` |
| `from geosave_engine.ml.registry import build_model` | `from geosave_engine.model.registry import build_model` |

Files, one at a time:

- `model/registry.py`: chain, published, factory lines.
- `model/encoder/clay.py`, `model/encoder/prithvi.py`: registry, chain, time lines.
- `model/encoder/dinov3.py`, `model/decoder/dpt.py`, `model/decoder/unet.py`, `model/head/classification.py`, `model/head/detection.py`, `model/monolith/ibm_granite_biomass.py`: registry and chain lines.
- `model/head/dense.py`: chain line.
- `model/head/regression.py`, `model/head/segmentation.py`: registry line.
- `ml/registry/criterion.py`, `ml/registry/optimizer.py`, `ml/registry/scheduler.py`: factory line.
- `release/artifact.py`, `release/huggingface.py`: chain line; `huggingface.py` also the `build_model` line.

Replace `src/geosave_engine/ml/registry/__init__.py` with:

```python
from .criterion import CriterionSpec, build_criterion
from .optimizer import OptimizerSpec, build_optimizer
from .scheduler import SchedulerSpec, build_scheduler

__all__ = [
    "CriterionSpec",
    "OptimizerSpec",
    "SchedulerSpec",
    "build_criterion",
    "build_optimizer",
    "build_scheduler",
]
```

In `src/geosave_engine/ml/lightning/tasks/semantic_segmentation.py`, replace

```python
from geosave_engine.ml.model_chain import ModelChain
from geosave_engine.ml.registry import (
    build_criterion,
    build_model,
    build_optimizer,
    build_scheduler,
)
```

with

```python
from geosave_engine.ml.registry import (
    build_criterion,
    build_optimizer,
    build_scheduler,
)
from geosave_engine.model.chain import ModelChain
from geosave_engine.model.registry import build_model
```

- [ ] **Step 6: Move the tests**

```bash
mkdir -p tests/model/encoder
git mv tests/ml/model_chain tests/model/chain
git mv tests/ml/registry/test_model.py tests/model/test_registry.py
git mv tests/ml/models/encoder/test_model_context.py tests/model/encoder/test_model_context.py
git mv tests/ml/encoding/test_time.py tests/model/encoder/test_time.py
git rm -q tests/ml/models/encoder/test_clay.py
mv tests/ml/models/head tests/model/head
```

`tests/ml/models/head` is untracked, so it uses plain `mv`.

- [ ] **Step 7: Update test imports**

Apply the Step 5 table, plus `geosave_engine.ml.models.` to `geosave_engine.model.` for stage classes, in:

- `tests/model/chain/conftest.py`, `test_chain.py`, `test_published.py` (also `geosave_engine.ml.model_chain.published` to `geosave_engine.model.chain.published`), `test_step.py` (also `geosave_engine.ml.model_chain.step`, and the two string literals at lines 234 and 279: `"from geosave_engine.ml.model_chain import chain_step\n"` to `"from geosave_engine.model.chain import chain_step\n"`).
- `tests/model/test_registry.py`: replace

  ```python
  from geosave_engine.ml.registry import BuildSpec, build_model, list_models, register_model
  from geosave_engine.ml.registry.model import MODEL_REGISTRY
  ```

  with

  ```python
  from geosave_engine.model.factory import BuildSpec
  from geosave_engine.model.registry import (
      MODEL_REGISTRY,
      build_model,
      list_models,
      register_model,
  )
  ```

- `tests/model/encoder/test_model_context.py`: `geosave_engine.ml.registry import build_model`, `geosave_engine.ml.models.encoder.clay`, `geosave_engine.ml.models.encoder.prithvi`.
- `tests/model/encoder/test_time.py`: the `time_labels` line.
- `tests/model/head/test_classification.py`, `test_detection.py`, `test_regression.py`, `test_segmentation.py`: chain line; `geosave_engine.ml.models.head.<name>` to `geosave_engine.model.head.<name>`; `from geosave_engine.ml.registry import ...` to `from geosave_engine.model.registry import ...`.
- `tests/ml/registry/test_builders.py`: replace

  ```python
  from geosave_engine.ml.registry import (
      BuildSpec,
      build_criterion,
      build_optimizer,
      build_scheduler,
  )
  ```

  with

  ```python
  from geosave_engine.ml.registry import (
      build_criterion,
      build_optimizer,
      build_scheduler,
  )
  from geosave_engine.model.factory import BuildSpec
  ```

- `tests/ml/callbacks/test_task_callbacks.py`, `tests/ml/lightning/tasks/test_semantic_segmentation.py`: chain line.
- `tests/release/test_artifact.py`: chain line and `build_model` line.
- `tests/release/test_huggingface.py`: chain line, `build_model` line, and `geosave_engine.ml.models.head.segmentation` to `geosave_engine.model.head.segmentation`.

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_layering.py tests/model tests/ml tests/release -m "slow or not slow" -q`
Expected: PASS.

Run: `grep -rnE "ml\.(model_chain|models|encoding)|ml\.registry(\.model|\.factory| import (BuildSpec|build_model|list_models|register_model))" src tests --include=*.py`
Expected: no output.

Run: `uv run ruff check src/geosave_engine/model src/geosave_engine/ml tests/model tests/test_layering.py`
Expected: no new findings.

- [ ] **Step 9: Commit** (only if Task 0 allows)

```bash
git add tests/model/head src/geosave_engine/model
git commit -m "refactor: gather chain, registry, and stage modules under model" -- src/geosave_engine/model src/geosave_engine/ml src/geosave_engine/release tests/model tests/ml tests/release tests/test_layering.py
```

---

### Task 3: Move the model spec and release under `model/`

**Files:**
- Move: `src/geosave_engine/model_spec/` to `src/geosave_engine/model/spec/`; `src/geosave_engine/release/` to `src/geosave_engine/model/release/`; `tests/model_spec/` to `tests/model/spec/`; `tests/release/` to `tests/model/release/`
- Modify: `model/release/artifact.py`, `workflow/flows/ingest.py`, `workflow/flows/prepare_dense_data.py`, `workflow/tasks/dense.py`, `templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py`, `templates/tasks/semantic_segmentation/supervised/README.md`, `model/README.md`, `docs/guides/workflows.md`, `tests/conftest.py`, `tests/cli/core/test_workspace.py`, three `tests/workflow` files, every moved test
- Test: `tests/test_layering.py`

**Interfaces:**
- Consumes: `geosave_engine.model.chain`, `geosave_engine.model.registry` from Task 2.
- Produces: `geosave_engine.model.spec` with the same exports as `model_spec` (`ModelSpec`, `RasterRequirement`, `StacRecipe`, `StageSpec`, `CallSpec`, `Ref`, ...). `geosave_engine.model.release` with `load_model`, `load_spec`, `publish_model`, `save_model`; `geosave_engine.model.release.huggingface` with `GeoSaveConfig`, `GeoSaveModel`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_layering.py`, add to the `path` list:

```python
        "geosave_engine.model_spec",
        "geosave_engine.release",
```

and add:

```python
@pytest.mark.slow
def test_spec_and_workflow_load_no_torch():
    assert not _loads("geosave_engine.model.spec, geosave_engine.workflow.flows", "torch")


@pytest.mark.slow
def test_release_loads_no_lightning():
    assert not _loads("geosave_engine.model.release", "lightning")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_layering.py -m "slow or not slow" -v`
Expected: the two new path cases FAIL; the two new slow tests FAIL with `CalledProcessError`.

- [ ] **Step 3: Move the files**

```bash
S=src/geosave_engine
git mv $S/model_spec $S/model/spec
git mv $S/release $S/model/release
git mv tests/model_spec tests/model/spec
git mv tests/release tests/model/release
find $S/model_spec $S/release tests/model_spec tests/release -type f -not -name '*.pyc' 2>/dev/null
```

The `find` must print nothing; then `rm -r` any of those four directories that still exists.

- [ ] **Step 4: Update source imports**

| Old | New |
| --- | --- |
| `from geosave_engine.model_spec import` | `from geosave_engine.model.spec import` |
| `from geosave_engine.release import` | `from geosave_engine.model.release import` |

- `model/release/artifact.py`: the `ModelSpec` import, and the `from geosave_engine.release import load_model, load_spec` line inside the `_model_card` text.
- `workflow/flows/ingest.py`, `workflow/flows/prepare_dense_data.py`, `workflow/tasks/dense.py`: the `model_spec` import.
- `templates/tasks/semantic_segmentation/supervised/scripts/prepare_example.py`: the `model_spec` import.
- `templates/tasks/semantic_segmentation/supervised/README.md`: lines 20, 56, 69.
- `model/README.md`: the three `from geosave_engine.release import` lines.
- `docs/guides/workflows.md`: line 226.

Imports inside `model/spec/` and `model/release/` are relative and need no edit.

- [ ] **Step 5: Update test imports and paths**

- `tests/conftest.py`, `tests/cli/core/test_workspace.py`, `tests/workflow/flows/test_ingest.py`, `tests/workflow/flows/test_prepare_dense_data.py`, `tests/workflow/tasks/test_dense.py`: the `model_spec` import.
- `tests/model/spec/test_call.py`, `test_examples.py`, `test_execution.py`, `test_model.py`, `test_rasters.py`, `test_stac.py`, `test_stage.py`, `test_acquisition.py`: the `model_spec` import. `test_acquisition.py` also has `import geosave_engine.model_spec.stac as stac_module`, which becomes `import geosave_engine.model.spec.stac as stac_module`.
- `tests/model/spec/test_examples.py`: the file is now one directory deeper. Change `Path(__file__).parents[2]` to `Path(__file__).parents[3]` at both sites (lines 15 and 47).
- `tests/model/release/test_artifact.py`: the `ModelSpec` import, both `from geosave_engine.release import` lines (21 and 248), and the four patch targets: `"geosave_engine.release.artifact.hf_hub_download"` (twice), `"geosave_engine.release.huggingface.GeoSaveModel.save_pretrained"`, `"geosave_engine.release.artifact.HfApi"`. Each gains `model.` after `geosave_engine.`.
- `tests/model/release/test_huggingface.py`: both `from geosave_engine.release.huggingface import` lines.
- `tests/model/encoder/test_model_context.py`: `from geosave_engine.release.huggingface import GeoSaveModel`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_layering.py tests/model tests/workflow tests/cli tests/ml -m "slow or not slow" -q`
Expected: PASS.

Run: `grep -rnE "geosave_engine\.(model_spec|release)\b" src tests docs/guides`
Expected: no output.

- [ ] **Step 7: Commit** (only if Task 0 allows)

```bash
git commit -m "refactor: move model spec and release under model" -- src/geosave_engine/model_spec src/geosave_engine/release src/geosave_engine/model src/geosave_engine/workflow src/geosave_engine/templates docs/guides/workflows.md tests/model_spec tests/release tests/model tests/workflow tests/cli tests/conftest.py tests/test_layering.py
```

---

### Task 4: Reshape `ml/` around head type and training method

**Files:**
- Move: `ml/registry/` to `ml/builders/`; `ml/metrics/semantic_segmentation.py` to `ml/segmentation/metrics.py`; `ml/transforms/semantic_segmentation.py` to `ml/segmentation/transforms.py`; `ml/callbacks/threshold_calibrator.py` to `ml/segmentation/calibrate.py`; `ml/lightning/tasks/semantic_segmentation.py` to `ml/segmentation/supervised/module.py`; `ml/lightning/data/semantic_segmentation.py` to `ml/segmentation/supervised/data.py`; `ml/lightning/cli.py` to `ml/cli.py`
- Create: `ml/segmentation/__init__.py`, `ml/segmentation/supervised/__init__.py`
- Delete: `ml/metrics/__init__.py`, `ml/lightning/__init__.py`, `ml/lightning/tasks/__init__.py`, `ml/lightning/data/__init__.py`
- Modify: `ml/callbacks/__init__.py`, `ml/transforms/__init__.py`, `templates/common/main.py`, `templates/tasks/semantic_segmentation/supervised/configs/train.yaml`, `templates/tasks/semantic_segmentation/supervised/README.md`, `model/README.md`
- Test: moved tests under `tests/ml/`, `tests/cli/core/test_workspace.py`, `tests/test_layering.py`

**Interfaces:**
- Consumes: `geosave_engine.model.registry.build_model`, `geosave_engine.model.chain.ModelChain` from Task 2.
- Produces:
  - `geosave_engine.ml.builders`: `CriterionSpec`, `OptimizerSpec`, `SchedulerSpec`, `build_criterion`, `build_optimizer`, `build_scheduler`
  - `geosave_engine.ml.cli.GeosaveCLI`
  - `geosave_engine.ml.segmentation.supervised.Module` (was `SemanticSegmentationTask`, same constructor)
  - `geosave_engine.ml.segmentation.supervised.DataModule` (was `SemanticSegmentationDataModule`, same constructor)
  - `geosave_engine.ml.segmentation.metrics.SemanticSegmentationMetrics`
  - `geosave_engine.ml.segmentation.transforms`: `softmax_argmax`, `apply_thresholds`
  - `geosave_engine.ml.segmentation.calibrate.ThresholdCalibrator`
  - `geosave_engine.ml.callbacks.DensePredictionLogger`
  - `geosave_engine.ml.transforms.ImageAugmenter`, unchanged and not moved

- [ ] **Step 1: Write the failing tests**

In `tests/test_layering.py`, add to the `path` list:

```python
        "geosave_engine.ml.registry",
        "geosave_engine.ml.lightning",
        "geosave_engine.ml.metrics",
        "geosave_engine.ml.transforms.semantic_segmentation",
```

and add:

```python
def test_supervised_segmentation_pairs_its_module_and_data():
    from lightning import LightningDataModule, LightningModule

    from geosave_engine.ml.segmentation import supervised

    assert issubclass(supervised.Module, LightningModule)
    assert issubclass(supervised.DataModule, LightningDataModule)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_layering.py -v`
Expected: the four new path cases FAIL; the pairing test FAILS with `ModuleNotFoundError: No module named 'geosave_engine.ml.segmentation'`.

- [ ] **Step 3: Move and delete source files**

```bash
S=src/geosave_engine/ml
git mv $S/registry $S/builders
mkdir -p $S/segmentation/supervised
git mv $S/metrics/semantic_segmentation.py $S/segmentation/metrics.py
git mv $S/transforms/semantic_segmentation.py $S/segmentation/transforms.py
git mv $S/callbacks/threshold_calibrator.py $S/segmentation/calibrate.py
git mv $S/lightning/tasks/semantic_segmentation.py $S/segmentation/supervised/module.py
git mv $S/lightning/data/semantic_segmentation.py $S/segmentation/supervised/data.py
git mv $S/lightning/cli.py $S/cli.py
git rm -q $S/metrics/__init__.py \
  $S/lightning/__init__.py $S/lightning/tasks/__init__.py $S/lightning/data/__init__.py
find $S/registry $S/metrics $S/lightning -type f -not -name '*.pyc' 2>/dev/null
```

The `find` must print nothing; then `rm -r` any of those three directories that still exists. `ml/transforms/` stays: it keeps `augmenter.py`.

- [ ] **Step 4: Write the two package files**

Create `src/geosave_engine/ml/segmentation/__init__.py`:

```python
"""Training for segmentation heads, one package per method."""
```

Create `src/geosave_engine/ml/segmentation/supervised/__init__.py`:

```python
"""Supervised segmentation: the training module and the data that feeds it."""

from .data import DataModule
from .module import Module

__all__ = ["DataModule", "Module"]
```

Replace `src/geosave_engine/ml/callbacks/__init__.py` with:

```python
from .prediction_logger import DensePredictionLogger

__all__ = ["DensePredictionLogger"]
```

Replace `src/geosave_engine/ml/transforms/__init__.py` with:

```python
from .augmenter import ImageAugmenter

__all__ = ["ImageAugmenter"]
```

- [ ] **Step 5: Rename the two classes and fix source imports**

`ml/segmentation/supervised/module.py`:

- `class SemanticSegmentationTask(LightningModule):` becomes `class Module(LightningModule):`.
- In its docstring, `class_path: geosave_engine.ml.lightning.tasks.SemanticSegmentationTask` becomes `class_path: geosave_engine.ml.segmentation.supervised.Module`.
- `from geosave_engine.ml.metrics.semantic_segmentation import SemanticSegmentationMetrics` becomes `from geosave_engine.ml.segmentation.metrics import SemanticSegmentationMetrics`.
- `from geosave_engine.ml.registry import (` becomes `from geosave_engine.ml.builders import (`.

`ml/segmentation/supervised/data.py`:

- `class SemanticSegmentationDataModule(LightningDataModule):` becomes `class DataModule(LightningDataModule):`.
- The error text `"SemanticSegmentationDataModule supervised Dataset is not implemented yet"` becomes `"segmentation supervised Dataset is not implemented yet"`. The existing test matches `"supervised Dataset"`.

`ml/segmentation/calibrate.py`: `from geosave_engine.ml.transforms.semantic_segmentation import softmax_argmax` becomes `from geosave_engine.ml.segmentation.transforms import softmax_argmax`.

`ml/cli.py`: delete the stale first line `# geosave_engine/ml/cli/cli.py`.

`ml/builders/criterion.py`, `optimizer.py`, `scheduler.py`: no edit; they import `geosave_engine.model.factory` since Task 2.

Run: `grep -rn "SemanticSegmentationTask\|SemanticSegmentationDataModule" src/geosave_engine --include=*.py`
Expected: no output.

- [ ] **Step 6: Update the templates and READMEs**

`templates/common/main.py` line 1: `from geosave_engine.ml.lightning.cli import GeosaveCLI` becomes `from geosave_engine.ml.cli import GeosaveCLI`.

`templates/tasks/semantic_segmentation/supervised/configs/train.yaml`:

- line 20: `class_path: geosave_engine.ml.lightning.tasks.SemanticSegmentationTask` becomes `class_path: geosave_engine.ml.segmentation.supervised.Module`
- line 47: `class_path: geosave_engine.ml.lightning.data.SemanticSegmentationDataModule` becomes `class_path: geosave_engine.ml.segmentation.supervised.DataModule`

`templates/tasks/semantic_segmentation/supervised/README.md` line 55 and `model/README.md` line 114: `from geosave_engine.ml.lightning.tasks import SemanticSegmentationTask` becomes `from geosave_engine.ml.segmentation import supervised`. In each file, list the remaining sites with `grep -n SemanticSegmentationTask <file>` and change each `SemanticSegmentationTask` to `supervised.Module`.

`model/README.md` responsibility table (lines 34 to 39) becomes:

```markdown
| `chain/step.py` | Method declarations and runtime value checks |
| `chain/routing.py` | Method selection and dependency ordering |
| `chain/published.py` | Constructor attribute declarations and argument wiring |
| `chain/chain.py` | Registered modules and execution |
| `registry.py` | Factory registration, construction, and saved arguments |
| `release/` | Native release saving, loading, and Hub publication |
```

and its line `from geosave_engine.ml.registry import build_model` becomes `from geosave_engine.model.registry import build_model`.

- [ ] **Step 7: Move the tests**

```bash
mkdir -p tests/ml/builders tests/ml/segmentation/supervised
git mv tests/ml/registry/test_builders.py tests/ml/builders/test_builders.py
git mv tests/ml/metrics/test_semantic_segmentation.py tests/ml/segmentation/test_metrics.py
git mv tests/ml/transforms/test_semantic_segmentation.py tests/ml/segmentation/test_transforms.py
git mv tests/ml/lightning/tasks/test_semantic_segmentation.py tests/ml/segmentation/supervised/test_module.py
git mv tests/ml/lightning/data/test_semantic_segmentation.py tests/ml/segmentation/supervised/test_data.py
```

`test_module.py` keeps its depth, so its `Path(__file__).parents[4]` stays correct.

- [ ] **Step 8: Update test imports and class names**

Work through one file at a time. In each, first run `grep -n "SemanticSegmentationTask\|SemanticSegmentationDataModule" <file>`, replace within that file, then rerun the grep and expect no output.

- `tests/ml/builders/test_builders.py`: `from geosave_engine.ml.registry import (` becomes `from geosave_engine.ml.builders import (`.
- `tests/ml/segmentation/test_metrics.py`: `geosave_engine.ml.metrics.semantic_segmentation` becomes `geosave_engine.ml.segmentation.metrics`.
- `tests/ml/segmentation/test_transforms.py`: `geosave_engine.ml.transforms.semantic_segmentation` becomes `geosave_engine.ml.segmentation.transforms`.
- `tests/ml/segmentation/supervised/test_module.py` (24 sites):
  - `from geosave_engine.ml.lightning.cli import GeosaveCLI` becomes `from geosave_engine.ml.cli import GeosaveCLI`
  - `from geosave_engine.ml.lightning.tasks import SemanticSegmentationTask` becomes `from geosave_engine.ml.segmentation import supervised`
  - every other `SemanticSegmentationTask` becomes `supervised.Module`
  - the string `"geosave_engine.ml.lightning.cli.GeosaveCLI"` (line 363) becomes `"geosave_engine.ml.cli.GeosaveCLI"`
- `tests/ml/segmentation/supervised/test_data.py` (3 sites): `from geosave_engine.ml.lightning.data import SemanticSegmentationDataModule` becomes `from geosave_engine.ml.segmentation import supervised`; every other `SemanticSegmentationDataModule` becomes `supervised.DataModule`.
- `tests/ml/callbacks/test_task_callbacks.py` (2 sites): replace

  ```python
  from geosave_engine.ml.callbacks import DensePredictionLogger, ThresholdCalibrator
  from geosave_engine.ml.lightning.tasks import SemanticSegmentationTask
  ```

  with

  ```python
  from geosave_engine.ml.callbacks import DensePredictionLogger
  from geosave_engine.ml.segmentation import supervised
  from geosave_engine.ml.segmentation.calibrate import ThresholdCalibrator
  ```

  and the remaining `SemanticSegmentationTask` becomes `supervised.Module`.
- `tests/ml/test_cli.py`: `from geosave_engine.ml.lightning.cli import GeosaveCLI` becomes `from geosave_engine.ml.cli import GeosaveCLI`.
- `tests/cli/core/test_workspace.py` (7 sites): replace

  ```python
  from geosave_engine.ml.lightning.data import SemanticSegmentationDataModule
  from geosave_engine.ml.lightning.tasks import SemanticSegmentationTask
  from geosave_engine.ml.lightning.cli import GeosaveCLI
  ```

  with

  ```python
  from geosave_engine.ml.cli import GeosaveCLI
  from geosave_engine.ml.segmentation import supervised
  ```

  Then `SemanticSegmentationTask(` becomes `supervised.Module(`, `SemanticSegmentationDataModule(` becomes `supervised.DataModule(`, the string `"geosave_engine.ml.lightning.data.SemanticSegmentationDataModule"` at lines 48 and 98 becomes `"geosave_engine.ml.segmentation.supervised.DataModule"`, and the script line 111 `from geosave_engine.ml.lightning.cli import GeosaveCLI` becomes `from geosave_engine.ml.cli import GeosaveCLI`.

- [ ] **Step 9: Run the tests**

Run: `uv run pytest tests/test_layering.py tests/ml tests/model tests/cli -m "slow or not slow" -q`
Expected: PASS.

Run: `grep -rnE "ml\.(registry|lightning|metrics)\b|ml\.transforms\.semantic_segmentation|SemanticSegmentation(Task|DataModule)" src tests docs/guides`
Expected: no output.

- [ ] **Step 10: Commit** (only if Task 0 allows)

```bash
git add src/geosave_engine/ml/segmentation
git commit -m "refactor: lay out ml by head type and training method" -- src/geosave_engine/ml src/geosave_engine/model/README.md src/geosave_engine/templates tests/ml tests/cli tests/test_layering.py
```

---

### Task 5: Full verification

**Files:** none changed unless a check fails.

**Interfaces:**
- Consumes: everything above.
- Produces: the handoff report.

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: the Task 0 pass count plus the new default-run `tests/test_layering.py` cases (ten path cases and the pairing test), and the same pre-existing failures, if any.

Run: `uv run pytest tests/test_layering.py tests/cli -m slow -q`
Expected: PASS.

- [ ] **Step 2: Lint and docstrings**

Run: `uv run ruff check .`
Expected: no finding in a file this plan touched.

Run: `uv run python scripts/check_docstrings.py src/geosave_engine/ml/segmentation/__init__.py src/geosave_engine/ml/segmentation/supervised/__init__.py`
Expected: clean.

- [ ] **Step 3: Confirm nothing names an old path**

Run: `grep -rnE "geosave_engine\.(model_spec|release|geodata\.datasets|ml\.(model_chain|models|registry|lightning|metrics|encoding))\b|ml\.transforms\.semantic_segmentation" src tests docs/guides`
Expected: no output. `workspace/` and `docs/superpowers/` are not searched; they are historical.

Run: `find src/geosave_engine -type d -empty; find src/geosave_engine -type d -name __pycache__ -prune -o -type d -print | while read d; do [ -z "$(find "$d" -maxdepth 1 -type f -name '*.py')" ] && [ -z "$(find "$d" -mindepth 1 -maxdepth 1 -type d -not -name __pycache__)" ] && echo "no python: $d"; done`
Expected: only directories that hold non-Python files on purpose (templates). Any `ml/...` or `model_spec`/`release` leftover is removed after the `find ... -not -name '*.pyc'` check from Global Constraints.

- [ ] **Step 4: Smoke-test the new import paths from outside the repo**

Run:

```bash
cd "$(mktemp -d)" && uv run --project /home/uwu/Projects/geosave-engine python -W ignore - <<'EOF'
from geosave_engine.ml.segmentation import supervised
from geosave_engine.model.registry import build_model
from geosave_engine.model.release import save_model
from geosave_engine.model.spec import ModelSpec

model = build_model({
    "head": {"name": "segmentation", "init_args": {"feature_channels": 2, "classes": ["a", "b"]}},
})
print(type(model).__module__, sorted(model.stage_specs))
print(supervised.Module.__module__, supervised.DataModule.__module__)
print(save_model.__module__, ModelSpec.__module__)
EOF
```

Expected, three lines:

```text
geosave_engine.model.chain.chain ['head']
geosave_engine.ml.segmentation.supervised.module geosave_engine.ml.segmentation.supervised.data
geosave_engine.model.release.artifact geosave_engine.model.spec.model
```

- [ ] **Step 5: Report**

Report only: what moved, the checks run with their counts, and the breaking changes listed in the spec. Note that `AGENTS.md` still describes the old structure and that updating it is the user's call.

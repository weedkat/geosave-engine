# Library Structure Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Complete the first library structure iteration, preserving training and geodata behavior.

**Architecture:** Move public I/O into geodata/io and private root helpers to their consumers. Keep common template startup, select flat workspace starters, and retain optional files as scaffolds. Add a direct original-file Dynamic World downloader.

**Tech Stack:** Python, native LightningCLI, fsspec HTTP, zipfile, pytest, Hatchling, Zensical.

**Spec:** docs/superpowers/specs/2026-10-04-library-structure-design.md

## Global Constraints

- Preserve unrelated staged, unstaged, and untracked work; use the current checkout containing the active model/ml split.
- No compatibility aliases; preserve geodata <- model <- ml and torch-free geodata/model.spec imports.
- Keep training algorithms, sampling, model specs, and prediction interpretation unchanged.
- templates/common stays; tasks becomes flat workspaces/{segmentation,custom_lightning}; boilerplate becomes scaffolds.
- create selects --workspace/-w; blank uses common only. Keep make's category/file arguments.
- Benchmark API: dynamic_world.download(root, subset='experts') -> Path; original publisher layout and metadata, no catalog conversion.
- Record changes without committing or staging the user's shared checkout.
- Use the editable installation for development checks. Local wheel builds are not required; the GitHub release workflow owns packaging/publication.

## Review Focus

- Workspace selector errors must fail before writing a project; bare generation must preserve common files.
- Discovery must ignore Python caches and include dotfiles during copying.
- Benchmark failures and partial extraction must never appear complete; completed results may be reused.
- Archive paths must stay within the extraction directory, including Windows-style traversal names.
- Editable-source generation must find workspace/scaffold resources and compose native Lightning entry points.

### Task 1: Module and template ownership

Files: geodata/utils/io -> geodata/io; utils/file_ops.py -> cli/core/copy.py; utils palette definitions -> geodata/attrs/palette.py; tensor colorize and prediction logger -> ml/segmentation/callbacks.py; private sentinel -> gdal_env.py; matching tests and all live imports.

Interfaces: keep top-level geodata readers; create_workspace(root, workspace=None); get_workspaces() -> list[str]; get_scaffolds() -> dict[str,list[str]]; create_toml(..., workspace, description=None).

- [x] Add CLI behavior tests for segmentation/custom/blank selection, invalid selection, scaffold copying, and exclusions; watch the new interface fail.
- [x] Move implementations and tests, update imports, rename templates, implement direct workspace selection and metadata, preserve collision handling.
- [x] Run affected CLI, persistence, metadata, callbacks, model-spec examples, and layering suites.

### Task 2: Dynamic World benchmark

Files: geodata/benchmarks/{__init__,dynamic_world}.py; tests/geodata/benchmarks/test_dynamic_world.py.

Interfaces: download(root: str|Path, *, subset: Literal['experts','non_expert','validation','test']='experts') -> Path.

- [x] Write tests using controlled local archives and transfers for original layout/metadata, reuse, incomplete destination, failed transfer, corrupt ZIP, checksum mismatch, extraction interruption, and escaping members; watch import fail.
- [x] Implement explicit publisher mapping, temporary downloads/extraction, archive/available checksum validation, and atomic completion. Keep transfer machinery private.
- [x] Run local tests and a bounded publisher download smoke. Document archive size and supplied imagery/label distinctions.

### Task 3: Documentation and development verification

Files: README.md, AGENTS.md structure, model/README.md, guides/{architecture,benchmarks,workspaces}.md, zensical.toml, design status.

- [x] Update docs for implemented names, new CLI selectors and benchmark API; no claims of future training/EDA support.
- [x] Run scoped Ruff, diff whitespace check, Zensical build, and editable-source generation/config parsing. Extra wheel checks already completed are recorded in the execution history.
- [x] Review the change against the pre-refactor snapshot; fix material issues and report actual checks and import/CLI breaks.

## Execution record

Ruling: work in place without commits because active uncommitted changes are the input to this refactor. Backup: /tmp/geosave-library-refactor-before.

Ruling: preserve make and rename its source to scaffolds rather than removing useful existing file generation. Generated workspace remains a consumer and is not rewritten.

Ruling: latest explicit implementation request authorizes this first iteration; future training/prediction/EDA iterations remain deferred.

Task 1: affected suites 445 passed, 22 deselected after the ownership and CLI moves.
Task 2: controlled benchmark tests 14 passed; publisher validation smoke extracted 5,794 files and verified reuse.
Check: scoped BasedPyright passed with 0 errors; Ruff src/tests passed; Zensical build passed.

Task 3: complete. Wheel build and wheel-only generation/config parsing/scaffolds passed outside the checkout; old package paths were absent from the archive. Zensical build, Ruff src/tests, scoped BasedPyright, and git diff --check passed.
Additional verification: model/spec + ml/datasets + supervised segmentation: 246 passed, 1 deselected; slow layering/custom Lightning workspace: 6 passed, 16 deselected.
Final review: fresh read-only reviewer found no Critical or Important issues. Minor malformed-completion-JSON error contract clarified in the public docstring and guide without changing behavior.
Final rulings: do not revalidate edited/deleted files after a completion record or coordinate concurrent benchmark writers; these limitations are explicit in the guide. Preserve existing callback/copy behavior and defer model-head postprocessing/EDA work. Wheel generation has been independently exercised.

User steering: local wheel builds are unnecessary during this development refactor. Verified that the environment imports src/geosave_engine directly and direct_url.json declares editable=true. GitHub release.yml builds/publishes on v* tags; main source pushes publish development builds to TestPyPI. No further local packaging work.

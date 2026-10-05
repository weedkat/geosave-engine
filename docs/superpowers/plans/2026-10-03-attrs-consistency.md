# Attrs Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AttrsHeader.root` never None, make model `NAME`s follow one rule, and give each attrs module one kind of content.

**Architecture:** A header's root defaults to an empty namespace, and `rebase` writes a root only when it carries attrs. Joins start from dropped attrs so the header writes them. `NAME` is the class name in snake_case. `attrs/xarray.py` holds everything that reads or writes xarray objects, and `model.py` holds every value-level helper.

**Tech Stack:** Python 3.12, Pydantic 2, xarray, pytest

**Spec:** this plan amends `docs/superpowers/specs/2026-10-03-attrs-header-factories-cleanup-design.md` (its section "A header holds only what it describes") and `2026-10-03-attrs-scoped-models-design.md` (model names). Both are updated in Task 4.

## Global Constraints

- `header.root` is always an `AttrsNamespace`.
- `rebase` accepts an `AttrsModel` sequence, one `AttrsNamespace`, or one `AttrsHeader`.
- No aliases for the old `NAME`s, `attrs.headers.xarray`, or `attrs.validate`.
- No commits.

## Review Focus

- A time concat whose rasters disagree on every root key ends with an empty root; Task 1 tests this.
- The array builder applies the grid header to a DataArray, whose root is variable-scoped; the empty dataset root must be skipped before any scope check. Task 1 runs the core suite.
- Model-spec configs naming models use the new names; Task 2 runs the model_spec suite.

---

### Task 1: The root is never None

- [ ] Pin test in `tests/geodata/transform/test_concat.py`: concatenating rasters with root `{"title": "a"}` and `{"title": "b"}` gives a result whose `attrs` has no `title`. Expected: PASS today.
- [ ] Change `AttrsHeader.root` to `field(default_factory=lambda: AttrsNamespace("dataset"))`; `from_attrs(root=None)` gives that empty root; `merge` merges every root. In `rebase`'s header form, append the root update only when `header.root.models or header.root.foreign`.
- [ ] Update the tests asserting `header.root is None` to assert an empty `header.root.to_attrs()`.
- [ ] Run the concat pin. Expected: FAIL, because `xr.concat`'s default `combine_attrs="override"` keeps raster 0's title.
- [ ] Pass `combine_attrs="drop"` to both `xr.concat` calls in `concat_time`. Run `uv run pytest tests/geodata -q`. Expected: pass.
- [ ] Run basedpyright on `src/geosave_engine/geodata` and `src/geosave_engine/model_spec`. Expected: no `reportOptionalMemberAccess` on `root`.

### Task 2: `NAME` is the class name in snake_case

- [ ] Add to `tests/geodata/attrs/test_models.py`:

```python
def test_model_names_are_their_class_in_snake_case() -> None:
    for models in MODELS.values():
        for model in models:
            assert model.NAME == re.sub(r"(?<=[a-z])(?=[A-Z])", "_", model.__name__).lower()
```

  Expected: FAIL for `CFVariable` and seven others. (`ACDD` and `GDALVariable` need the rule to keep acronym runs together; the expected names are listed in the spec table.)
- [ ] Rename the eight `NAME`s, then every `NAME`-keyed use: `rebase` keywords, `AttrsNamespace(models={...})` keys in tests, model-spec configs in tests, docstrings, and error-message expectations. Read each file; no regex sweep.
- [ ] Run `uv run pytest tests/geodata tests/model_spec -q`. Expected: pass.

### Task 3: One module per kind of content

- [ ] Move `create_header` and `XarrayObject` from `attrs/headers/xarray.py` into `attrs/xarray.py` and delete the old module. `attrs/__init__.py` re-exports `create_header` from `.xarray`.
- [ ] Move `parse_collection_text` into `model.py`, delete `validate.py`, and update the imports in four models and `test_field_values.py`.
- [ ] Fix `namespace.py`'s merge so the scope is typed `Scope`, not `str`.
- [ ] Run `uv run pytest -q` and basedpyright on `attrs`. Expected: pass, and no new attrs errors.

### Task 4: Docs and verification

- [ ] Update the two specs (empty-root rule, concat `combine_attrs="drop"`, name table) and the 09-24 header-factories spec's module tree.
- [ ] Run `uv run pytest -q`, `uv run ruff check src tests`, and the docstring checker on touched files.

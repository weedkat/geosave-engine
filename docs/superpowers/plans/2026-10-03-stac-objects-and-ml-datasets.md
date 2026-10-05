# STAC Config Objects and ML Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `StacSource` be built with typed `StacSourceConfig` and `StacQuery` objects, alongside the existing `set_config`/`set_query`, and move `TileDataset` from `geodata` into `ml`.

**Architecture:** `StacQuery` accepts the text spellings of a filter and a sort order and normalises them itself, so one object serves both Python callers and YAML recipes. `StacSource` takes optional `config=` and `query=` at construction and owns the collection name; `set_config` and `set_query` stay as they are. `TileDataset` moves unchanged to `geosave_engine.ml.datasets`.

**Tech Stack:** Python 3.12, Pydantic 2, cql2, pystac-client, odc-stac, PyTorch, pytest

**Spec:** Agreed in conversation on 2026-10-03; no spec file. The decisions are restated in Global Constraints.

## Global Constraints

- `StacSource.set_config` and `StacSource.set_query` keep their signatures and behaviour. `StacQuery.set_filter` and `StacQuery.sort_by` are left untouched.
- Passing `config=` or `query=` at construction is equivalent to calling the setters afterwards.
- `StacQuery.filter` accepts CQL2 text or CQL2-JSON. `StacQuery.sortby` accepts `"-field"` strings or STAC POST dicts. Both are stored in the JSON form.
- `StacQuery.collections` defaults to empty. A `StacSource` always sets it to its own collection.
- `model_spec` YAML recipes keep their current spelling and behaviour.
- `TileDataset` keeps its signature and its `(model_inputs, index)` samples.
- `geodata` must not import `torch` at runtime after the move.
- The working tree holds unrelated staged and unstaged user work, including edits to `stac/source.py` and `tests/geodata/stac/test_source.py`. Edit on top of them. Commit only the listed paths with `git commit -m ... -- <paths>`. Never `git add -A`, `git checkout --`, `git restore`, or `git stash`.

## Review Focus

- A recipe's CQL2-JSON filter and dict sort keys must pass through `StacQuery` untouched. Task 1 pins this.
- `sortby=[]` must be refused like `ids=[]`, not sent as an empty sort. Task 1 pins this.
- A query handed to a source for a different collection must end up searching the source's collection. Task 2 pins this.
- `set_config` on a source built with `config=` must merge onto that config, not reset it. Task 2 pins this.
- `cql2.Expr("eo:cloud_cover <=").to_json()` raised nothing when checked, so no test asserts that malformed filter text is rejected. The executor must not add a `Raises` claim for it.

## File Structure

| File | Change |
| --- | --- |
| `src/geosave_engine/geodata/stac/query.py` | Text spellings in the constructor |
| `src/geosave_engine/geodata/stac/source.py` | `config=`/`query=` added to the constructor |
| `src/geosave_engine/geodata/stac/client.py` | `source()` forwards `config` and `query` |
| `src/geosave_engine/model_spec/stac.py` | Recipe passes both to `client.source` |
| `tests/geodata/stac/test_query.py` | New: query normalisation |
| `tests/geodata/stac/test_source.py` | New constructor tests |
| `src/geosave_engine/ml/datasets/` | New home of `TileDataset` |
| `src/geosave_engine/geodata/datasets/` | Deleted |
| `tests/ml/datasets/test_tiles.py` | Moved test |

---

### Task 1: `StacQuery` takes the text spellings

**Files:**
- Modify: `src/geosave_engine/geodata/stac/query.py`
- Create: `tests/geodata/stac/test_query.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `StacQuery(collections: list[str] = [], ..., filter: str | dict[str, Any] | None = None, sortby: list[str | dict[str, str]] | None = None)`. After construction `filter` is CQL2-JSON or None and `sortby` is a list of `{"field", "direction"}` dicts or None. `set_filter` and `sort_by` are unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/geodata/stac/test_query.py`:

```python
from __future__ import annotations

import pytest

from geosave_engine.geodata.stac import StacQuery


def test_filter_text_is_stored_as_cql2_json() -> None:
    query = StacQuery(filter="eo:cloud_cover <= 10")

    assert query.filter == {
        "op": "<=",
        "args": [{"property": "eo:cloud_cover"}, 10.0],
    }
    assert query.to_search_params()["filter_lang"] == "cql2-json"


def test_a_cql2_json_filter_passes_through() -> None:
    expression = {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10.0]}

    assert StacQuery(filter=expression).filter == expression


def test_signed_sort_fields_become_stac_sort_keys() -> None:
    query = StacQuery(sortby=["-properties.eo:cloud_cover", "+id", "collection"])

    assert query.sortby == [
        {"field": "properties.eo:cloud_cover", "direction": "desc"},
        {"field": "id", "direction": "asc"},
        {"field": "collection", "direction": "asc"},
    ]


def test_stac_sort_keys_pass_through() -> None:
    keys = [{"field": "properties.datetime", "direction": "desc"}]

    assert StacQuery(sortby=keys).sortby == keys


def test_a_sort_field_with_no_name_is_refused() -> None:
    with pytest.raises(ValueError, match="no name after the sign"):
        StacQuery(sortby=["-"])


def test_an_empty_sort_order_is_refused() -> None:
    with pytest.raises(ValueError, match="sortby is set but empty"):
        StacQuery(sortby=[])


def test_a_query_without_collections_sends_none() -> None:
    assert "collections" not in StacQuery(max_items=1).to_search_params()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/stac/test_query.py -v`
Expected: all 7 FAIL with `TypeError: StacQuery.__init__() missing 1 required positional argument: 'collections'`.

- [ ] **Step 3: Implement**

In `src/geosave_engine/geodata/stac/query.py`:

Change the dataclasses import to:

```python
from dataclasses import dataclass, field
```

(keep `import dataclasses`; `set_filter` and `sort_by` use it.)

Add above the class:

```python
def _sort_key(key: str | dict[str, str]) -> dict[str, str]:
    """Spell one sort key in STAC POST form."""
    if not isinstance(key, str):
        return key
    signed = key[:1] in ("+", "-")
    path = key[1:] if signed else key
    if not path:
        raise ValueError(f"sort field {key!r} has no name after the sign")
    return {"field": path, "direction": "desc" if key[:1] == "-" else "asc"}
```

Replace the class docstring's `collections`, `filter`, and `sortby` entries and its `Raises` with:

```python
        collections: Collection IDs to search. Empty searches every
            collection the catalog offers.
        filter: CQL2 text such as `"eo:cloud_cover <= 10"`, or CQL2-JSON.
        sortby: Sort keys in priority order, each a field path optionally
            prefixed `-` for descending or `+` for ascending, or a STAC POST
            key `{"field": <path>, "direction": "asc" | "desc"}`. Property
            fields need the `properties.` prefix.
```

```python
    Raises:
        ValueError: `ids` or `sortby` is given but empty, `bbox` is invalid or
            combined with `intersects`, a sort field has no name after its
            sign, or a limit is below one.

    Examples:
        >>> StacQuery(filter="eo:cloud_cover <= 10", sortby=["-properties.eo:cloud_cover"])
```

Replace the three field declarations:

```python
    collections: list[str] = field(default_factory=list)
```

```python
    filter: str | dict[str, Any] | None = None
```

```python
    sortby: list[str | dict[str, str]] | None = None
```

In `__post_init__`, delete:

```python
        if not self.collections:
            raise ValueError("StacQuery needs at least one collection")
```

and append at the end of `__post_init__`:

```python
        if self.sortby is not None and not self.sortby:
            raise ValueError("sortby is set but empty; pass at least one key or None")
        # The dataclass is frozen, so the stored spelling is set past __setattr__.
        if isinstance(self.filter, str):
            object.__setattr__(self, "filter", Expr(self.filter).to_json())
        if self.sortby is not None:
            object.__setattr__(self, "sortby", [_sort_key(key) for key in self.sortby])
```

In `to_search_params`, change the collections entry to:

```python
            "collections": self.collections or None,
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/stac tests/model_spec -q`
Expected: PASS, including the 7 new tests and `tests/model_spec/test_stac.py`.

- [ ] **Step 5: Commit**

```bash
git add tests/geodata/stac/test_query.py
git commit -m "feat: let StacQuery take text filters and signed sort fields" -- src/geosave_engine/geodata/stac/query.py tests/geodata/stac/test_query.py
```

---

### Task 2: `StacSource` takes `config` and `query` at construction

**Files:**
- Modify: `src/geosave_engine/geodata/stac/source.py` (`StacSource` docstring and `__init__`)
- Modify: `src/geosave_engine/geodata/stac/client.py:134-155`
- Modify: `src/geosave_engine/model_spec/stac.py` (`StacRecipe.load_raster`)
- Test: `tests/geodata/stac/test_source.py`

**Interfaces:**
- Consumes: `StacQuery` from Task 1.
- Produces:
  - `StacSource(client: SearchClient, *, collection: str, config: StacSourceConfig | None = None, query: StacQuery | None = None)`. `set_config` and `set_query` are unchanged.
  - `StacClient.source(collection: str, *, config: StacSourceConfig | None = None, query: StacQuery | None = None) -> StacSource`.

- [ ] **Step 1: Write the failing tests**

In `tests/geodata/stac/test_source.py`, add after `test_item_ids_do_not_gain_target_selectors`:

```python
def test_a_source_searches_its_own_collection_whatever_the_query_names() -> None:
    source = StacSource(
        FakeClient(),  # type: ignore[arg-type]
        collection="example",
        query=StacQuery(collections=["another"], max_items=3),
    )

    assert source.query.collections == ["example"]
    assert source.query.max_items == 3


def test_a_source_keeps_the_config_it_is_given() -> None:
    config = StacSourceConfig(bands=["red"], groupby="time")

    source = StacSource(FakeClient(), collection="example", config=config)  # type: ignore[arg-type]

    assert source.config is config


def test_set_config_merges_onto_the_config_given_at_construction() -> None:
    source = StacSource(
        FakeClient(),  # type: ignore[arg-type]
        collection="example",
        config=StacSourceConfig(bands=["red"], groupby="time"),
    ).set_config(dtype="float32")

    assert source.config.bands == ["red"]
    assert source.config.groupby == "time"
    assert source.config.dtype == "float32"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/geodata/stac/test_source.py -k "its_own_collection or keeps_the_config or merges_onto" -v`
Expected: all three FAIL with `TypeError: StacSource.__init__() got an unexpected keyword argument`.

- [ ] **Step 3: Extend `StacSource.__init__`**

In `src/geosave_engine/geodata/stac/source.py`, replace the class docstring's second paragraph, `Args`, and `Examples` with:

```python
    Pixels arrive as the provider published them, with no radiometric scaling.
    Settings may be given here or narrowed afterwards with `set_config` and
    `set_query`.

    Args:
        client: Client used to search the catalog.
        collection: Collection ID to load.
        config: How pixels load. None loads on every default.
        query: Search narrowing merged with each anchor's extent and window.
            Its `collections` are replaced by `collection`. None searches the
            whole collection.

    Examples:
        >>> from geosave_engine.geodata import configure_gdal
        >>> configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)
        >>> source = client.source(
        ...     "sentinel-2-l2a",
        ...     config=StacSourceConfig(bands=["B04", "B08"]),
        ...     query=StacQuery(filter="eo:cloud_cover <= 10"),
        ... )
        >>> source = client.source("sentinel-2-l2a").set_config(bands=["B04", "B08"])
        >>> ds = source.load(anchor)
        >>> ds.gs.attrs.root.get(StacMetadata).properties()
        ('eo:cloud_cover', 'platform')
```

Replace `__init__` with:

```python
    def __init__(
        self,
        client: SearchClient,
        *,
        collection: str,
        config: StacSourceConfig | None = None,
        query: StacQuery | None = None,
    ) -> None:
        """Bind the client, collection, and settings.

        Args:
            client: Client used to search the catalog.
            collection: Collection ID to load.
            config: How pixels load. None loads on every default.
            query: Search narrowing. None searches the whole collection.
        """
        self.client = client
        self.collection = collection
        self.config = config or StacSourceConfig()
        self.query = replace(query or StacQuery(), collections=[collection])
```

Leave `set_config` and `set_query` untouched.

- [ ] **Step 4: Forward the objects from `StacClient.source`**

In `src/geosave_engine/geodata/stac/client.py`, change the import and the method:

```python
from .source import StacSource, StacSourceConfig
```

```python
    def source(
        self,
        collection: str,
        *,
        config: StacSourceConfig | None = None,
        query: StacQuery | None = None,
    ) -> StacSource:
        """Build a source for one collection on this catalog.

        Settings may be given here or narrowed afterwards with `set_config`
        and `set_query`.

        Args:
            collection: Collection ID. Discover them with `collections`.
            config: How pixels load. None loads on every default.
            query: Search narrowing. None searches the whole collection.

        Returns:
            Source for `collection`.

        Raises:
            ValueError: `collection` is not on this catalog.

        Examples:
            >>> source = client.source("sentinel-2-l1c").set_config(bands=["B02"])
            >>> source = client.source(
            ...     "sentinel-2-l1c", config=StacSourceConfig(bands=["B02"])
            ... )
        """
        self.collection(collection)
        return StacSource(self, collection=collection, config=config, query=query)
```

- [ ] **Step 5: Pass both from the recipe**

In `src/geosave_engine/model_spec/stac.py`, in `StacRecipe.load_raster`, replace:

```python
        source = client.source(self.collection)
        source.query = self.query.to_query(self.collection)
        source.config = self.load
        return source.load(anchor)
```

with:

```python
        source = client.source(
            self.collection,
            config=self.load,
            query=self.query.to_query(self.collection),
        )
        return source.load(anchor)
```

- [ ] **Step 6: Run the tests and lint**

Run: `uv run pytest tests/geodata/stac tests/model_spec tests/workflow -q`
Expected: PASS.

Run: `uv run ruff check src/geosave_engine/geodata/stac src/geosave_engine/model_spec/stac.py tests/geodata/stac`
Expected: `All checks passed!`

Run: `uv run python scripts/check_docstrings.py src/geosave_engine/geodata/stac/source.py src/geosave_engine/geodata/stac/query.py src/geosave_engine/geodata/stac/client.py`
Expected: `clean.`

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: accept config and query when building a StacSource" -- src/geosave_engine/geodata/stac/source.py src/geosave_engine/geodata/stac/client.py src/geosave_engine/model_spec/stac.py tests/geodata/stac/test_source.py
```

---

### Task 3: Move `TileDataset` into `ml`

> **Superseded 2026-10-03.** Done by Task 1 of `2026-10-03-package-restructure.md`. Skip this task; `geodata/datasets/` no longer exists.

**Files:**
- Create: `src/geosave_engine/ml/datasets/__init__.py`
- Move: `src/geosave_engine/geodata/datasets/tiles.py` to `src/geosave_engine/ml/datasets/tiles.py`
- Delete: `src/geosave_engine/geodata/datasets/__init__.py`
- Move: `tests/geodata/datasets/test_tiles.py` to `tests/ml/datasets/test_tiles.py`
- Modify: `tests/ml/models/encoder/test_model_context.py:15`
- Modify: `tests/ml/lightning/tasks/test_semantic_segmentation.py:525-527`
- Modify: `src/geosave_engine/ml/models/README.md:165`

**Interfaces:**
- Consumes: nothing.
- Produces: `from geosave_engine.ml.datasets import TileDataset`, same class.

- [ ] **Step 1: Write the failing test**

Create `tests/geodata/test_imports.py`:

```python
from __future__ import annotations

import subprocess
import sys


def test_geodata_imports_without_torch() -> None:
    script = (
        "import sys, geosave_engine.geodata\n"
        "assert 'torch' not in sys.modules, 'geodata imported torch'\n"
        "import importlib.util\n"
        "assert importlib.util.find_spec('geosave_engine.geodata.datasets') is None\n"
    )

    subprocess.run([sys.executable, "-c", script], check=True)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/geodata/test_imports.py -v`
Expected: FAIL with `CalledProcessError`; the subprocess asserts on `geosave_engine.geodata.datasets` still being found. If it fails earlier on `'geodata imported torch'`, stop and report which import pulled torch in; that is outside this task.

- [ ] **Step 3: Move the files**

```bash
mkdir -p src/geosave_engine/ml/datasets tests/ml/datasets
git mv src/geosave_engine/geodata/datasets/tiles.py src/geosave_engine/ml/datasets/tiles.py
git rm src/geosave_engine/geodata/datasets/__init__.py
git mv tests/geodata/datasets/test_tiles.py tests/ml/datasets/test_tiles.py
```

Create `src/geosave_engine/ml/datasets/__init__.py`:

```python
"""Torch datasets, one module per task."""

from .tiles import TileDataset

__all__ = ["TileDataset"]
```

- [ ] **Step 4: Update the imports**

`tests/ml/datasets/test_tiles.py` and `tests/ml/models/encoder/test_model_context.py`, replace:

```python
from geosave_engine.geodata.datasets import TileDataset
```

with:

```python
from geosave_engine.ml.datasets import TileDataset
```

`tests/ml/lightning/tasks/test_semantic_segmentation.py`, replace:

```python
    from geosave_engine.geodata.datasets import TileDataset
    from geosave_engine.geodata.transform.tiling import Tiles
    from tests.geodata.datasets.test_tiles import _raster
```

with:

```python
    from geosave_engine.geodata.transform.tiling import Tiles
    from geosave_engine.ml.datasets import TileDataset
    from tests.ml.datasets.test_tiles import _raster
```

`src/geosave_engine/ml/models/README.md`, replace:

```python
from geosave_engine.geodata.datasets import TileDataset
```

with:

```python
from geosave_engine.ml.datasets import TileDataset
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/geodata/test_imports.py tests/ml/datasets tests/ml/models/encoder/test_model_context.py tests/ml/lightning/tasks/test_semantic_segmentation.py -q`
Expected: PASS.

Run: `grep -rn "geodata.datasets\|geodata/datasets" src tests`
Expected: only the two lines inside `tests/geodata/test_imports.py`.

- [ ] **Step 6: Commit**

```bash
git add src/geosave_engine/ml/datasets/__init__.py tests/geodata/test_imports.py
git commit -m "refactor: move TileDataset into ml.datasets" -- src/geosave_engine/ml/datasets src/geosave_engine/geodata/datasets tests/ml/datasets tests/geodata/datasets tests/geodata/test_imports.py tests/ml/models/encoder/test_model_context.py tests/ml/lightning/tasks/test_semantic_segmentation.py src/geosave_engine/ml/models/README.md
```

---

### Task 4: Full verification

**Files:** none.

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: every test passes. Report any failure by name, including ones this plan did not cause.

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`
Expected: no new findings in the files this plan touched.

- [ ] **Step 3: Report**

State what changed, the checks run with their results, and the breaking changes: `StacQuery` no longer requires `collections`, and the `geosave_engine.geodata.datasets` import path is gone.

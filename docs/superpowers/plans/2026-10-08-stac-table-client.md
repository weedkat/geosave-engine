# STAC Table Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Search a STAC GeoParquet table with the same `StacQuery` a STAC API takes, so a `StacSource` can sit on a table.

**Architecture:** `StacTableClient` wraps rustac's DuckDB search over one Parquet file or a folder of them and satisfies the `SearchClient` protocol `StacSource` already takes. `StacClient.open` hands a table path to it; STAC APIs stay on pystac-client, which carries provider signing and retries.

**Tech Stack:** rustac 0.9.17 (bundles DuckDB), PySTAC, existing `stac.table`.

**Spec:** This document. Design agreed in conversation 2026-10-08.

## Design

```python
client = StacClient.open("samples/catalog")            # a table: file or folder of parts
client.collections()                                   # {'forest'}
items = client.search(StacQuery(collections=["forest"]).set_filter("eo:cloud_cover <= 10"))
ds = client.source("forest").set_config(bands=["red", "nir"]).load(anchor)
```

```python
class StacTableClient:
    def __init__(self, path: str | PathLike[str]) -> None: ...
    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...
    def collections(self) -> set[str]: ...
    def collection(self, collection: str) -> pystac.Collection: ...
    def source(self, collection: str) -> StacSource: ...
```

`search` hands the query to `rustac.DuckdbClient.search` and adapts two
things: a datetime or `(start, end)` becomes the one RFC 3339 string rustac
takes, and asset hrefs, stored relative to the table, are made absolute. A
sort or filter names a property as the table's column, without the
`properties.` prefix a STAC API takes.

A Collection comes from the file metadata `table.write(..., collections=)`
stored, else the one rustac derives from that id's rows.

Simplified 2026-10-08 at the user's direction: no storage-options adapter,
DuckDB secret builder, or local copy of remote tables. A URL is passed directly
to rustac. `StacClient.open(..., duckdb=session)` and
`StacTableClient(..., duckdb=session)` accept a native `rustac.DuckdbClient`,
which the caller can configure with DuckDB's own secret API. The same session
reads rows and stored Collection metadata; rustac's Arrow extra supplies the
native metadata query result.

## Global Constraints

- pystac-client stays the client for STAC APIs. No behaviour of `StacClient` for a URL changes.
- `StacClient.open` picks the table client for a path ending in `.parquet` / `.geoparquet` or a local directory.
- Returned Items carry absolute asset hrefs.
- No commits.

## Review Focus

1. A folder of part files searches as one table and its hrefs resolve against the folder. (Task 1)
2. A naive `(start, end)` tuple is read as UTC and selects the same rows a STAC API would. (Task 1)
3. A table with no stored Collections still builds a source. (Task 1)
4. A collection id no row carries raises `CollectionNotFoundError`. (Task 1)
5. `StacSource.load` over a split-band table returns the written pixels. (Task 2)

Known limit, documented and not worked around: odc-stac 0.5.2 does not read STAC
1.1 `bands`, so a source over a GeoSave table loads `float32` without nodata,
and a multi-band asset needs `stac_cfg` aliases. `stac.table.load` stays the
faithful reader of GeoSave tables.

---

### Task 1: `StacTableClient`

**Files:** `StacTableClient` in `src/geosave_engine/geodata/stac/client.py`, beside `StacClient`; Modify `src/geosave_engine/geodata/stac/table.py` (share one href resolver), `stac/__init__.py`, `pyproject.toml`; Test `tests/geodata/stac/test_table_client.py`.

- [x] Failing tests for: collection, bbox, tuple datetime, CQL2 filter, `sort_by("-eo:cloud_cover")`, ids; absolute hrefs from a file and from a folder; stored Collection; derived Collection; missing collection.
- [x] Implement; `uv run pytest tests/geodata/stac/test_table_client.py -q`.

### Task 2: `StacClient.open` and `StacSource` on a table

**Files:** Modify `src/geosave_engine/geodata/stac/client.py`; Test `tests/geodata/stac/test_table_client.py`, `tests/geodata/stac/test_client.py`.

- [x] Failing tests: `StacClient.open(table)` returns a `StacTableClient`; `client.source("forest").load(anchor)` equals the written pixels.
- [x] Implement; `uv run pytest`, ruff, BasedPyright, docstring check; add the example to `docs/guides/architecture.md`.

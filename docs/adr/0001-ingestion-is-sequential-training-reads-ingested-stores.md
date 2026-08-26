# 1. Ingestion is sequential; training reads ingested stores

Date: 2026-08-26

## Status

Accepted

## Context

Cutting a surface into windows and reading each one on its own multiplies
network reads. A window narrower than a Dask chunk makes that chunk fetch
once per window: at `tile_size_px=512` over `chunks=1024`, four windows land
in one chunk, so it is fetched four times. Measured on a 2048x2048 surface
with two layers, 16 windows over 4 chunks cost 16 reads against an ideal of 4.

The obvious fix — cache chunks in memory and serve windows out of the cache —
requires knowing which windows come next. That holds while ingesting, where
read order is ours to choose. It does not hold under a shuffling DataLoader,
where the next index is random by design.

A shared mutable buffer would also not survive the boundaries this library
already crosses. `litdata.optimize()` pickles its inputs to worker processes,
so a window holding a live buffer reference cannot be shipped. Results would
depend on read history, breaking the value semantics of a frozen
`_SpatialArray`, and `compute()` is already documented as unsafe from several
threads of one process.

No dataset in this library reads a live remote surface. `StackDataset` takes a
directory of `.zarr` stores, `RasterDataset` takes files, and `StoreDataset`
takes a `LitDataStore`, whose remote streaming is litdata's own chunk-sequential
concern. The constraint already held in practice; it was nowhere in writing.

## Decision

Live remote surfaces are ingestion-only, and ingestion is sequential — read
order belongs to us. Training reads ingested stores, where random access is
local or streamed chunk-wise, and shuffling happens at the store level after
ingestion.

Read amplification during ingestion is therefore solved by widening the unit
of `dask.compute` (`read_windows`), not by caching. Handing several windows
to one compute lets Dask deduplicate the tasks they share. Measured on the
same surface: 16 reads per-window, 8 at batch 4, 4 — the ideal — at batch 8
and above.

## Consequences

- No cache, no buffer object, no global state, and nothing new to invalidate.
- `tiles()` keeps its contract unchanged: lazy in, lazy out. Batching is a
  consumption concern, not a property of a window.
- Batch size is a memory budget. 16 windows of 512px over 13 bands at uint16
  is about 109 MB, which is what makes the ideal reachable.
- Adding a Dataset that reads a live STAC surface would reintroduce the
  problem this records. Ingest first, then train off the store.
- Row-major batches are strips rather than squares, so a batch smaller than a
  chunk row leaves some amplification. Grouping windows by containing chunk
  would close that, and is deliberately not built until a real workload shows
  it matters.

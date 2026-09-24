# GeoVector Spatial Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `GeoVector` into the composable spatial collection for geometry, anchored xarray metadata, spatial queries, portable raster references, and flag raster conversion, then remove the duplicate `Manifest` abstraction.

**Architecture:** Constructors translate geometry, `GeoAnchor`, and native xarray objects into ordinary one-row `GeoVector` values. Functional `concat` and keyed `upsert` grow collections, while exact spatial queries use GeoPandas' index only for candidates. GeoParquet remains the durable catalog format, and Rasterio performs explicit eager vectorize/rasterize operations on native ODC grids.

**Tech Stack:** Python 3.12+, GeoPandas, pandas, Shapely, xarray, Dask, ODC Geo, Rasterio, PyArrow/GeoParquet, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-24-geovector-spatial-collection-design.md`

## Global Constraints

- Keep native `GeoDataFrame`, xarray, ODC, and Rasterio objects in public APIs; add no dependency or replacement table/raster abstraction.
- Every materialized `GeoVector` has one active CRS; collection coordinates are never silently rewritten.
- Registering an xarray object must not compute its pixel values; `vectorize` and `rasterize` are explicit eager boundaries.
- Geometry is the only required record value; bbox and irregular polygon are not separate row types.
- `Manifest` is removed without a compatibility alias; the generated workspace CSV/XLSX ingest ledger remains unchanged.
- Persisted local asset paths stay below and relative to the GeoParquet file's parent, including after symlink resolution.
- Preserve unrelated working-tree changes and commit only files named by each task.
- Before implementation, record which task paths are already dirty. Do not stage
  or commit a path that contained pre-existing work; keep the implementation
  change in place and report that the task checkpoint was intentionally skipped.

## Review Focus

- Frames whose active geometry column is not named `geometry`: composition must canonicalize it without losing properties or selecting a secondary geometry.
- Directional spatial predicates (`contains`, `within`, `covers`, `covered_by`): the collection row must be the left operand and the target the right operand.
- Null paths and stale staging files: valid rows must round-trip, and a failed path validation must leave the previous catalog untouched.
- Dask-backed xarray input: registration must schedule zero pixel tasks, while vectorization must intentionally compute the selected two-dimensional flags.
- Flag dtype and overlap behavior: unsupported polygonization dtypes must fail clearly, representability must be checked before rasterization, and later rows must win.

---

## File map

- `src/geosave_engine/geodata/core/vector.py`: `GeoVector` state, record constructors, composition, queries, asset resolution, vectorization, and rasterization.
- `src/geosave_engine/geodata/utils/io/geoparquet.py`: atomic GeoParquet writing and portable `path` normalization.
- `src/geosave_engine/geodata/utils/io/__init__.py`: attach a local source path to vectors returned by `read_vector`.
- `src/geosave_engine/geodata/pipeline/manifest.py`: remove after its identity and persistence behavior moves to `GeoVector`.
- `src/geosave_engine/geodata/pipeline/__init__.py`: remove the obsolete `Manifest` export; remove the package if it has no remaining surface.
- `tests/geodata/test_vector.py`: focused construction, collection, query, conversion, laziness, and error tests.
- `tests/geodata/test_raster_io_smoke.py`: public `read_vector` source context and filtered GeoParquet read smoke coverage.
- `tests/geodata/pipeline/test_manifest.py`: remove after equivalent portable-path and upsert tests exist under `GeoVector`.

### Task 1: Spatial record constructors and exact anchor identity

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py:1-250`
- Create: `tests/geodata/test_vector.py`

**Interfaces:**
- Consumes: existing `GeoAnchor`, `GeoAccessor.anchor`, `GeoStack.rasters`, `naive_utc`, and `GeoVector.from_geometry` CRS parsing.
- Produces: `GeoVector.empty(crs)`, extended `GeoVector.from_geometry(..., **properties)`, `GeoVector.from_anchor(anchor, *, crs=None, fields=("time", "grid"), **properties)`, `GeoVector.from_xarray(data, *, geometry=None, crs=None, path=None, fields=("time", "grid"), **properties)`, and the `source` persistence context used by Task 4.

- [ ] **Step 1: Write constructor and empty-collection smoke tests**

Create `tests/geodata/test_vector.py` with the public behavior first:

```python
from datetime import UTC

import dask
import geopandas as gpd
import pytest
from pandas import isna
from shapely.geometry import Point, box

from geosave_engine.geodata import GeoAnchor, GeoVector

from .conftest import build_raster


def test_empty_vector_has_a_crs_and_no_footprint() -> None:
    vector = GeoVector.empty("EPSG:4326")

    assert len(vector) == 0
    assert vector.crs.to_epsg() == 4326
    with pytest.raises(ValueError, match="empty"):
        _ = vector.footprint


def test_geometry_properties_become_columns() -> None:
    vector = GeoVector.from_geometry(
        Point(112.15, -8.05), place_name="Malang", split="train"
    )

    assert vector.gdf.loc[0, "place_name"] == "Malang"
    assert vector.gdf.loc[0, "split"] == "train"


def test_anchor_record_has_exact_grid_time_and_identity() -> None:
    anchor = GeoAnchor.from_coordinates(
        -8.05, 112.15, shape=(3, 4), resolution=10, timespan="2025-01"
    )

    vector = GeoVector.from_anchor(anchor, crs="EPSG:4326")
    row = vector.gdf.iloc[0]

    assert len(row.anchor_id) == 16
    assert row.grid_crs == str(anchor.crs)
    assert tuple(row.grid_transform) == tuple(anchor.geobox.transform)[:6]
    assert (row.grid_height, row.grid_width) == (3, 4)
    assert row.start_datetime.tzinfo is UTC
    assert row.end_datetime.tzinfo is UTC
    assert vector.crs.to_epsg() == 4326


def test_timeless_anchor_uses_null_time_columns() -> None:
    anchor = GeoAnchor.from_coordinates(-8.05, 112.15, shape=2, resolution=10)

    row = GeoVector.from_anchor(anchor).gdf.iloc[0]

    assert isna(row.start_datetime)
    assert isna(row.end_datetime)
```

- [ ] **Step 2: Run the constructor tests and verify they fail**

Run:

```bash
uv run pytest tests/geodata/test_vector.py -q
```

Expected: failures for missing `empty`, property keywords, and `from_anchor`.

- [ ] **Step 3: Implement empty state, property construction, field validation, and anchor identity**

In `core/vector.py`, add named constants and helpers rather than a schema class:

```python
import hashlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from geosave_engine.geodata.utils.datetime import naive_utc

type VectorField = Literal["time", "grid", "variables"]

_ANCHOR_FIELDS: Final = frozenset({"time", "grid"})
_XARRAY_FIELDS: Final = frozenset({"time", "grid", "variables"})
_DERIVED_COLUMNS: Final = frozenset(
    {
        "anchor_id",
        "start_datetime",
        "end_datetime",
        "grid_crs",
        "grid_transform",
        "grid_height",
        "grid_width",
        "variables",
        "geometry",
        "path",
    }
)


def _anchor_id(anchor: GeoAnchor) -> str:
    span = anchor.timespan
    when = None if span is None else (naive_utc(span[0]), naive_utc(span[1]))
    geobox = anchor.geobox
    grid = (str(geobox.crs), tuple(geobox.transform)[:6], geobox.shape.yx)
    return hashlib.sha1(f"{grid}|{when}".encode()).hexdigest()[:16]


def _utc(value: datetime) -> datetime:
    return naive_utc(value).replace(tzinfo=UTC)


def _checked_fields(
    fields: Collection[VectorField], allowed: frozenset[str]
) -> frozenset[str]:
    selected = frozenset(fields)
    unsupported = sorted(selected - allowed)
    if unsupported:
        raise ValueError(f"unsupported vector fields {unsupported}; choose from {sorted(allowed)}")
    return selected
```

Extend the dataclass and relax only the empty-frame rejection:

```python
@dataclass(frozen=True, eq=False)
class GeoVector:
    gdf: gpd.GeoDataFrame
    source: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def empty(cls, crs: SomeCRS) -> GeoVector:
        return cls(gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs)))

    @property
    def footprint(self) -> Geometry:
        if self.gdf.empty:
            raise ValueError("an empty GeoVector has no footprint")
        return Geometry(self.gdf.geometry.union_all(), crs=self.crs)
```

Preserve persistence context when reprojecting an opened collection:

```python
def to_crs(self, crs: SomeCRS) -> GeoVector:
    if self.crs == crs:
        return self
    return type(self)(self.gdf.to_crs(crs), source=self.source)
```

Make `from_geometry` accept scalar properties after rejecting the active
geometry name:

```python
@classmethod
def from_geometry(
    cls,
    geometry: SomeGeometry,
    *,
    crs: SomeCRS | None = None,
    **properties: Any,
) -> GeoVector:
    if "geometry" in properties:
        raise ValueError("geometry is owned by GeoVector; do not pass it as a property")
    # Keep the existing ODC CRS conflict validation and resolved_crs logic.
    return cls(
        gpd.GeoDataFrame(
            {**{name: [value] for name, value in properties.items()}},
            geometry=[to_shapely(geometry)],
            crs=resolved_crs,
        )
    )
```

Build one anchor row with only selected derived groups:

```python
@classmethod
def from_anchor(
    cls,
    anchor: GeoAnchor,
    *,
    crs: SomeCRS | None = None,
    fields: Collection[VectorField] = ("time", "grid"),
    **properties: Any,
) -> GeoVector:
    selected = _checked_fields(fields, _ANCHOR_FIELDS)
    collisions = sorted(set(properties) & _DERIVED_COLUMNS)
    if collisions:
        raise ValueError(f"properties collide with derived columns {collisions}")
    row: dict[str, Any] = {"anchor_id": _anchor_id(anchor), **properties}
    if "time" in selected:
        span = anchor.timespan
        row.update(
            start_datetime=None if span is None else _utc(span[0]),
            end_datetime=None if span is None else _utc(span[1]),
        )
    if "grid" in selected:
        row.update(
            grid_crs=str(anchor.crs),
            grid_transform=tuple(anchor.geobox.transform)[:6],
            grid_height=anchor.geobox.height,
            grid_width=anchor.geobox.width,
        )
    vector = cls.from_geometry(anchor.geobox.extent, **row)
    return vector if crs is None else vector.to_crs(crs)
```

- [ ] **Step 4: Add xarray registration and laziness tests**

Append tests that cover metadata extraction, semantic geometry, field selection,
reserved names, and zero Dask execution:

```python
def test_xarray_record_keeps_semantic_geometry_and_native_grid() -> None:
    raster = build_raster(times=2)
    semantic = box(112.0, -8.2, 112.2, -8.0)

    vector = GeoVector.from_xarray(
        raster,
        geometry=semantic,
        crs="EPSG:4326",
        path="rasters/prediction.zarr",
        fields=("time", "grid", "variables"),
        model="forest-v1",
    )
    row = vector.gdf.iloc[0]

    assert row.geometry.equals(semantic)
    assert row.grid_crs == str(raster.gs.anchor.crs)
    assert tuple(row.variables) == ("red", "nir")
    assert row.path == "rasters/prediction.zarr"
    assert row.model == "forest-v1"


def test_xarray_registration_does_not_compute_pixels() -> None:
    raster = build_raster().chunk({"y": 1, "x": 1})
    started: list[object] = []

    with dask.callbacks.Callback(pretask=lambda key, *_: started.append(key)):
        vector = GeoVector.from_xarray(raster)

    assert len(vector) == 1
    assert started == []


def test_xarray_fields_are_explicit() -> None:
    vector = GeoVector.from_xarray(build_raster(), fields=())

    assert set(vector.gdf) == {"anchor_id", "geometry"}
    with pytest.raises(ValueError, match="unsupported vector fields"):
        GeoVector.from_xarray(build_raster(), fields=("attrs",))


def test_derived_properties_cannot_be_replaced() -> None:
    with pytest.raises(ValueError, match="anchor_id"):
        GeoVector.from_anchor(
            GeoAnchor.from_coordinates(-8.05, 112.15, shape=2, resolution=10),
            anchor_id="caller-value",
        )
```

- [ ] **Step 5: Implement xarray registration without reading values**

Use only accessors, coordinates, and variable names:

```python
@staticmethod
def _variable_names(data: xr.DataArray | xr.Dataset | xr.DataTree) -> tuple[str, ...]:
    if isinstance(data, xr.DataArray):
        return () if data.name is None else (str(data.name),)
    if isinstance(data, xr.Dataset):
        return tuple(data.data_vars)
    return tuple(
        f"{group}/{name}"
        for group, raster in data.gs.rasters.items()
        for name in raster.data_vars
    )

@classmethod
def from_xarray(
    cls,
    data: xr.DataArray | xr.Dataset | xr.DataTree,
    *,
    geometry: SomeGeometry | None = None,
    crs: SomeCRS | None = None,
    path: str | PathLike[str] | None = None,
    fields: Collection[VectorField] = ("time", "grid"),
    **properties: Any,
) -> GeoVector:
    selected = _checked_fields(fields, _XARRAY_FIELDS)
    anchor = data.gs.anchor
    base = cls.from_anchor(
        anchor,
        fields=tuple(selected - {"variables"}),
        **properties,
    )
    row = base.gdf.drop(columns="geometry").iloc[0].to_dict()
    if "variables" in selected:
        row["variables"] = cls._variable_names(data)
    if path is not None:
        row["path"] = path
    footprint = anchor.geobox.extent if geometry is None else geometry
    vector = cls.from_geometry(footprint, **row)
    return vector if crs is None else vector.to_crs(crs)
```

Keep `PathLike`, xarray types, and `GeoAnchor` imports under `TYPE_CHECKING`
where runtime dispatch does not require them. Import `xarray as xr` at runtime
for the three native type checks.

- [ ] **Step 6: Run focused constructors, existing core smoke tests, and Ruff**

Run:

```bash
uv run pytest tests/geodata/test_vector.py tests/geodata/test_core_smoke.py -q
uv run ruff check src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit constructor behavior**

```bash
git add src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
git commit -m "feat: construct GeoVector spatial records" -- src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

### Task 2: Functional collection growth

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `tests/geodata/test_vector.py`

**Interfaces:**
- Consumes: Task 1 `GeoVector.empty`, constructor output, and `to_crs`.
- Produces: `GeoVector.concat(vectors)` and `GeoVector.upsert(records, *, on)`.

- [ ] **Step 1: Write failing concatenation and upsert tests**

Append:

```python
def test_concat_unions_columns_resets_index_and_keeps_duplicates() -> None:
    first = GeoVector.from_geometry(Point(0, 0), name="first")
    renamed = GeoVector(
        GeoVector.from_geometry(Point(1, 1), score=4)
        .gdf.rename_geometry("footprint")
    )

    combined = GeoVector.concat([first, renamed, first])

    assert list(combined.gdf.index) == [0, 1, 2]
    assert list(combined.gdf.geometry) == [Point(0, 0), Point(1, 1), Point(0, 0)]
    assert list(combined.gdf.name.iloc[[0, 2]]) == ["first", "first"]
    assert combined.gdf.loc[1, "score"] == 4


def test_concat_requires_one_explicit_crs_plane() -> None:
    geographic = GeoVector.from_geometry(Point(0, 0))
    projected = GeoVector.from_geometry(Point(0, 0), crs="EPSG:3857")

    with pytest.raises(ValueError, match="to_crs"):
        GeoVector.concat([geographic, projected])


def test_upsert_replaces_anchor_row_and_appends_new_key() -> None:
    old = GeoVector.from_geometry(Point(0, 0), anchor_id="a", status="old")
    untouched = GeoVector.from_geometry(Point(1, 1), anchor_id="b", status="same")
    replacement = GeoVector.from_geometry(Point(2, 2), anchor_id="a", status="new")
    added = GeoVector.from_geometry(Point(3, 3), anchor_id="c", status="new")
    catalog = GeoVector.concat([old, untouched])

    result = catalog.upsert(GeoVector.concat([replacement, added]), on="anchor_id")

    assert list(result.gdf.anchor_id) == ["b", "a", "c"]
    assert list(result.gdf.status) == ["same", "new", "new"]


@pytest.mark.parametrize("problem", ["missing", "null", "duplicate"])
def test_upsert_rejects_invalid_incoming_identity(problem: str) -> None:
    existing = GeoVector.from_geometry(Point(0, 0), anchor_id="a")
    if problem == "missing":
        incoming = GeoVector.from_geometry(Point(1, 1))
    elif problem == "null":
        incoming = GeoVector.from_geometry(Point(1, 1), anchor_id=None)
    else:
        incoming = GeoVector.concat(
            [
                GeoVector.from_geometry(Point(1, 1), anchor_id="b"),
                GeoVector.from_geometry(Point(2, 2), anchor_id="b"),
            ]
        )

    with pytest.raises((KeyError, ValueError)):
        existing.upsert(incoming, on="anchor_id")
```

- [ ] **Step 2: Run the collection tests and verify missing methods fail**

Run:

```bash
uv run pytest tests/geodata/test_vector.py -q -k "concat or upsert"
```

Expected: failures because `concat` and `upsert` do not exist.

- [ ] **Step 3: Implement canonical native concatenation**

Canonicalize active geometry names on copies so foreign GeoDataFrames compose:

```python
@staticmethod
def _canonical_frame(vector: GeoVector) -> gpd.GeoDataFrame:
    frame = vector.gdf.copy()
    active = frame.active_geometry_name
    if active != "geometry":
        if "geometry" in frame.columns:
            raise ValueError(
                "cannot canonicalize the active geometry because another geometry column exists"
            )
        frame = frame.rename_geometry("geometry")
    return frame

@classmethod
def concat(cls, vectors: Iterable[GeoVector]) -> GeoVector:
    items = list(vectors)
    if not items:
        raise ValueError("concat needs at least one GeoVector")
    crs = items[0].crs
    mismatches = [str(item.crs) for item in items if item.crs != crs]
    if mismatches:
        raise ValueError(
            f"GeoVector CRSs differ from {crs}: {mismatches}; call to_crs explicitly"
        )
    frame = gpd.GeoDataFrame(
        pd.concat([cls._canonical_frame(item) for item in items], ignore_index=True),
        geometry="geometry",
        crs=crs,
    )
    return cls(frame)
```

Import `Iterable` from `collections.abc`. Ensure an empty first vector retains
its declared CRS and an active geometry column after concatenation.

- [ ] **Step 4: Implement explicit keyed replacement**

```python
def upsert(self, records: GeoVector, *, on: str) -> GeoVector:
    if self.crs != records.crs:
        raise ValueError(
            f"GeoVector CRSs differ ({self.crs} and {records.crs}); call to_crs explicitly"
        )
    for label, frame in (("existing", self.gdf), ("incoming", records.gdf)):
        if on not in frame:
            raise KeyError(f"{label} GeoVector has no {on!r} column")
    incoming = records.gdf[on]
    if incoming.isna().any():
        raise ValueError(f"incoming {on!r} keys must not be null")
    duplicates = incoming[incoming.duplicated(keep=False)].tolist()
    if duplicates:
        raise ValueError(f"incoming {on!r} keys must be unique, got {duplicates}")
    retained = GeoVector(self.gdf.loc[~self.gdf[on].isin(incoming)].copy())
    return type(self).concat([retained, records])
```

- [ ] **Step 5: Run collection and constructor tests**

```bash
uv run pytest tests/geodata/test_vector.py -q
uv run ruff check src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit functional collection growth**

```bash
git add src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
git commit -m "feat: compose GeoVector collections" -- src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

### Task 3: Indexed spatial queries with row-relative predicates

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `tests/geodata/test_vector.py`

**Interfaces:**
- Consumes: Task 1 constructors, existing `GeoVector.footprint`, `GeoAnchor.geobox.extent`, and native `.gs.anchor`.
- Produces: `GeoVector.query(target, *, predicate="intersects")`.

- [ ] **Step 1: Write failing query tests, including directional predicates**

```python
def test_query_returns_original_intersecting_rows_in_source_order() -> None:
    collection = GeoVector(
        gpd.GeoDataFrame(
            {"name": ["left", "middle", "right"]},
            geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2), box(8, 0, 10, 2)],
            crs="EPSG:4326",
        )
    )

    result = collection.query(box(1, -1, 3, 3), predicate="intersects")

    assert list(result.gdf.name) == ["left", "middle"]
    assert result.gdf.iloc[0].geometry.equals(collection.gdf.iloc[0].geometry)


def test_query_predicate_describes_each_row_relative_to_target() -> None:
    collection = GeoVector(
        gpd.GeoDataFrame(
            {"name": ["inside", "outside", "container"]},
            geometry=[box(1, 1, 2, 2), box(5, 5, 6, 6), box(-1, -1, 4, 4)],
            crs="EPSG:4326",
        )
    )
    target = box(0, 0, 3, 3)

    assert list(collection.query(target, predicate="within").gdf.name) == ["inside"]
    assert list(collection.query(target, predicate="contains").gdf.name) == ["container"]


def test_query_transforms_only_an_anchor_footprint() -> None:
    collection = GeoVector.from_geometry(
        box(112.0, -8.2, 112.2, -8.0), crs="EPSG:4326", name="place"
    )
    anchor = GeoAnchor.from_coordinates(-8.1, 112.1, shape=4, resolution=100)

    result = collection.query(anchor)

    assert list(result.gdf.name) == ["place"]
    assert result.crs.to_epsg() == 4326


def test_query_empty_collection_preserves_schema() -> None:
    empty = GeoVector.empty("EPSG:4326")

    result = empty.query(box(0, 0, 1, 1))

    assert len(result) == 0
    assert result.crs.to_epsg() == 4326


def test_query_rejects_unknown_predicate() -> None:
    vector = GeoVector.from_geometry(Point(0, 0))

    with pytest.raises(ValueError, match="predicate"):
        vector.query(Point(0, 0), predicate="touch-ish")
```

- [ ] **Step 2: Run query tests and verify the method is absent**

```bash
uv run pytest tests/geodata/test_vector.py -q -k query
```

Expected: failures because `GeoVector.query` does not exist.

- [ ] **Step 3: Implement target-footprint dispatch**

Add one private helper. Check native types before falling back to
`SomeGeometry`; do not inspect pixel values:

```python
def _query_geometry(self, target: Any) -> Geometry:
    from .anchor import GeoAnchor

    if isinstance(target, GeoVector):
        geometry = target.footprint
    elif isinstance(target, GeoAnchor):
        geometry = target.geobox.extent
    elif isinstance(target, (xr.DataArray, xr.Dataset, xr.DataTree)):
        geometry = target.gs.anchor.geobox.extent
    else:
        geometry = GeoVector.from_geometry(target).footprint
    return geometry.to_crs(self.crs)
```

- [ ] **Step 4: Implement candidate indexing and exact row-relative predicates**

Do not pass directional predicates into `sindex.query`, whose operand direction
is easy to reverse. Use it only for bbox candidates:

```python
_PREDICATES: Final = frozenset(
    {"intersects", "within", "contains", "covers", "covered_by"}
)

def query(self, target: Any, *, predicate: str = "intersects") -> GeoVector:
    if predicate not in _PREDICATES:
        raise ValueError(
            f"predicate must be one of {sorted(_PREDICATES)}, got {predicate!r}"
        )
    if self.gdf.empty:
        return type(self)(self.gdf.copy(), source=self.source)
    target_geometry = self._query_geometry(target).geom
    positions = sorted(set(self.gdf.sindex.query(target_geometry)))
    candidates = self.gdf.iloc[positions]
    exact = getattr(candidates.geometry, predicate)(target_geometry)
    return type(self)(candidates.loc[exact].copy(), source=self.source)
```

- [ ] **Step 5: Run query and full vector tests**

```bash
uv run pytest tests/geodata/test_vector.py -q
uv run ruff check src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

Expected: all tests pass, including directional cases.

- [ ] **Step 6: Commit spatial selection**

```bash
git add src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
git commit -m "feat: query GeoVector spatial records" -- src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

### Task 4: Portable and filtered GeoParquet catalogs

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py:105-198`
- Modify: `src/geosave_engine/geodata/utils/io/geoparquet.py:1-105`
- Modify: `src/geosave_engine/geodata/utils/io/__init__.py:186-216`
- Modify: `tests/geodata/test_vector.py`
- Modify: `tests/geodata/test_raster_io_smoke.py`

**Interfaces:**
- Consumes: Task 1 `GeoVector.source`, the canonical optional `path` column, and existing `read_vector`/`to_geoparquet` methods.
- Produces: atomic GeoParquet writes, default covering bboxes, `GeoVector.resolve_path(row)`, and local source-path propagation from `read_vector`.

- [ ] **Step 1: Write portable-path round-trip and relocation tests**

Append to `test_vector.py`:

```python
import shutil
from pathlib import Path


def test_asset_path_round_trips_relative_to_movable_catalog(tmp_path: Path) -> None:
    original = tmp_path / "original"
    (original / "rasters" / "prediction.zarr").mkdir(parents=True)
    catalog = GeoVector.from_geometry(
        Point(0, 0), path=original / "rasters" / "prediction.zarr"
    )

    catalog.to_geoparquet(original / "catalog.parquet")
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    restored = __import__("geosave_engine.geodata", fromlist=["read_vector"]).read_vector(
        moved / "catalog.parquet"
    )

    row = restored.gdf.iloc[0]
    assert row.path == "rasters/prediction.zarr"
    assert restored.resolve_path(row).is_dir()


@pytest.mark.parametrize("escape", ["parent", "symlink"])
def test_unsafe_asset_path_does_not_replace_catalog(
    tmp_path: Path, escape: str
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    safe = GeoVector.from_geometry(Point(0, 0), name="safe")
    target = safe.to_geoparquet(root / "catalog.parquet")
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape == "parent":
        asset = root / ".." / "outside" / "prediction.zarr"
    else:
        (root / "linked").symlink_to(outside, target_is_directory=True)
        asset = root / "linked" / "prediction.zarr"
    unsafe = GeoVector.from_geometry(Point(1, 1), path=asset)

    with pytest.raises(ValueError, match="outside"):
        unsafe.to_geoparquet(target, overwrite=True)

    restored = __import__("geosave_engine.geodata", fromlist=["read_vector"]).read_vector(target)
    assert list(restored.gdf.name) == ["safe"]


def test_resolve_path_requires_source_and_asset() -> None:
    unsaved = GeoVector.from_geometry(Point(0, 0), path="asset.zarr")
    without_asset = GeoVector.from_geometry(Point(0, 0))

    with pytest.raises(ValueError, match="source"):
        unsaved.resolve_path(unsaved.gdf.iloc[0])
    with pytest.raises(KeyError, match="path"):
        without_asset.resolve_path(without_asset.gdf.iloc[0])
```

Use a normal `from geosave_engine.geodata import read_vector` import in the
actual test file; the dynamic import above only keeps this plan snippet
self-contained with the earlier import block.

- [ ] **Step 2: Write atomic write, null path, and bbox-filter smoke tests**

In `tests/geodata/test_raster_io_smoke.py`, add:

```python
def test_geoparquet_catalog_supports_bbox_and_column_filtering(tmp_path):
    vector = GeoVector(
        gpd.GeoDataFrame(
            {"name": ["near", "far"], "path": [None, None]},
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs="EPSG:4326",
        )
    )
    path = vector.to_geoparquet(tmp_path / "catalog.parquet")

    selected = read_vector(path, bbox=(-1, -1, 2, 2), columns=["name", "geometry"])

    assert list(selected.gdf.name) == ["near"]
    assert selected.source == path


def test_geoparquet_write_replaces_stale_staging_file(tmp_path):
    target = tmp_path / "catalog.parquet"
    staged = tmp_path / ".catalog.staging.parquet"
    staged.write_text("interrupted")

    GeoVector.from_geometry(Point(0, 0)).to_geoparquet(target)

    assert target.is_file()
    assert not staged.exists()


def test_empty_geoparquet_catalog_round_trips(tmp_path):
    path = GeoVector.empty("EPSG:4326").to_geoparquet(
        tmp_path / "catalog.parquet"
    )

    restored = read_vector(path)

    assert len(restored) == 0
    assert restored.crs.to_epsg() == 4326
```

- [ ] **Step 3: Run persistence tests and verify current behavior fails**

```bash
uv run pytest tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py -q -k "path or parquet or staging or bbox"
```

Expected: failures for absent source context/resolution, unnormalized paths, and
non-atomic stage handling.

- [ ] **Step 4: Implement path normalization on a copied frame**

In `utils/io/geoparquet.py`, add a private helper used only by
`GeoVector.to_geoparquet`:

```python
def portable_paths(
    gdf: gpd.GeoDataFrame, destination: str | PathLike[str]
) -> gpd.GeoDataFrame:
    if "path" not in gdf:
        return gdf
    target = Path(destination)
    root = target.parent.resolve()
    frame = gdf.copy()

    def stored(value: Any) -> Any:
        if pd.isna(value):
            return value
        if isinstance(value, str) and "://" in value:
            raise ValueError(f"URI asset path {value!r} is unsupported")
        candidate = Path(value)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"asset path {value!r} resolves outside catalog root {root}"
            ) from error

    frame["path"] = frame["path"].map(stored)
    return frame
```

Import pandas and `PathLike`. Do not mutate `gdf`.

- [ ] **Step 5: Make GeoParquet writing atomic**

Replace the direct write tail in `geoparquet.write`:

```python
import os

target = Path(path)
# Keep suffix and overwrite validation before touching staging state.
staged = target.with_name(f".{target.stem}.staging{target.suffix}")
try:
    gdf.to_parquet(staged, **write_options, **parquet_options)  # type: ignore
    os.replace(staged, target)
except BaseException:
    staged.unlink(missing_ok=True)
    raise
return target
```

An existing stale staging file may be replaced because it is never the durable
catalog. The target is replaced only after a complete staged write.

- [ ] **Step 6: Enable covering bboxes, source propagation, and resolution**

In `GeoVector.to_geoparquet`, prepare paths and default the covering bbox:

```python
options = dict(options)
options.setdefault("write_covering_bbox", True)
frame = geoparquet.portable_paths(self.gdf, path)
return geoparquet.write(frame, path, overwrite=overwrite, **options)
```

In `read_vector`, wrap each local reader result with source context:

```python
def _local_source(source: str | PathLike[str]) -> Path | None:
    spelled = str(source)
    return None if "://" in spelled else Path(source).resolve()

# For each supported vector suffix:
return GeoVector(reader.read(source, **options), source=_local_source(source))
```

In `GeoVector`:

```python
def resolve_path(self, row: Mapping[str, Any]) -> Path:
    if "path" not in row or pd.isna(row["path"]):
        raise KeyError("row has no path asset reference")
    if self.source is None:
        raise ValueError("GeoVector has no local source path to resolve against")
    root = self.source.parent.resolve()
    resolved = (root / Path(row["path"])).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"asset path {row['path']!r} resolves outside {root}") from error
    return resolved
```

- [ ] **Step 7: Run persistence, path safety, and existing I/O tests**

```bash
uv run pytest tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py tests/geodata/pipeline/test_manifest.py -q
uv run ruff check src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py
```

Expected: new and existing tests pass. The old Manifest tests still pass until
Task 7 removes that surface.

- [ ] **Step 8: Commit durable catalog behavior**

```bash
git add src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py
git commit -m "feat: persist portable GeoVector catalogs" -- src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io/geoparquet.py src/geosave_engine/geodata/utils/io/__init__.py tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py
```

### Task 5: Explicit flag raster polygonization

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `tests/geodata/test_vector.py`

**Interfaces:**
- Consumes: Task 1 `GeoVector` constructors, `DataArray.gs.geobox`, registered `Nodata`, and Rasterio `features.shapes`.
- Produces: `GeoVector.vectorize(flags, *, value_name="value", mask=None, connectivity=4)`.

- [ ] **Step 1: Write failing polygonization and eager-boundary tests**

```python
import numpy as np
from dask.callbacks import Callback

from geosave_engine.geodata.core.array import array


def test_vectorize_creates_one_row_per_connected_flag_region(raster) -> None:
    flags = array(
        np.array([[1, 0, 1], [1, 0, 0], [2, 2, 0]], dtype="uint8"),
        GeoAnchor.from_bbox(
            raster.gs.geobox.boundingbox, shape=(3, 3), crs=raster.gs.crs
        ).geobox,
        nodata=0,
    )

    vector = GeoVector.vectorize(flags, value_name="class_id")

    assert list(vector.gdf.class_id) == [1, 1, 2]
    assert vector.crs == flags.gs.crs
    assert vector.gdf.geometry.is_valid.all()


def test_vectorize_computes_dask_flags_explicitly(raster) -> None:
    flags = raster.red.chunk({"y": 1, "x": 1})
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        GeoVector.vectorize(flags)

    assert started


def test_vectorize_requires_one_spatial_plane(raster) -> None:
    flags = raster.red.expand_dims(time=["2025-01-01"])

    with pytest.raises(ValueError, match="two-dimensional"):
        GeoVector.vectorize(flags)


def test_vectorize_rejects_unsupported_values(raster) -> None:
    flags = raster.red.astype("complex64")

    with pytest.raises(ValueError, match="dtype"):
        GeoVector.vectorize(flags)


def test_vectorize_rejects_same_shape_mask_on_another_grid(raster) -> None:
    shifted = raster.red.assign_coords(x=raster.x + 10)

    with pytest.raises(ValueError, match="exact flags grid"):
        GeoVector.vectorize(raster.red, mask=shifted)
```

- [ ] **Step 2: Run vectorize tests and verify the method is absent**

```bash
uv run pytest tests/geodata/test_vector.py -q -k vectorize
```

Expected: failures because `GeoVector.vectorize` does not exist.

- [ ] **Step 3: Implement validated two-dimensional value preparation**

Use the exact ODC dimension order and registered nodata:

```python
@staticmethod
def _polygonize_values(flags: xr.DataArray) -> tuple[np.ndarray, np.ndarray, GeoBox]:
    geobox = flags.gs.geobox
    if not isinstance(geobox, GeoBox):
        raise ValueError("flags carry no locatable grid")
    if flags.ndim != 2 or tuple(flags.dims) != tuple(geobox.dimensions):
        raise ValueError(
            f"vectorize needs one two-dimensional spatial plane on {geobox.dimensions}; "
            f"select extra dimensions from {flags.dims} first"
        )
    values = np.asarray(flags.compute().values)
    if values.dtype == np.bool_:
        values = values.astype("uint8")
    allowed = {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("float32"),
    }
    if values.dtype not in allowed:
        raise ValueError(f"raster polygonization does not support dtype {values.dtype}")
    valid = np.isfinite(values)
    nodata = flags.gs.attrs.root.get(attrs.Nodata)
    if nodata is not None and nodata.fill_value is not None:
        valid &= values != nodata.fill_value
    return values, valid, geobox
```

Import `numpy`, `GeoBox`, and `geosave_engine.geodata.attrs as attrs`.

- [ ] **Step 4: Implement Rasterio polygonization and explicit masks**

```python
@classmethod
def vectorize(
    cls,
    flags: xr.DataArray,
    *,
    value_name: str = "value",
    mask: xr.DataArray | np.ndarray | None = None,
    connectivity: Literal[4, 8] = 4,
) -> GeoVector:
    from rasterio.features import shapes
    from shapely.geometry import shape

    if value_name == "geometry":
        raise ValueError("value_name must not replace the geometry column")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    values, valid, geobox = cls._polygonize_values(flags)
    if mask is not None:
        if isinstance(mask, xr.DataArray):
            if mask.gs.geobox != geobox:
                raise ValueError("mask must be on the exact flags grid")
            selected = np.asarray(mask.compute().values)
        else:
            selected = np.asarray(mask)
        if selected.shape != values.shape:
            raise ValueError(
                f"mask shape {selected.shape} does not match flags {values.shape}"
            )
        valid &= selected.astype(bool)
    features = list(
        shapes(values, mask=valid, transform=geobox.transform, connectivity=connectivity)
    )
    if not features:
        return cls.empty(geobox.crs)
    scalar = values.dtype.type
    return cls(
        gpd.GeoDataFrame(
            {value_name: [scalar(value).item() for _, value in features]},
            geometry=[shape(geometry) for geometry, _ in features],
            crs=geobox.crs,
        )
    )
```

The explicit scalar cast keeps integer flags as integers rather than accepting
Rasterio's generic Python-float representation.

- [ ] **Step 5: Run polygonization and all vector tests**

```bash
uv run pytest tests/geodata/test_vector.py -q
uv run ruff check src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit flag polygonization**

```bash
git add src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
git commit -m "feat: vectorize flag rasters" -- src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

### Task 6: Rasterize vector properties onto exact native grids

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `tests/geodata/test_vector.py`

**Interfaces:**
- Consumes: `GeoAnchor`, native xarray `.gs.anchor`, `GeoVector.to_crs`, Rasterio `features.rasterize`, and the public `array(pixels, geobox, nodata=...)` constructor.
- Produces: `GeoVector.rasterize(like, *, column=None, fill=0, dtype=None, all_touched=False)` returning a geolocated `DataArray`.

- [ ] **Step 1: Write failing rasterization, overlap, and validation tests**

```python
def test_rasterize_values_back_onto_reference_grid(raster) -> None:
    flags = array(
        np.array([[1, 1], [2, 0]], dtype="uint8"),
        raster.gs.geobox,
        nodata=0,
    )
    vector = GeoVector.vectorize(flags, value_name="class_id")

    restored = vector.rasterize(
        like=raster,
        column="class_id",
        fill=0,
        dtype="uint8",
    )

    np.testing.assert_array_equal(restored.values, flags.values)
    assert restored.gs.geobox == raster.gs.geobox
    assert restored.name == "class_id"


def test_rasterize_without_column_returns_boolean_presence(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.geobox.extent)

    mask = vector.rasterize(like=raster)

    assert mask.dtype == np.dtype("bool")
    assert mask.values.all()


def test_rasterize_later_rows_win_on_overlap(raster) -> None:
    bounds = raster.gs.geobox.extent.geom
    vector = GeoVector(
        gpd.GeoDataFrame(
            {"class_id": [1, 2]},
            geometry=[bounds, bounds],
            crs=raster.gs.crs,
        )
    )

    result = vector.rasterize(like=raster, column="class_id", dtype="uint8")

    assert (result.values == 2).all()


def test_rasterize_rejects_values_outside_dtype(raster) -> None:
    vector = GeoVector.from_geometry(
        raster.gs.geobox.extent, class_id=300
    )

    with pytest.raises(ValueError, match="uint8"):
        vector.rasterize(like=raster, column="class_id", dtype="uint8")


def test_rasterize_empty_vector_returns_only_fill(raster) -> None:
    empty = GeoVector.empty(raster.gs.crs)

    result = empty.rasterize(like=raster, fill=0)

    assert not result.values.any()
```

- [ ] **Step 2: Run rasterize tests and verify the method is absent**

```bash
uv run pytest tests/geodata/test_vector.py -q -k rasterize
```

Expected: failures because `GeoVector.rasterize` does not exist.

- [ ] **Step 3: Implement exact target-grid extraction and safe value casting**

```python
@staticmethod
def _target_geobox(like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree) -> GeoBox:
    from .anchor import GeoAnchor

    geobox = like.geobox if isinstance(like, GeoAnchor) else like.gs.anchor.geobox
    if geobox.crs is None:
        raise ValueError("rasterize target carries no CRS")
    return geobox

@staticmethod
def _burn_values(series: pd.Series, dtype: np.dtype[Any]) -> np.ndarray:
    if series.isna().any():
        raise ValueError("rasterize values must not be null")
    try:
        converted = series.to_numpy().astype(dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"values cannot be represented by {dtype}") from error
    original = series.to_numpy()
    if not np.array_equal(converted.astype(original.dtype), original):
        raise ValueError(f"values cannot be represented exactly by {dtype}")
    return converted
```

For object/string properties, fail with the same representability error rather
than inventing a category registry.

- [ ] **Step 4: Implement ordered Rasterio burning and native xarray output**

```python
def rasterize(
    self,
    like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
    *,
    column: str | None = None,
    fill: int | float | bool = 0,
    dtype: DTypeLike | None = None,
    all_touched: bool = False,
) -> xr.DataArray:
    from rasterio.features import rasterize as burn

    from .array import array

    geobox = self._target_geobox(like)
    frame = self.to_crs(geobox.crs).gdf
    if column is None:
        output_dtype = np.dtype("uint8")
        values = np.ones(len(frame), dtype=output_dtype)
        output_name = "mask"
        output_fill: int | float | bool = bool(fill)
    else:
        if column not in frame:
            raise KeyError(f"GeoVector has no {column!r} column")
        output_dtype = np.dtype(dtype or np.result_type(frame[column].dtype, type(fill)))
        values = self._burn_values(frame[column], output_dtype)
        output_name = column
        output_fill = self._burn_values(pd.Series([fill]), output_dtype)[0].item()
    if frame.empty:
        pixels = np.full(geobox.shape.yx, output_fill, dtype=output_dtype)
    else:
        pixels = burn(
            zip(frame.geometry, values, strict=True),
            out_shape=geobox.shape.yx,
            transform=geobox.transform,
            fill=output_fill,
            dtype=output_dtype,
            all_touched=all_touched,
        )
    if column is None:
        pixels = pixels.astype(bool)
    return array(pixels, geobox).rename(output_name)
```

The same `_burn_values` check validates `fill` before Rasterio is invoked.

- [ ] **Step 5: Run raster conversion and existing raster tests**

```bash
uv run pytest tests/geodata/test_vector.py tests/geodata/test_raster.py tests/geodata/transform/test_nodata.py -q
uv run ruff check src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit rasterization**

```bash
git add src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
git commit -m "feat: rasterize GeoVector properties" -- src/geosave_engine/geodata/core/vector.py tests/geodata/test_vector.py
```

### Task 7: Remove Manifest and verify the complete public flow

**Files:**
- Delete: `src/geosave_engine/geodata/pipeline/manifest.py`
- Delete: `src/geosave_engine/geodata/pipeline/__init__.py`
- Delete: `tests/geodata/pipeline/test_manifest.py`
- Modify: `src/geosave_engine/geodata/core/vector.py` docstrings and examples
- Modify: `tests/geodata/test_vector.py`

**Interfaces:**
- Consumes: Tasks 1-6 complete `GeoVector` surface.
- Produces: one public spatial collection abstraction with no `Manifest` compatibility path.

- [ ] **Step 1: Add the end-to-end replacement test before deleting Manifest**

Append one smoke test demonstrating the complete replacement:

```python
def test_catalog_register_query_upsert_and_relocate(tmp_path, raster) -> None:
    root = tmp_path / "dataset"
    asset = root / "rasters" / "prediction.zarr"
    asset.mkdir(parents=True)
    labels = GeoVector(
        gpd.GeoDataFrame(
            {"label": ["inside", "outside"]},
            geometry=[raster.gs.geobox.extent.geom, box(0, 0, 1, 1)],
            crs=raster.gs.crs,
        )
    )
    record = GeoVector.from_xarray(raster, path=asset)
    catalog = GeoVector.concat([labels, record])

    matches = labels.query(raster)
    updated = catalog.upsert(
        GeoVector.from_xarray(raster, path=asset, status="complete"),
        on="anchor_id",
    )
    path = updated.to_geoparquet(root / "catalog.parquet")
    reopened = read_vector(path)

    assert list(matches.gdf.label) == ["inside"]
    assert len(reopened.gdf.loc[reopened.gdf.anchor_id.notna()]) == 1
    assert reopened.resolve_path(
        reopened.gdf.loc[reopened.gdf.anchor_id.notna()].iloc[0]
    ).is_dir()
```

When concatenating keyed and unkeyed rows, `upsert` cannot require non-null keys
in the existing collection; it only replaces existing non-null keys matching
the incoming records. This test locks that mixed-catalog behavior.

- [ ] **Step 2: Run the end-to-end smoke test**

```bash
uv run pytest tests/geodata/test_vector.py::test_catalog_register_query_upsert_and_relocate -q
```

Expected: PASS before removing Manifest.

- [ ] **Step 3: Delete the duplicate pipeline abstraction**

Remove:

```text
src/geosave_engine/geodata/pipeline/manifest.py
src/geosave_engine/geodata/pipeline/__init__.py
tests/geodata/pipeline/test_manifest.py
```

Do not alter `src/geosave_engine/templates/boilerplate/scripts/ingest_imagery.py`:
its `IngestManifest` is a generated-workspace CSV/XLSX execution ledger, not the
GeoParquet spatial catalog removed here.

- [ ] **Step 4: Replace live documentation and import references**

Search:

```bash
rg -n "geodata\.pipeline|from .*pipeline import Manifest|\bManifest\(" src tests --glob '!**/*.ipynb'
```

Expected after deletion: no library/test reference to the removed GeoParquet
`Manifest`; only the explicitly separate generated-workspace `IngestManifest`
may remain.

Update `GeoVector` class and method docstrings to show this public flow:

```python
record = GeoVector.from_xarray(prediction, path="rasters/prediction.zarr")
catalog = GeoVector.concat([catalog, record])
matches = labels.query(prediction)
catalog.to_geoparquet("dataset/catalog.parquet")
```

- [ ] **Step 5: Run focused geodata verification**

```bash
uv run pytest tests/geodata/test_vector.py tests/geodata/test_core_smoke.py tests/geodata/test_raster_io_smoke.py tests/geodata/test_raster.py -q
uv run pytest tests/geodata -q
```

Expected: all focused tests and the full geodata suite pass.

- [ ] **Step 6: Run formatting, static checks, and the full suite**

```bash
uv run ruff check .
uv run pytest -q
```

Expected: Ruff reports no errors and the full suite passes. If persistence tests
fail only because restricted background sockets or async permissions are denied,
rerun the exact failing tests with the required local permission before treating
the failure as a product defect.

- [ ] **Step 7: Inspect the final change boundary**

```bash
git status --short
git diff --check
git diff --stat HEAD -- src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io src/geosave_engine/geodata/pipeline tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py tests/geodata/pipeline
```

Expected: only task-owned paths are part of the implementation; unrelated
working-tree changes remain untouched.

- [ ] **Step 8: Commit the Manifest removal and integrated behavior**

```bash
git add src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io src/geosave_engine/geodata/pipeline tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py tests/geodata/pipeline
git commit -m "refactor: replace Manifest with GeoVector catalogs" -- src/geosave_engine/geodata/core/vector.py src/geosave_engine/geodata/utils/io src/geosave_engine/geodata/pipeline tests/geodata/test_vector.py tests/geodata/test_raster_io_smoke.py tests/geodata/pipeline
```

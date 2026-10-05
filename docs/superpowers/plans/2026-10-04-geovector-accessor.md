# GeoVector Accessor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `GeoVector` from a wrapper instance into the `gs` accessor of a plain `GeoDataFrame`, give `geodata/stac` ownership of what a STAC row is, and add `to_xarray` and `query(time=)`.

**Architecture:** `GeoVector` is registered with `pd.api.extensions.register_dataframe_accessor("gs")`, and a `GeoDataFrame` stand-in declared under `TYPE_CHECKING` gives `.gs` its autocomplete, as `Dataset`, `DataArray`, and `DataTree` already do. Every GeoSave function that returns a vector returns that stand-in. STAC row rules move into `geodata/stac/item.py`, which `core/vector.py`, the GeoParquet writer, and the manifest task all import.

**Tech Stack:** Python 3.12, GeoPandas, pandas, xarray, odc-geo, pystac, stac-geoparquet, pytest, basedpyright, Ruff

**Spec:** `docs/superpowers/specs/2026-10-04-geovector-accessor-design.md`, Part B.

## Revisions agreed after this plan was written

These override the task text below wherever the two differ.

- No `check_table`. The unique-`id` and EPSG:4326 rules stay inside the GeoParquet writer's `stored_assets`, unchanged. `stac/item.py` owns only the row's content: constants, `ITEM_COLUMNS`, `item`, `sources`.
- No `check_geometries`. `rasterize` validates null, empty, and invalid geometries inline; the writers and classmethods validate nothing.
- `query(target: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree, *, predicate="intersects")`. It takes an anchor, or an xarray object whose anchor is used, and no bare geometry or vector. Time is compared only when the anchor has a timespan and the table has `datetime`, `start_datetime`, and `end_datetime`; there is no `time=` argument.
- New names use `validate`, never `check`.

## Global Constraints

- No compatibility alias. `GeoVector(frame)`, `.gdf`, `from_anchor`, and the `assets`, `fields`, and `crs` arguments of `from_xarray` are removed, not deprecated.
- A row is read from files: `from_assets` opens the paths it is given, and `from_xarray` finds the path an object was read from and calls `from_assets`. Both return one STAC item in EPSG:4326.
- Accessor construction raises `AttributeError`, which pandas expects, for a non-`GeoDataFrame`, a missing active geometry column, or a missing CRS.
- Null, empty, and invalid geometries raise `ValueError` in `rasterize`, the three writers, and every `GeoVector` classmethod that builds a frame. Reading a file does not check them.
- `geodata` imports no torch. `model` does not import `ml`.
- `geodata/stac/__init__.py` imports its client eagerly, so `core/vector.py` and `utils/io/geoparquet.py` import `geodata.stac.item` inside functions, never at module level.
- Do not commit, stage, stash, checkout, or restore anything. The files carry the user's uncommitted work.
- Run commands with `uv run`. Do not touch `.ipynb` files.
- Docstrings are concise Google style. Comments explain domain constraints only.

## Review Focus

- A `GeoDataFrame` whose active geometry column is not named `geometry` must still work through `gs.query`, `gs.rasterize`, `gs.footprint`, and `GeoVector.concat`. Task 2 pins this.
- A frame produced by a native operation (`frame[mask]`, `pd.concat`) must still answer `.gs` at runtime. Task 2 pins this.
- `to_xarray` on a row read back from GeoParquet must open the resolved hrefs, including when the manifest was moved with its samples. Task 3 pins this.
- `query(time=)` must match a row that carries only `datetime`, with null `start_datetime` and `end_datetime`. Task 4 pins this.
- `crop` with a vector in another CRS must give the same pixels as cropping with that vector reprojected first. Task 5 pins this.

## File Structure

| File | Change |
| --- | --- |
| `src/geosave_engine/geodata/stac/item.py` | New: STAC version, extension, media types, item columns, asset builder, row builder, provider sources, table rules |
| `src/geosave_engine/geodata/utils/io/gdal.py`, `layout.py` | Readers record the origin; `write_tree` returns its path; `read_raster` opens a tree directory |
| `src/geosave_engine/geodata/transform/nodata.py`, `packing.py` | Results forget the origin |
| `src/geosave_engine/geodata/core/stack.py` | `to_cog` returns its path |
| `src/geosave_engine/geodata/core/vector.py` | `GeoVector` becomes the accessor; `check_geometries`; `to_xarray`; `query(time=)` |
| `src/geosave_engine/geodata/__init__.py` | `GeoDataFrame` stand-in, exported |
| `src/geosave_engine/__init__.py` | Export `GeoDataFrame` |
| `src/geosave_engine/geodata/utils/io/__init__.py` | `read_vector` returns the frame |
| `src/geosave_engine/geodata/utils/io/geoparquet.py` | Table rules come from `stac.item` |
| `src/geosave_engine/geodata/transform/vector.py` | `vectorize`, `rasterize`, `crop` take and return frames; `crop` reprojects the vector |
| `src/geosave_engine/geodata/core/anchor.py` | `from_geometry` reads the footprint through `.gs` |
| `src/geosave_engine/geodata/core/base.py` | `crop` annotation |
| `src/geosave_engine/workflow/tasks/manifest.py` | `properties=`, `ITEM_COLUMNS` from `stac.item` |
| `src/geosave_engine/workflow/configs/anchor.py` | Footprint through `.gs` |
| `src/geosave_engine/ml/segmentation/supervised/data.py` | Opens samples with `to_xarray` |
| `tests/geodata/stac/test_item.py` | New |
| `tests/geodata/core/test_vector.py` and every test listed in Task 2 | Migrated spelling |
| `examples/data/dw_imagery/manifest.parquet` | Regenerated as a STAC table |
| `docs/guides/workflows.md` | New spelling |

## Spelling migration, used by Tasks 1 and 2

| Old | New |
| --- | --- |
| `GeoVector(frame)` | `frame` |
| `vector.gdf` | `vector` |
| `len(vector)` | `len(vector)` (GeoPandas) |
| `vector.crs` | `vector.gs.crs` |
| `vector.footprint` | `vector.gs.footprint` |
| `vector.query(...)`, `.upsert(...)`, `.rasterize(...)` | `vector.gs.query(...)`, `.gs.upsert(...)`, `.gs.rasterize(...)` |
| `vector.to_geojson(...)`, `.to_geopackage(...)`, `.to_geoparquet(...)` | `vector.gs.to_geojson(...)`, `.gs.to_geopackage(...)`, `.gs.to_geoparquet(...)` |
| `vector.to_crs(...)` | `vector.to_crs(...)` (GeoPandas) |
| `GeoVector.from_geometry(g, name="a")` | `GeoVector.from_geometry(g, properties={"name": "a"})` |
| `GeoVector.from_xarray(d, assets=a, model="m")` | `GeoVector.from_assets(a, properties={"model": "m"})`, with `a` naming written files |
| `GeoVector.from_anchor(...)` | removed |
| annotation `GeoVector` on a value | `gpd.GeoDataFrame` |

Apply each row by reading the call site and editing it. Do not use a regex or scripted rename across files.

---

### Task 1: `geodata/stac/item.py` owns the STAC row, and `from_assets` builds it

`GeoVector` stays a wrapper in this task. A row is now read from files.

**Files:**
- Create: `src/geosave_engine/geodata/stac/item.py`
- Create: `tests/geodata/stac/test_item.py`
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `src/geosave_engine/geodata/utils/io/geoparquet.py`
- Modify: `src/geosave_engine/workflow/tasks/manifest.py`
- Modify: `tests/geodata/core/test_vector.py`, `tests/workflow/flows/test_prepare_dense_data.py`, `tests/ml/segmentation/supervised/test_data.py`

**Interfaces:**
- Consumes: `data.gs.anchor`, `data.gs.rasters`, `data.gs.variables`, `raster.gs.attrs.root.get(StacMetadata)`, `StacMetadata.merge`, `io.read_raster(href, chunks="auto")`, `stack(rasters)`.
- Produces, in `geosave_engine.geodata.stac.item`:
  - `ITEM_COLUMNS: tuple[str, ...]`, which includes `"sources"`.
  - `item(data, *, assets: Mapping[str, Asset], id: str | None = None, datetime: dt.datetime | None = None) -> dict[str, object]`
  - `sources(data) -> list[dict[str, object]] | None`
  - `check_table(frame: gpd.GeoDataFrame) -> None`
  - `type Asset = str | PathLike[str] | Mapping[str, object]`
- Produces, on `GeoVector`:
  - `from_assets(assets: Mapping[str, Asset] | str | PathLike[str], *, id=None, datetime=None, geometry=None, properties=None)`
  - `from_geometry(geometry, *, crs=None, properties=None)`
  - `from_xarray`, `from_anchor`, and `AnchorField` are removed in this task. Task 1b brings `from_xarray` back as path discovery.

- [ ] **Step 1: Write the failing tests**

Create `tests/geodata/stac/test_item.py` with the five tests below, then add the `sources` tests after them.

```python
from __future__ import annotations

from datetime import UTC, datetime

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geosave_engine.geodata.attrs.models.stac import StacItem, StacMetadata
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.stac.item import ITEM_COLUMNS, check_table, item, sources

from tests.geodata.conftest import build_raster


def test_an_item_lists_its_columns_in_stac_order() -> None:
    row = item(build_raster(times=2), assets={"image": "image.tif"})

    assert list(row)[:6] == [
        "id",
        "type",
        "stac_version",
        "stac_extensions",
        "links",
        "datetime",
    ]
    assert list(row)[-2:] == ["assets", "sources"]
    assert set(row) <= set(ITEM_COLUMNS)
    assert row["proj:code"] == "EPSG:32749"
    assert tuple(row["proj:shape"]) == (2, 2)


def test_an_asset_is_described_from_the_raster_it_names() -> None:
    row = item(build_raster(times=1), assets={"image": "scene/image.tif"})

    assert row["assets"]["image"] == {
        "href": "scene/image.tif",
        "type": "image/tiff; application=geotiff",
        "roles": ["data"],
        "bands": [{"name": "red"}, {"name": "nir"}],
    }


def test_a_timeless_item_needs_a_datetime() -> None:
    with pytest.raises(ValueError, match="datetime="):
        item(build_raster(), assets={"image": "image.tif"})

    row = item(
        build_raster(),
        assets={"image": "image.tif"},
        datetime=datetime(2025, 6, 1, tzinfo=UTC),
    )

    assert row["datetime"] == datetime(2025, 6, 1, tzinfo=UTC)


def _table(ids: list[object], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": ids, "assets": [{} for _ in ids]},
        geometry=[Point(index, 0) for index in range(len(ids))],
        crs=crs,
    )


def test_a_table_needs_ids_that_identify() -> None:
    check_table(_table(["a", "b"]))
    with pytest.raises(ValueError, match="unique"):
        check_table(_table(["a", "a"]))
    with pytest.raises(ValueError, match="null"):
        check_table(_table(["a", None]))
    with pytest.raises(ValueError, match="'id' column"):
        check_table(_table(["a"]).drop(columns="id"))


def test_a_table_is_in_longitude_and_latitude() -> None:
    with pytest.raises(ValueError, match="EPSG:4326"):
        check_table(_table(["a"], crs="EPSG:3857"))


def _loaded(prefix: str, days: tuple[int, ...], **properties: object) -> StacMetadata:
    return StacMetadata(
        stac_items=tuple(
            StacItem(
                id=f"{prefix}_{day:02d}",
                datetime=datetime(2025, 6, day, 2, 30),
                properties={"eo:cloud_cover": 10.0 * day, **properties},
            )
            for day in days
        )
    )


def test_sources_list_every_layers_provider_items_once() -> None:
    raster = build_raster(times=1)
    optical = raster[["red"]].gs.rebase(_loaded("S2A", (1, 3), platform="sentinel-2a"))
    radar = raster[["nir"]].gs.rebase(_loaded("S1A", (2,), platform="sentinel-1a"))

    found = sources(stack({"optical": optical, "radar": radar}))

    assert found == [
        {
            "eo:cloud_cover": 10.0,
            "platform": "sentinel-2a",
            "id": "S2A_01",
            "datetime": "2025-06-01T02:30:00",
        },
        {
            "eo:cloud_cover": 30.0,
            "platform": "sentinel-2a",
            "id": "S2A_03",
            "datetime": "2025-06-03T02:30:00",
        },
        {
            "eo:cloud_cover": 20.0,
            "platform": "sentinel-1a",
            "id": "S1A_02",
            "datetime": "2025-06-02T02:30:00",
        },
    ]


def test_a_raster_loaded_from_no_catalog_has_no_sources() -> None:
    assert sources(build_raster(times=1)) is None
```

In `tests/geodata/core/test_vector.py`:

- Delete `test_anchor_record_has_exact_grid_and_time`, `test_timeless_anchor_uses_null_time_columns`, `test_xarray_fields_are_explicit`, `test_derived_properties_cannot_be_replaced`, and `test_xarray_registration_does_not_compute_pixels`.
- Remove `AnchorField` from the `core.vector` import.
- Apply the `from_geometry` row of the migration table to every call in the file.
- Add this helper near the top, and rewrite every remaining `GeoVector.from_xarray(raster_expression, assets={key: href}, ...)` call as `GeoVector.from_assets({key: _written(tmp_path, raster_expression, href)}, ...)`, adding the `tmp_path: Path` fixture to the test where it lacks it:

```python
def _written(root: Path, raster: xr.Dataset, name: str = "image.tif") -> Path:
    """Write one raster the way a sample layer is written and return its path."""
    from geosave_engine.geodata.utils.io import geotiff, zarr

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zarr":
        zarr.write(raster, path)
    else:
        flat = raster.squeeze("time", drop=False) if "time" in raster.dims else raster
        geotiff.write_cog(flat, path)
    return path
```

  A test that built a row from a two-instant raster and a `.tif` href writes a one-instant raster instead, or a `.zarr`, since a GeoTIFF layer holds one instant. Tests that assert on the stored href (`_record`, `_href`, the relocation tests) pass the written path and assert against it.
- Add:

```python
def test_a_row_is_read_from_the_files_it_names(tmp_path: Path) -> None:
    raster = build_raster(times=1)
    path = _written(tmp_path, raster, "scene/image.tif")
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        vector = GeoVector.from_assets(path)

    row = vector.gdf.iloc[0]
    assert vector.crs.to_epsg() == 4326
    assert list(row.assets) == ["image"]
    assert row.assets["image"]["href"] == str(path)
    assert row.assets["image"]["bands"] == [{"name": "red"}, {"name": "nir"}]
    assert row["proj:code"] == "EPSG:32749"
    assert started == []


def test_a_file_that_does_not_exist_cannot_be_registered(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        GeoVector.from_assets(tmp_path / "missing.tif")


def test_layers_on_different_grids_are_refused(tmp_path: Path) -> None:
    raster = build_raster(times=1)
    whole = _written(tmp_path, raster[["red"]], "whole.tif")
    part = _written(tmp_path, raster[["nir"]].isel(x=slice(0, 1)), "part.tif")

    with pytest.raises(ValueError, match="grid"):
        GeoVector.from_assets({"optical": whole, "label": part})


def test_a_caller_column_may_share_a_name_with_an_argument(tmp_path: Path) -> None:
    vector = GeoVector.from_assets(
        _written(tmp_path, build_raster(times=1)),
        properties={"crs": "caller", "geometry_source": "survey"},
    )

    assert vector.gdf.loc[0, "crs"] == "caller"
    assert vector.gdf.loc[0, "geometry_source"] == "survey"


def test_a_caller_column_cannot_replace_an_item_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_datetime"):
        GeoVector.from_assets(
            _written(tmp_path, build_raster(times=1)),
            properties={"start_datetime": "caller"},
        )
```

In `tests/ml/segmentation/supervised/test_data.py::_manifest`, the samples are already written to disk before the row is built; replace `GeoVector.from_xarray(sample, assets=paths, ...)` with `GeoVector.from_assets(paths, ...)`, keeping `id` and moving any caller keyword into `properties=`.

In `tests/workflow/flows/test_prepare_dense_data.py`, replace both `GeoVector.from_anchor(..., sample_id="old").to_geoparquet(manifest)` lines with:

```python
    GeoVector.from_geometry(
        Point(0, 0), properties={"sample_id": "old"}
    ).to_geoparquet(manifest)
```

and add `from shapely.geometry import Point` if absent. These tests only need a pre-existing manifest file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/stac/test_item.py tests/geodata/core/test_vector.py tests/workflow/flows/test_prepare_dense_data.py tests/ml -q`
Expected: `test_item.py` fails to collect with `ModuleNotFoundError`; the migrated calls fail with `AttributeError: ... has no attribute 'from_assets'` or `TypeError` on `properties`.

- [ ] **Step 3: Create `stac/item.py`**

```python
"""What one STAC item row is, and what a table of them must satisfy."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from os import PathLike
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

import xarray as xr

from geosave_engine.geodata.attrs.models.stac import StacMetadata

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd

# One raster of a record: where it is stored, or a STAC asset stating `href`.
type Asset = str | PathLike[str] | Mapping[str, object]

STAC_VERSION = "1.1.0"
PROJECTION_EXTENSION = (
    "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
)
MEDIA_TYPES = {
    ".tif": "image/tiff; application=geotiff",
    ".tiff": "image/tiff; application=geotiff",
    ".zarr": "application/vnd+zarr",
    ".nc": "application/x-netcdf",
    ".nc4": "application/x-netcdf",
    ".cdf": "application/x-netcdf",
}
# Every column a GeoSave STAC table owns, in the order a row states them.
ITEM_COLUMNS = (
    "id",
    "type",
    "stac_version",
    "stac_extensions",
    "links",
    "datetime",
    "start_datetime",
    "end_datetime",
    "proj:code",
    "proj:wkt2",
    "proj:shape",
    "proj:transform",
    "assets",
    "sources",
    "bbox",
    "geometry",
)


def item(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
    *,
    assets: Mapping[str, Asset],
    id: str | None = None,
    datetime: dt.datetime | None = None,
) -> dict[str, object]:
    """Describe one geolocated xarray object as a STAC item row.

    Args:
        data: Geolocated array, raster, or single-grid stack.
        assets: Where each raster is stored, by layer name. A value is a path
            or URL, or a STAC asset mapping stating `href`. A key names a
            layer of a stack; a lone raster takes any key.
        id: Item identifier. None uses the anchor stem.
        datetime: Item instant. None leaves it null, which the timespan in
            `start_datetime` and `end_datetime` then stands in for.

    Returns:
        The row's columns in STAC order, without its geometry.

    Raises:
        ValueError: `data` has no shared locatable grid, an asset names no
            layer or states no href, or the item has no time.

    Examples:
        >>> item(scene, assets={"image": "scene.tif"})["proj:code"]
        'EPSG:32633'
    """
    anchor = data.gs.anchor
    span = anchor.timespan
    if datetime is None and span is None:
        raise ValueError(
            "this raster carries no time and a STAC item needs one; pass datetime="
        )

    epsg = anchor.crs.epsg
    # STAC names a grid CRS by authority code, falling back to its WKT.
    grid: dict[str, object] = (
        {"proj:code": f"EPSG:{epsg}"}
        if epsg is not None
        else {"proj:wkt2": anchor.crs.to_wkt()}
    )
    return {
        "id": anchor.stem if id is None else id,
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [PROJECTION_EXTENSION],
        "links": [],
        "datetime": datetime,
        "start_datetime": None if span is None else span[0],
        "end_datetime": None if span is None else span[1],
        **grid,
        "proj:shape": (anchor.geobox.height, anchor.geobox.width),
        "proj:transform": tuple(anchor.geobox.transform)[:6],
        "assets": _assets(data, assets),
        "sources": sources(data),
    }


def sources(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
) -> list[dict[str, object]] | None:
    """List the provider items a raster or stack was loaded from.

    Args:
        data: Raster, band, or stack carrying `StacMetadata` in its attrs.

    Returns:
        One entry per provider item across every layer, each once, holding
        its captured properties, its id, and its datetime as ISO text. None
        where no layer was loaded from a catalog.

    Examples:
        >>> [entry["id"] for entry in sources(sample)]
        ['S2A_T49MHM_20250603', 'S2A_T49MHM_20250608']
    """
    rasters = list(data.gs.rasters.values()) if isinstance(data, xr.DataTree) else [data]
    carried = [
        model
        for model in (raster.gs.attrs.root.get(StacMetadata) for raster in rasters)
        if model is not None
    ]
    if not carried:
        return None
    merged, _ = StacMetadata.merge(carried)
    return [
        {**entry.properties, "id": entry.id, "datetime": entry.datetime.isoformat()}
        for entry in merged.stac_items or ()
    ]


def _assets(
    data: xr.DataArray | xr.Dataset | xr.DataTree, assets: Mapping[str, Asset]
) -> dict[str, dict[str, object]]:
    """Describe each stored raster as a STAC asset."""
    if isinstance(data, xr.DataTree):
        layers = data.gs.rasters
        unknown = [name for name in assets if name not in layers]
        if unknown:
            raise ValueError(
                f"assets {unknown} name no layer of this stack; its layers are "
                f"{list(layers)}"
            )
        bands = {name: layers[name].gs.variables for name in assets}
    else:
        bands = {name: data.gs.variables for name in assets}

    described: dict[str, dict[str, object]] = {}
    for name, asset in assets.items():
        fields = dict(asset) if isinstance(asset, Mapping) else {"href": asset}
        if "href" not in fields:
            raise ValueError(f"asset {name!r} states no href")
        href = fields["href"]
        if isinstance(href, PathLike):
            href = os.fspath(href)
        defaults: dict[str, object] = {"href": href}
        media = MEDIA_TYPES.get(PurePosixPath(str(href)).suffix.lower())
        if media is not None:
            defaults["type"] = media
        defaults["roles"] = ["data"]
        defaults["bands"] = [{"name": band} for band in bands[name]]
        # Fields the caller states win over the ones filled in here.
        described[name] = {**defaults, **fields, "href": href}
    return described


def check_table(frame: gpd.GeoDataFrame) -> None:
    """Check that a table of items can be written as a STAC table.

    Args:
        frame: Table carrying `assets`.

    Raises:
        ValueError: `id` is absent, null, or repeated, or the table is not in
            longitude/latitude.
    """
    if "id" not in frame:
        raise ValueError("a STAC table needs an 'id' column naming each item")
    ids = cast("pd.Series", frame["id"])
    if ids.isna().any():
        raise ValueError("STAC item ids must not be null")
    duplicates = sorted(set(ids[ids.duplicated()]))
    if duplicates:
        raise ValueError(f"STAC item ids must be unique, got {duplicates}")
    if frame.crs != "EPSG:4326":
        raise ValueError(
            f"a STAC table states footprints in EPSG:4326, got {frame.crs}; "
            "call to_crs('EPSG:4326') first"
        )
```

- [ ] **Step 4: Build rows from files in `core/vector.py`**

Delete `AnchorField`, `Asset`, `_STAC_VERSION`, `_PROJECTION_EXTENSION`, `_MEDIA_TYPES`, `from_anchor`, `from_xarray`, and the module-level `_assets`. Remove the imports they alone used. Add under `if TYPE_CHECKING:` `from geosave_engine.geodata.stac.item import Asset`. Keep `_column`.

Replace `from_geometry` so it takes `properties: Mapping[str, object] | None = None` in place of `**properties`, with `properties = properties or {}` as its first statement and "properties: Scalar columns stored beside the geometry." in its Args.

Add `from_assets`:

```python
    @classmethod
    def from_assets(
        cls,
        assets: Mapping[str, Asset] | str | PathLike[str],
        *,
        id: str | None = None,
        datetime: dt.datetime | None = None,
        geometry: SomeGeometry | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> GeoVector:
        """Register stored rasters as one STAC item, read from the files.

        The row is read off what is on disk, so it cannot describe pixels that
        were never written. No pixel is read.

        Args:
            assets: Where each raster is stored, by layer name. A value is a
                path or URL, or a STAC asset mapping stating `href`. One bare
                path registers a lone raster under its file stem.
            id: Item identifier. None uses the anchor stem.
            datetime: Item instant. None leaves it null, which the timespan in
                `start_datetime` and `end_datetime` then stands in for.
            geometry: Semantic geometry to retain. None uses the grid extent.
            properties: Caller-owned scalar columns.

        Returns:
            One-row vector in longitude/latitude describing the rasters, its
            `sources` listing the provider items they were loaded from.

        Raises:
            FileNotFoundError: An asset names no stored raster.
            ValueError: The rasters share no grid, an asset states no href,
                the item has no time, or a property takes the name of an item
                column.

        Examples:
            >>> record = GeoVector.from_assets(
            ...     {
            ...         "label": "samples/s1/label.tif",
            ...         "sentinel_2_l2a": "samples/s1/sentinel_2_l2a.tif",
            ...     },
            ...     properties={"land_cover": "forest"},
            ... )
            >>> record.gdf.loc[0, "id"]
            '2.7415W_5.6550N_5.1kmx5.1km_20181226_10m'
        """
        from geosave_engine.geodata.stac.item import ITEM_COLUMNS, item
        from geosave_engine.geodata.utils import io

        from .stack import stack

        if isinstance(assets, (str, PathLike)):
            assets = {PurePosixPath(str(assets)).stem: assets}
        properties = properties or {}
        collisions = sorted(set(properties) & set(ITEM_COLUMNS))
        if collisions:
            raise ValueError(f"properties collide with item columns {collisions}")

        hrefs: dict[str, object] = {}
        for name, asset in assets.items():
            if isinstance(asset, Mapping) and "href" not in asset:
                raise ValueError(f"asset {name!r} states no href")
            hrefs[name] = asset["href"] if isinstance(asset, Mapping) else asset

        with stack(
            {
                name: io.read_raster(cast("str", href), chunks="auto")
                for name, href in hrefs.items()
            }
        ) as stored:
            row = {
                **item(stored, assets=assets, id=id, datetime=datetime),
                **properties,
            }
            footprint = stored.gs.anchor.geobox.extent if geometry is None else geometry

        vector = cls.from_geometry(footprint)
        frame = (
            vector.gdf[[]]
            .assign(**{name: _column(name, value) for name, value in row.items()})
            .set_geometry(vector.gdf.geometry)
        )
        return cls(gpd.GeoDataFrame(frame, crs=vector.gdf.crs)).to_crs("EPSG:4326")
```

Keep `PurePosixPath` and `PathLike` imported. `stored.gs.anchor` raises the existing "publishes no grid at its root" `ValueError` where the layers share none, which is the refusal the grid test expects.

- [ ] **Step 5: Use it from the writer and the manifest task**

In `src/geosave_engine/geodata/utils/io/geoparquet.py`, inside `stored_assets`, replace the block from `if "id" not in frame:` through the `EPSG:4326` `raise` with:

```python
    from geosave_engine.geodata.stac.item import check_table

    check_table(frame)
```

and remove the `cast` and `pd` imports if nothing else in the file uses them.

In `src/geosave_engine/workflow/tasks/manifest.py`, delete `_ITEM_COLUMNS`, import `from geosave_engine.geodata.stac.item import ITEM_COLUMNS`, use it in the reserved-name check, and replace the body of `write_manifest` after its docstring with:

```python
    properties = properties or {}
    items = [
        GeoVector.from_assets(
            sample_assets(Path(path).resolve()),
            id=sample_id,
            properties=properties.get(sample_id, {}),
        )
        for sample_id, path in sample_paths.items()
    ]

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    GeoVector.concat(items).to_geoparquet(destination, overwrite=True)
    return str(destination)
```

Remove the imports this leaves unused (`cast`, `gpd`, `open_sample`). The workflow adapts to the library here, not the reverse: `from_assets` places caller columns after the item columns and before `geometry`, and `GeoVector.concat` unions them, leaving null where a sample lacks one. That is the order and nullness the manifest tests assert. Where a manifest test lists the item columns itself, add `sources` to its list.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/geodata tests/workflow tests/ml -q`
Expected: all pass.

Run: `uv run python -c "import geosave_engine.geodata.utils.io.geoparquet, geosave_engine.geodata.core.vector, geosave_engine.geodata.stac.item"`
Expected: no output, no import cycle.

Run: `grep -rn "from_anchor\|AnchorField\|_ITEM_COLUMNS\|_MEDIA_TYPES\|from_xarray" src tests`
Expected: no output.

Check a row with provider sources still converts to a valid STAC item: write one with `from_assets` on a raster carrying `StacMetadata`, then run `pystac.Item.from_dict(next(iter(stac_geoparquet.arrow.stac_table_to_items(pq.read_table(path))))).validate()`.
Expected: the two schema URLs, no error.

Run: `uv run basedpyright src/geosave_engine/geodata src/geosave_engine/workflow` and `uv run ruff check src tests`
Expected: `0 errors`, no findings.

---

### Task 1b: An object knows the file it was read from

**Files:**
- Modify: `src/geosave_engine/geodata/utils/io/gdal.py` (`read`), `src/geosave_engine/geodata/utils/io/layout.py` (`read_tree`, `write_tree`), `src/geosave_engine/geodata/utils/io/__init__.py` (`read_raster`)
- Modify: `src/geosave_engine/geodata/core/stack.py` (`GeoStack.to_cog`)
- Modify: `src/geosave_engine/geodata/transform/nodata.py` (`mask`, `to_nan`), `packing.py` (`unpack`), `vector.py` (`crop`)
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Test: `tests/geodata/utils/io/test_layout.py`, `tests/geodata/utils/io/test_geotiff.py`, `tests/geodata/core/test_vector.py`

**Interfaces:**
- Consumes: `GeoVector.from_assets` from Task 1.
- Produces:
  - Every Dataset `read_raster` returns carries `encoding["source"]`: the file for a GeoTIFF, the root directory for a tree. Zarr and NetCDF already do.
  - `read_raster(directory)` opens a tree through `read_tree`.
  - `write_tree(...) -> Path` and `GeoStack.to_cog(...) -> Path` return the directory they wrote.
  - `mask`, `to_nan`, `unpack`, and `crop` return objects without `encoding["source"]`.
  - `GeoVector.from_xarray(data: xr.Dataset | xr.DataTree, *, id=None, datetime=None, geometry=None, properties=None)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/core/test_vector.py`:

```python
def test_a_read_raster_registers_itself(tmp_path: Path) -> None:
    from geosave_engine.geodata import read_raster

    path = _written(tmp_path, build_raster(times=1), "scene.tif")

    found = GeoVector.from_xarray(read_raster(path, chunks="auto"))

    assert found.gdf.loc[0, "assets"]["scene"]["href"] == str(path)
    assert found.gdf.loc[0, "id"] == GeoVector.from_assets(path).gdf.loc[0, "id"]


def test_a_stack_registers_one_asset_per_group(tmp_path: Path) -> None:
    from geosave_engine.geodata import read_raster
    from geosave_engine.geodata.core.stack import stack as build_stack

    raster = build_raster(times=1)
    label = _written(tmp_path, raster[["nir"]], "label.tif")
    optical = _written(tmp_path, raster[["red"]], "optical.tif")
    sample = build_stack(
        {"label": read_raster(label), "sentinel_2_l2a": read_raster(optical)}
    )

    row = GeoVector.from_xarray(sample).gdf.iloc[0]

    assert list(row.assets) == ["label", "sentinel_2_l2a"]
    assert row.assets["sentinel_2_l2a"]["href"] == str(optical)


def test_an_object_never_written_must_be_saved_first() -> None:
    with pytest.raises(ValueError, match="from_assets"):
        GeoVector.from_xarray(build_raster(times=1))


@pytest.mark.parametrize("change", ["mask", "to_nan", "unpack", "crop"])
def test_an_object_geosave_changed_must_be_saved_first(
    tmp_path: Path, change: str
) -> None:
    from geosave_engine.geodata import read_raster

    stored = build_raster(times=1, packed=change == "unpack")
    if change != "unpack":
        stored = stored.gs.write_nodata(0)
    read = read_raster(_written(tmp_path, stored, "scene.zarr"), chunks="auto")
    if change == "mask":
        changed = read.gs.mask(np.array([[True, False], [True, True]]))
    elif change == "to_nan":
        changed = read.gs.to_nan()
    elif change == "unpack":
        changed = read.gs.to_nan().gs.unpack()
    else:
        left, bottom, right, top = read.gs.bounds.bbox
        half = gpd.GeoDataFrame(
            geometry=[box(left, bottom, (left + right) / 2, top)], crs=read.gs.crs
        )
        changed = read.gs.crop(GeoVector(half), mask=False)

    with pytest.raises(ValueError, match="from_assets"):
        GeoVector.from_xarray(changed)
```

Add to `tests/geodata/utils/io/test_layout.py`, using that file's existing cube helper in place of `cube` where it has one:

```python
def test_a_tree_is_written_read_and_registered_by_its_directory(tmp_path) -> None:
    from geosave_engine.geodata import GeoVector, read_raster, write_tree
    from tests.geodata.conftest import build_raster

    cube = build_raster(times=2)

    root = write_tree(cube, tmp_path / "optical", split_bands=True)
    read = read_raster(root, chunks="auto")
    row = GeoVector.from_xarray(read).gdf.iloc[0]

    assert root == tmp_path / "optical"
    assert set(read.data_vars) == {"red", "nir"}
    assert read.sizes["time"] == 2
    assert read.encoding["source"] == str(root)
    assert row.assets["optical"]["href"] == str(root)
```

Add to `tests/geodata/utils/io/test_geotiff.py`:

```python
def test_a_read_geotiff_names_the_file_it_came_from(tmp_path) -> None:
    from geosave_engine.geodata import read_raster
    from geosave_engine.geodata.utils.io import geotiff
    from tests.geodata.conftest import build_raster

    path = geotiff.write_cog(build_raster(), tmp_path / "scene.tif")

    assert read_raster(path).encoding["source"] == str(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_vector.py tests/geodata/utils/io/test_layout.py tests/geodata/utils/io/test_geotiff.py -q`
Expected: failures on `from_xarray` not existing, `KeyError: 'source'`, `write_tree` returning `None`, and `read_raster` refusing a directory.

- [ ] **Step 3: Record and forget the origin**

`utils/io/gdal.py`, at the end of `read`, before returning the Dataset: `dataset.encoding["source"] = str(source)`, using the name the function already gives its result.

`utils/io/layout.py`: `read_tree` sets `encoding["source"] = str(source)` on the Dataset it returns, replacing whatever a leaf left there; `write_tree` returns `Path(...)` of the root it wrote, and its signature and Returns say so.

`utils/io/__init__.py`, in `read_raster` before the suffix dispatch:

```python
    # A tree of COG leaves has no suffix of its own; its directory names it.
    if not suffix and Path(str(source)).is_dir():
        return layout.read_tree(source, **options)
```

with `Path` and `layout` imported, and "a directory written by `write_tree`" added to the Args text.

`core/stack.py`: `GeoStack.to_cog` returns the `Path` of the directory it wrote, annotated `-> Path`.

In `nodata.mask` (Dataset branch), `nodata.to_nan` (Dataset branch), `packing.unpack` (Dataset branch), and `vector.crop` (both returns of the non-stack path), give the result a fresh encoding without the origin before returning it:

```python
    # These pixels are no longer the file's, so the result names no file.
    result.encoding = {
        key: value for key, value in result.encoding.items() if key != "source"
    }
```

Assign a new dict as shown; popping from the existing one would also change the object the result was derived from. A stack goes through these same branches group by group.

- [ ] **Step 4: Add `from_xarray`**

```python
    @classmethod
    def from_xarray(
        cls,
        data: xr.Dataset | xr.DataTree,
        *,
        id: str | None = None,
        datetime: dt.datetime | None = None,
        geometry: SomeGeometry | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> GeoVector:
        """Register an object by the file it was read from.

        The path is found on the object and the row is read from that file,
        as `from_assets` reads it. An object GeoSave changed since it was read
        names no file and must be written first. Plain xarray arithmetic keeps
        the path, so the row then describes the file, not the changed values.

        Args:
            data: Raster read from a file, or a stack whose groups each were;
                a group's name becomes its asset key.
            id: Item identifier. None uses the anchor stem.
            datetime: Item instant, as `from_assets` takes it.
            geometry: Semantic geometry to retain. None uses the grid extent.
            properties: Caller-owned scalar columns.

        Returns:
            One-row vector in longitude/latitude describing the stored rasters.

        Raises:
            ValueError: `data`, or a group of it, was not read from a file.

        Examples:
            >>> GeoVector.from_xarray(read_raster("samples/s1/scene.tif"))
        """
        rasters = (
            data.gs.rasters if isinstance(data, xr.DataTree) else {"": data}
        )
        origins = {name: raster.encoding.get("source") for name, raster in rasters.items()}
        unsaved = [name for name, origin in origins.items() if origin is None]
        if unsaved:
            named = f"groups {unsaved}" if isinstance(data, xr.DataTree) else "this raster"
            raise ValueError(
                f"{named} were not read from a file, so no path names them; "
                f"write them, then register the paths with GeoVector.from_assets"
            )
        assets = origins if isinstance(data, xr.DataTree) else origins[""]
        return cls.from_assets(
            cast("Mapping[str, Asset] | str", assets),
            id=id,
            datetime=datetime,
            geometry=geometry,
            properties=properties,
        )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/geodata tests/workflow tests/ml -q`
Expected: all pass.

Run: `uv run basedpyright src/geosave_engine/geodata` and `uv run ruff check src tests`
Expected: `0 errors`, no findings.

---

### Task 2: `GeoVector` becomes the `gs` accessor

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `src/geosave_engine/geodata/__init__.py`, `src/geosave_engine/__init__.py`
- Modify: `src/geosave_engine/geodata/utils/io/__init__.py`
- Modify: `src/geosave_engine/geodata/transform/vector.py`
- Modify: `src/geosave_engine/geodata/core/anchor.py:264-267`, `src/geosave_engine/geodata/core/base.py` (`crop`)
- Modify: `src/geosave_engine/workflow/tasks/manifest.py`, `src/geosave_engine/workflow/configs/anchor.py:53`
- Modify: `src/geosave_engine/ml/segmentation/supervised/data.py`
- Modify tests: `tests/geodata/core/test_vector.py`, `test_base.py`, `test_stack.py`, `tests/geodata/transform/test_vector.py`, `test_nodata.py`, `tests/geodata/utils/io/test_geoparquet.py`, `tests/ml/segmentation/supervised/test_data.py`, `tests/workflow/tasks/test_manifest.py`, `tests/workflow/flows/test_prepare_dense_data.py`

**Interfaces:**
- Consumes: `item`, `ITEM_COLUMNS`, `check_table` from Task 1.
- Produces:
  - `GeoVector` registered as the `gs` accessor of `gpd.GeoDataFrame`, holding the frame in `self._data`.
  - `check_geometries(frame: gpd.GeoDataFrame) -> None` in `core/vector.py`.
  - `GeoDataFrame` stand-in exported from `geosave_engine.geodata` and `geosave_engine`.
  - `read_vector(...) -> GeoDataFrame`; `transform.vector.vectorize(...) -> GeoDataFrame`; `transform.vector.rasterize(vector: gpd.GeoDataFrame, like, ...)`; `transform.vector.crop(data, vector: gpd.GeoDataFrame, ...)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/core/test_vector.py`:

```python
def test_the_accessor_reads_only_a_located_geodataframe() -> None:
    import pandas as pd

    with pytest.raises(AttributeError, match="GeoDataFrame"):
        _ = pd.DataFrame({"a": [1]}).gs
    with pytest.raises(AttributeError, match="CRS"):
        _ = gpd.GeoDataFrame(geometry=[Point(0, 0)]).gs


def test_the_accessor_survives_native_selection_and_concatenation() -> None:
    import pandas as pd

    frame = gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )

    assert frame[frame.name == "b"].gs.crs.to_epsg() == 4326
    assert len(pd.concat([frame, frame]).gs.query(box(-1, -1, 2, 2))) == 4


def test_a_renamed_geometry_column_still_answers() -> None:
    frame = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:4326"
    ).rename_geometry("footprint")

    assert frame.gs.footprint.geom.equals(box(0, 0, 2, 2))
    assert list(frame.gs.query(box(1, 1, 3, 3)).name) == ["a"]
    assert GeoVector.concat([frame, frame]).geometry.name == "geometry"


@pytest.mark.parametrize("problem", ["null", "empty", "invalid"])
def test_writing_refuses_geometries_that_name_no_ground(
    problem: str, tmp_path: Path
) -> None:
    from shapely.geometry import Polygon

    bad = {
        "null": None,
        "empty": Polygon(),
        "invalid": Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]),
    }[problem]
    frame = gpd.GeoDataFrame(geometry=[bad], crs="EPSG:4326")

    with pytest.raises(ValueError, match=problem):
        frame.gs.to_geoparquet(tmp_path / "bad.parquet")


def test_rasterize_refuses_an_invalid_geometry(raster) -> None:
    from shapely.geometry import Polygon

    bowtie = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], crs=raster.gs.crs
    )

    with pytest.raises(ValueError, match="invalid"):
        bowtie.gs.rasterize(raster)
```

Then migrate every file in the Files list with the migration table. In `tests/geodata/core/test_vector.py` also:

- `test_empty_vector_has_a_crs_and_no_footprint`: `vector.gs.crs`, `vector.gs.footprint`.
- `test_concat_unions_columns_resets_index_and_keeps_duplicates`: `renamed` is `GeoVector.from_geometry(Point(1, 1), properties={"score": 4}).rename_geometry("footprint")`.
- Delete `test_vector_method_delegates_to_vector_transform` and `test_raster_method_delegates_to_vector_transform` only if they compare a `GeoVector` method against the transform function by identity of a wrapper; if they compare results, migrate them.
- `_record` returns `gpd.GeoDataFrame`; `_href(vector)` reads `vector.iloc[0].assets["prediction"]["href"]`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_vector.py -q`
Expected: failures with `AttributeError: 'GeoDataFrame' object has no attribute 'gs'`.

- [ ] **Step 3: Rewrite `core/vector.py`**

Replace the class statement, its docstring, `__post_init__`, `__len__`, `crs`, `footprint`, and `to_crs` with the following, and delete `from dataclasses import dataclass`. Add `import pandas as pd` if absent (it is already imported).

```python
def check_geometries(frame: gpd.GeoDataFrame) -> None:
    """Check that every geometry names some ground.

    Args:
        frame: Frame about to be burned or written.

    Raises:
        ValueError: A geometry is null, empty, or invalid.
    """
    if frame.geometry.isna().any():
        raise ValueError("a geometry is null; drop the row or give it a geometry")
    if frame.geometry.is_empty.any():
        raise ValueError("a geometry is empty; drop the row or give it a geometry")
    if not frame.geometry.is_valid.all():
        raise ValueError("a geometry is invalid; repair it with geometry.make_valid()")


@pd.api.extensions.register_dataframe_accessor("gs")
class GeoVector:
    """GeoSave operations on a GeoDataFrame, read through `gs`.

    The frame stays a native GeoDataFrame, so every GeoPandas operation still
    applies to it. Containment against a raster geobox belongs to the
    operation combining vector and raster data.

    Args:
        data: GeoDataFrame with an active geometry column and a CRS.

    Raises:
        AttributeError: `data` is not a GeoDataFrame, or has no active
            geometry column or no CRS.

    Examples:
        >>> plots = read_vector("plots.geojson")
        >>> plots.gs.crs.to_epsg()
        4326
        >>> record = GeoVector.from_assets({"prediction": "rasters/prediction.zarr"})
        >>> catalog = GeoVector.concat([catalog, record])
        >>> matches = labels.gs.query(prediction)
        >>> catalog.gs.to_geoparquet("dataset/catalog.parquet")
        PosixPath('dataset/catalog.parquet')
    """

    def __init__(self, data: gpd.GeoDataFrame) -> None:
        """Bind the accessor to one GeoDataFrame."""
        # pandas reads AttributeError as "this object has no such accessor".
        if not isinstance(data, gpd.GeoDataFrame):
            raise AttributeError(
                f"gs reads a GeoDataFrame, got {type(data).__name__}; build one "
                f"with gpd.GeoDataFrame(...)"
            )
        if data.active_geometry_name is None:
            raise AttributeError("gs needs an active geometry column")
        if data.crs is None:
            raise AttributeError("gs needs a CRS; call set_crs(...) first")
        self._data = data

    @property
    def crs(self) -> OdcCRS:
        """Return the coordinate reference system every geometry shares."""
        return OdcCRS(self._data.crs)

    @property
    def footprint(self) -> Geometry:
        """Return the union of all geometries in the vector CRS.

        Returns:
            ODC geometry carrying the vector CRS.

        Raises:
            ValueError: The frame holds no row.
        """
        if self._data.empty:
            raise ValueError("an empty GeoVector has no footprint")
        return Geometry(self._data.geometry.union_all(), crs=self.crs)
```

For the rest of the class, apply these exact changes and nothing else:

- `to_geojson`, `to_geopackage`, `to_geoparquet`: insert `check_geometries(self._data)` as the first statement of the body, and pass `self._data` where `self.gdf` was passed. Their examples read `plots.gs.to_geojson(...)`, `plots.gs.to_geopackage(...)`, `plots.gs.to_geoparquet(...)`, `catalog.gs.to_geoparquet(...)`.
- `concat(cls, vectors: Iterable[gpd.GeoDataFrame]) -> GeoDataFrame`: `item.crs` becomes `OdcCRS(item.crs)`, `item.gdf.copy()` becomes `item.copy()`, and it ends with

```python
        frame = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs
        )
        check_geometries(frame)
        return cast("GeoDataFrame", frame)
```

- `upsert(self, records: gpd.GeoDataFrame, *, on: str) -> GeoDataFrame`: `self.gdf` becomes `self._data`, `records.gdf` becomes `records`, `records.crs` becomes `records.gs.crs`, and it ends with

```python
        retained = self._data.loc[~self._data[on].isin(incoming)]
        return type(self).concat([retained, records])
```

  Add to its docstring Returns: "Replaced rows move to the end, so row order is not preserved."
- `query(...) -> GeoDataFrame`: the target union gains `gpd.GeoDataFrame` in place of `GeoVector`; `isinstance(target, gpd.GeoDataFrame)` reads `target.gs.footprint`; the last branch reads `type(self).from_geometry(target).gs.footprint`; `self.gdf` becomes `self._data`; it returns `cast("GeoDataFrame", ...)` of the copied frames.
- `vectorize(cls, ...) -> GeoDataFrame`: unchanged body.
- `rasterize(self, like, ...)`: passes `self._data` as the first argument.
- `empty(cls, crs) -> GeoDataFrame`: returns `cast("GeoDataFrame", gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs)))`.
- `from_geometry(...) -> GeoDataFrame`: builds the frame, calls `check_geometries(frame)`, returns `cast("GeoDataFrame", frame)`.
- `from_xarray(...) -> GeoDataFrame`: only the annotation changes; it still returns what `from_assets` returns. Its test for a GeoSave-changed object passes `half` to `crop` directly, not `GeoVector(half)`.
- `from_assets(...) -> GeoDataFrame`: `vector.gdf` becomes `vector`, its example's last lines read `>>> record.loc[0, "id"]`, and it ends with

```python
        return cast(
            "GeoDataFrame", gpd.GeoDataFrame(frame, crs=vector.crs).to_crs("EPSG:4326")
        )
```

  Its example's last lines read `>>> record.loc[0, "id"]`.

Under `if TYPE_CHECKING:` add `from geosave_engine.geodata import GeoDataFrame`.

- [ ] **Step 4: Declare and export the stand-in**

In `src/geosave_engine/geodata/__init__.py` add `import geopandas as gpd`, add inside `if TYPE_CHECKING:`

```python
    class GeoDataFrame(gpd.GeoDataFrame):
        """GeoPandas GeoDataFrame carrying GeoSave's `gs` accessor.

        The runtime value is a `geopandas.GeoDataFrame`; this declaration
        exists only so type checkers can resolve `.gs`.
        """

        gs: GeoVector
```

add `GeoDataFrame = gpd.GeoDataFrame` to the `else:` branch, and add `"GeoDataFrame"` to `__all__`. In `src/geosave_engine/__init__.py` add `GeoDataFrame` to the import list and to `__all__`.

- [ ] **Step 5: Migrate the library callers**

`src/geosave_engine/geodata/utils/io/__init__.py`: remove the module-level `GeoVector` import; under `if TYPE_CHECKING:` import `GeoDataFrame` from `geosave_engine.geodata`; `read_vector` returns `GeoDataFrame`, each branch returning `cast("GeoDataFrame", frame)`. Its docstring Returns reads "GeoDataFrame holding the file's features in one CRS, read through `gs`." and its examples read `read_vector("plantations.geojson").gs.crs.to_epsg()` and `catalog.gs.query(prediction).iloc[0]["assets"]["prediction"]["href"]`.

`src/geosave_engine/geodata/transform/vector.py`:

- The `TYPE_CHECKING` import becomes `from geosave_engine.geodata import GeoDataFrame`, and `import geopandas as gpd` is already present.
- `vectorize(...) -> GeoDataFrame`: delete the local `GeoVector` import; both returns become `cast("GeoDataFrame", gpd.GeoDataFrame(...))`.
- `rasterize(vector: gpd.GeoDataFrame, like, ...)`: after the geobox check add

```python
    from geosave_engine.geodata.core.vector import check_geometries

    check_geometries(vector)
    frame = vector.to_crs(geobox.crs)
```

  replacing `frame = vector.to_crs(geobox.crs).gdf`. The `KeyError` text reads `f"the vector has no {column!r} column"`.
- `crop(data, vector: gpd.GeoDataFrame, ...)`: `vector.crs` becomes `vector.gs.crs` and `vector.footprint` becomes `vector.gs.footprint`. The CRS mismatch still raises in this task.

`src/geosave_engine/geodata/core/anchor.py`: `footprint = GeoVector.from_geometry(geometry).gs.footprint`.

`src/geosave_engine/geodata/core/base.py`: the `TYPE_CHECKING` import of `GeoVector` becomes `import geopandas as gpd`, and `crop` takes `vector: gpd.GeoDataFrame`.

`src/geosave_engine/workflow/configs/anchor.py`: `io.read_vector(self.path).gs.footprint`.

`src/geosave_engine/workflow/tasks/manifest.py`: `GeoVector.concat(items).gs.to_geoparquet(destination, overwrite=True)`.

`src/geosave_engine/ml/segmentation/supervised/data.py`: `manifest: gpd.GeoDataFrame` in `Dataset.__init__`, with `import geopandas as gpd` under `TYPE_CHECKING`; `manifest.gdf["id"]` and `manifest.gdf["assets"]` become `manifest["id"]` and `manifest["assets"]`; remove `GeoVector` from the import if nothing else uses it.

- [ ] **Step 6: Run the tests and checks**

Run: `uv run pytest -q`
Expected: all pass.

Run: `grep -rn "\.gdf\b\|GeoVector(" src tests docs/guides`
Expected: only `docs/guides/workflows.md`, which Task 6 updates.

Run: `uv run basedpyright src/geosave_engine` and `uv run ruff check src tests`
Expected: no new errors against the count before this task; no findings.

Verify autocomplete with a scratch file outside the repository:

```python
from geosave_engine.geodata import GeoVector, read_vector

frame = read_vector("a.geojson")
reveal_type(frame.gs)
reveal_type(frame.gs.query(frame.gs.footprint).gs.footprint)
reveal_type(GeoVector.from_geometry("POINT (0 0)").gs)
```

Run: `uv run basedpyright <scratch file>`
Expected: `GeoVector`, `Geometry`, `GeoVector`.

---

### Task 3: `to_xarray`

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py`
- Modify: `src/geosave_engine/ml/segmentation/supervised/data.py`
- Test: `tests/geodata/core/test_vector.py`

**Interfaces:**
- Consumes: `io.read_raster(href, chunks="auto")`, `stack(rasters)`.
- Produces: `GeoVector.to_xarray(self, id: str | None = None, *, layers: Collection[str] | None = None) -> DataTree`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/core/test_vector.py`:

```python
def _sample_table(root: Path) -> gpd.GeoDataFrame:
    """Write two samples of two layers each and return their manifest."""
    from geosave_engine.geodata.utils.io import geotiff

    rows = []
    for name in ("a", "b"):
        raster = build_raster(times=1).squeeze("time", drop=False)
        folder = root / name
        folder.mkdir(parents=True)
        geotiff.write_cog(raster[["red"]], folder / "optical.tif")
        geotiff.write_cog(raster[["nir"]], folder / "label.tif")
        rows.append(
            GeoVector.from_assets(
                {"optical": folder / "optical.tif", "label": folder / "label.tif"},
                id=name,
            )
        )
    return GeoVector.concat(rows)


def test_a_row_opens_as_a_lazy_stack_of_its_assets(tmp_path: Path) -> None:
    table = _sample_table(tmp_path)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        sample = table.gs.to_xarray("b")

    assert sample.gs.groups == ("optical", "label")
    assert sample.gs.rasters["optical"].gs.variables == ("red",)
    assert started == []


def test_layers_names_the_assets_to_open(tmp_path: Path) -> None:
    table = _sample_table(tmp_path)

    assert table.gs.to_xarray("a", layers=["label"]).gs.groups == ("label",)
    with pytest.raises(KeyError, match="thermal"):
        table.gs.to_xarray("a", layers=["thermal"])


def test_a_one_row_table_needs_no_id(tmp_path: Path) -> None:
    table = _sample_table(tmp_path)

    assert table.iloc[[0]].gs.to_xarray().gs.groups == ("optical", "label")
    with pytest.raises(ValueError, match="id="):
        table.gs.to_xarray()
    with pytest.raises(KeyError, match="missing"):
        table.gs.to_xarray("missing")


def test_a_table_without_assets_names_no_rasters() -> None:
    plain = GeoVector.from_geometry(Point(0, 0))

    with pytest.raises(KeyError, match="assets"):
        plain.gs.to_xarray()


def test_a_moved_manifest_still_opens_its_samples(tmp_path: Path) -> None:
    source = tmp_path / "prepared"
    _sample_table(source).gs.to_geoparquet(source / "manifest.parquet")
    moved = tmp_path / "moved"
    shutil.move(source, moved)

    sample = read_vector(moved / "manifest.parquet").gs.to_xarray("a")

    assert sample.gs.groups == ("optical", "label")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_vector.py -q -k "opens or layers or one_row or without_assets or moved"`
Expected: FAIL with `AttributeError: 'GeoVector' object has no attribute 'to_xarray'`.

- [ ] **Step 3: Implement**

Add to `GeoVector`, after `upsert`. Add `Collection` to the `collections.abc` import and `DataTree` to the `TYPE_CHECKING` import from `geosave_engine.geodata`.

```python
    def to_xarray(
        self, id: str | None = None, *, layers: Collection[str] | None = None
    ) -> DataTree:
        """Open one row's assets as a lazy stack, one group per asset.

        Args:
            id: Row to open. None opens the only row of a one-row table.
            layers: Asset keys to open. None opens every one.

        Returns:
            Stack whose groups are the opened assets, in the order named.

        Raises:
            KeyError: The table carries no `assets`, no row carries `id`, or
                the row has no asset under a named layer.
            ValueError: `id` is None and the table holds several rows.

        Examples:
            >>> manifest.gs.to_xarray("s1").gs.groups
            ('label', 'sentinel_2_l2a')
        """
        from geosave_engine.geodata.utils import io

        from .stack import stack

        frame = self._data
        if "assets" not in frame:
            raise KeyError("this table carries no 'assets' column naming rasters")
        if id is None:
            if len(frame) != 1:
                raise ValueError(
                    f"the table holds {len(frame)} rows; name the one to open "
                    f"with id="
                )
            assets = frame["assets"].iloc[0]
        else:
            rows = frame.loc[frame["id"] == id, "assets"]
            if rows.empty:
                raise KeyError(f"no row carries the id {id!r}")
            assets = rows.iloc[0]

        names = list(assets) if layers is None else list(layers)
        missing = [name for name in names if name not in assets]
        if missing:
            raise KeyError(
                f"the row has no {missing} layer; its layers are {list(assets)}"
            )
        return stack(
            {name: io.read_raster(assets[name]["href"], chunks="auto") for name in names}
        )
```

- [ ] **Step 4: Use it in the training `Dataset`**

In `src/geosave_engine/ml/segmentation/supervised/data.py`, `Dataset.__init__` keeps its missing-layer check and error text, but stores the table and its ids instead of hrefs:

```python
        self._layers = (*spec.rasters, target)
        for sample_id, assets in zip(manifest["id"], manifest["assets"], strict=True):
            missing = [name for name in self._layers if name not in assets]
            if missing:
                raise ValueError(
                    f"sample {sample_id!r} has no {missing} layer; its layers are "
                    f"{list(assets)}"
                )
        self._manifest = manifest
```

and `_cut` opens each sample through the accessor:

```python
        for sample_id in self._manifest["id"]:
            sample = self._manifest.gs.to_xarray(sample_id, layers=self._layers)
```

replacing the `for hrefs in self._hrefs.values():` loop header and its `stack({...read_raster...})` expression. Remove `self._hrefs` and any import this leaves unused.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/geodata/core/test_vector.py tests/ml -q`
Expected: all pass.

Run: `uv run basedpyright src/geosave_engine/geodata/core src/geosave_engine/ml` and `uv run ruff check src tests`
Expected: no new errors; no findings.

---

### Task 4: `query(time=)`

**Files:**
- Modify: `src/geosave_engine/geodata/core/vector.py` (`query`)
- Test: `tests/geodata/core/test_vector.py`

**Interfaces:**
- Consumes: `parse_daterange(value: AnchorDatetime) -> DateRange` and `naive_utc` from `geosave_engine.geodata.utils.datetime`.
- Produces: `query(self, target, *, time: AnchorDatetime | None = None, predicate: SpatialPredicate = "intersects") -> GeoDataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
def _dated_table() -> gpd.GeoDataFrame:
    from datetime import timedelta

    rows = []
    for name, start in (("jan", datetime(2025, 1, 10, tzinfo=UTC)), ("jun", datetime(2025, 6, 10, tzinfo=UTC))):
        rows.append(
            gpd.GeoDataFrame(
                {
                    "id": [name],
                    "datetime": [None],
                    "start_datetime": [start],
                    "end_datetime": [start + timedelta(days=5)],
                },
                geometry=[box(0, 0, 1, 1)],
                crs="EPSG:4326",
            )
        )
    instant = gpd.GeoDataFrame(
        {
            "id": ["instant"],
            "datetime": [datetime(2025, 6, 12, tzinfo=UTC)],
            "start_datetime": [None],
            "end_datetime": [None],
        },
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    table = GeoVector.concat([*rows, instant])
    for column in ("datetime", "start_datetime", "end_datetime"):
        table[column] = pd.to_datetime(table[column], utc=True)
    return table


def test_query_keeps_rows_whose_span_overlaps_the_time() -> None:
    table = _dated_table()
    everywhere = box(-1, -1, 2, 2)

    assert list(table.gs.query(everywhere).id) == ["jan", "jun", "instant"]
    assert list(table.gs.query(everywhere, time="2025-01").id) == ["jan"]
    assert list(table.gs.query(everywhere, time=("2025-06-14", "2025-06-30")).id) == ["jun"]


def test_query_reads_a_lone_datetime_as_the_rows_time() -> None:
    table = _dated_table()

    matched = table.gs.query(box(-1, -1, 2, 2), time=("2025-06-12", "2025-06-12"))

    assert list(matched.id) == ["jun", "instant"]


def test_query_by_time_needs_the_time_columns() -> None:
    plain = GeoVector.from_geometry(Point(0, 0))

    with pytest.raises(KeyError, match="start_datetime"):
        plain.gs.query(Point(0, 0), time="2025-01")
```

Add `import pandas as pd` to the test module's imports, and reformat the long lines with `uv run ruff format tests/geodata/core/test_vector.py --diff`, applying only the hunks inside these new tests by hand.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_vector.py -q -k "time or lone_datetime"`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'time'`.

- [ ] **Step 3: Implement**

Add `time: AnchorDatetime | None = None` before `predicate` in `query`, import `AnchorDatetime` under `TYPE_CHECKING` and `parse_daterange` at module level beside `naive_utc`, add to the docstring

```text
            time: Timespan a row must overlap, as `parse_daterange` accepts
                it. A row is read by its `start_datetime` and `end_datetime`,
                or by its `datetime` where those are null. None applies no
                time filter.
```

and `KeyError: \`time\` is given and the table lacks the STAC time columns.` under Raises, and replace the final `return` with:

```python
        matched = candidates.loc[exact]
        if time is not None:
            columns = ("datetime", "start_datetime", "end_datetime")
            absent = [name for name in columns if name not in matched]
            if absent:
                raise KeyError(
                    f"the table has no {absent} columns, so its rows state no time"
                )
            start, end = (
                pd.Timestamp(naive_utc(edge), tz="UTC")
                for edge in parse_daterange(time)
            )
            first = matched["start_datetime"].fillna(matched["datetime"])
            last = matched["end_datetime"].fillna(matched["datetime"])
            matched = matched.loc[(first <= end) & (last >= start)]
        return cast("GeoDataFrame", matched.copy())
```

The early return for an empty frame keeps its place above; move the `time` column check above it so a table without the columns raises even when empty.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/core/test_vector.py -q`
Expected: all pass.

---

### Task 5: `crop` reprojects the vector

**Files:**
- Modify: `src/geosave_engine/geodata/transform/vector.py` (`crop`), `src/geosave_engine/geodata/core/base.py` (`crop` docstring)
- Test: `tests/geodata/transform/test_vector.py`

**Interfaces:**
- Consumes: Task 2's `crop(data, vector: gpd.GeoDataFrame, ...)`.
- Produces: same signature; a vector in another CRS is reprojected onto the raster's.

- [ ] **Step 1: Write the failing test**

```python
def test_crop_takes_a_vector_in_another_crs() -> None:
    triangle = _triangle()

    native = _band().gs.crop(triangle)
    geographic = _band().gs.crop(triangle.to_crs("EPSG:4326"))

    assert geographic.shape == native.shape
    assert geographic.values.tolist() == native.values.tolist()
```

If an existing test asserts that a CRS mismatch raises "reproject the vector before cropping", delete it; this task replaces that rule.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/geodata/transform/test_vector.py -q -k another_crs`
Expected: FAIL with `ValueError: vector is in ... but the raster is in ...`.

- [ ] **Step 3: Implement**

In `crop`, replace the `if vector.gs.crs != crs: raise ...` block with:

```python
    # A vector follows the raster's grid; the raster's pixels never move here.
    vector = vector.to_crs(crs)
```

In both `crop` docstrings (`transform/vector.py` and `core/base.py`), the `vector` argument reads "Geometries to cut against, reprojected onto `data`'s CRS where they sit in another." and Raises no longer lists a different CRS.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/transform tests/geodata/core -q`
Expected: all pass.

---

### Task 6: Example manifest and guide

**Files:**
- Modify: `examples/data/dw_imagery/manifest.parquet`
- Modify: `docs/guides/workflows.md:147-149`

**Interfaces:**
- Consumes: `write_manifest(sample_paths, output)` from Task 1.
- Produces: nothing other tasks rely on.

- [ ] **Step 1: Confirm the example is stale**

Run: `uv run python -c "from geosave_engine.geodata import read_vector; print(list(read_vector('examples/data/dw_imagery/manifest.parquet')))"`
Expected: columns `path`, `format`, `grid_crs`, with no `assets`.

- [ ] **Step 2: Regenerate it**

Run:

```bash
uv run python -c "
from pathlib import Path
from geosave_engine.workflow.tasks.manifest import write_manifest
root = Path('examples/data/dw_imagery')
samples = {p.name: str(p) for p in sorted(root.iterdir()) if p.is_dir()}
print(write_manifest(samples, root / 'manifest.parquet'))
"
```

Expected: prints `examples/data/dw_imagery/manifest.parquet`.

- [ ] **Step 3: Verify it**

Run:

```bash
uv run python -c "
import pyarrow.parquet as pq, pystac, stac_geoparquet
from geosave_engine.geodata import read_vector
path = 'examples/data/dw_imagery/manifest.parquet'
table = read_vector(path)
print(len(table), table.gs.to_xarray(table['id'].iloc[0]).gs.groups)
print(pq.read_table(path).column('assets')[0].as_py()['label']['href'])
items = list(stac_geoparquet.arrow.stac_table_to_items(pq.read_table(path)))
print(pystac.Item.from_dict(items[0]).validate())
"
```

Expected: `3 ('label', 'sentinel_2_l2a')`; a relative href such as `dw_.../label.tif`; and the two schema URLs for item 1.1.0 and projection v2.0.0.

- [ ] **Step 4: Update the guide**

In `docs/guides/workflows.md`, the line reading `row = read_vector("data/prepared/manifest.parquet").gdf.iloc[0]` becomes `row = read_vector("data/prepared/manifest.parquet").iloc[0]`. Read the surrounding example; where it then opens the row's rasters by hand, replace that with `sample = manifest.gs.to_xarray(sample_id)` and adjust the two or three lines around it so the example still runs top to bottom.

- [ ] **Step 5: Final verification**

Run: `uv run pytest -q`
Expected: all pass.

Run: `uv run basedpyright src/geosave_engine/geodata src/geosave_engine/workflow src/geosave_engine/ml`
Expected: no new errors against the count before Task 1.

Run: `uv run ruff check src tests` and `git diff --check`
Expected: no findings, no output.

Run: `grep -rn "\.gdf\b\|GeoVector(\|from_anchor" src tests docs/guides README.md`
Expected: no output.

- [ ] **Step 6: Report**

Report what changed, the checks run with their results, and breaking changes. Do not commit.

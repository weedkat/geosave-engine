# STAC Through PySTAC Classes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build STAC Assets, Items and Collections from saved rasters through PySTAC's extension classes, keep them as a stac-geoparquet table through stac-geoparquet, and search that table through rustac, with no hand-inserted keys.

**Architecture:** PySTAC builds the objects (`pystac.Item`, `item.add_asset`, `pystac.Collection`); GeoSave only describes them. `stac/extensions/` holds one module per schema, each writing its fields onto an asset from a saved raster through that schema's PySTAC class; `attrs/headers/stac.py` reads those classes back into attrs when loading from a catalog, and `stac/item.py` assembles Items from `gs` properties. `stac/table.py` converts Items to rows and back and writes one file; `StacTableClient` searches it. Two extensions PySTAC has no class for get our own class on PySTAC's base.

**Tech Stack:** pystac 1.14.3 (Projection v2.0.0, Raster v1.1.0, EO v1.1.0, Classification v2.0.0), stac-geoparquet 0.8.2, rustac 0.9.17, geopandas, xarray, odc-geo, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-stac-pystac-classes-design.md`

## Global Constraints

- No `extra_fields[...]` or `properties[...]` write outside an extension class. Reading foreign STAC 1.1 `bands` in `read_bands` is the one keyed read.
- Use a PySTAC class where one exists: `X.ext(obj, add_if_missing=True).apply(...)`.
- Our own extension classes subclass `PropertiesExtension` and `ExtensionManagementMixin[pystac.Item]`.
- Item <-> row conversion happens only in `table.from_items` and `table.to_items`.
- Facts come from `gs`: `gs.geobox`, `gs.timespan`, `gs.times`, `gs.anchor.format`, `gs.attrs`, `gs.variables`.
- `geodata.attrs` must not import `geodata.stac` (a test enforces it).
- Public functions carry concise Google-style docstrings; comments explain domain constraints only.
- Do not commit. The working tree holds unrelated uncommitted work; commit only when the user asks, and then only the files a task names.
- Do not touch `.ipynb` files or files a task does not name.
- Run tests with `uv run pytest`; lint with `uv run ruff check .`.
- The source modules in Tasks 2-4 (`headers/stac.py`, every extension module, `item.py`, `table.py`), the Task 2 test file and the Task 3 extension test files were run as prototypes on 2026-10-08. Blocks marked "not run" were not, and neither were the other test files of Tasks 1, 3 and 4; expect to adjust an assertion where a test and the tested code disagree, and report it.

## Review Focus

1. **Rows filtered to one collection.** The other collection's asset column is all null; `to_items` must still return Items holding only their own assets. Pinned in Task 4 (`test_filtered_rows_keep_only_the_assets_they_have`).
2. **A table rewritten without `collections=`.** The stored Collections are gone; a person expects either to be told or to keep them. Pinned in Task 4 (`test_a_rewrite_without_collections_stores_none`) and stated in `write`'s docstring.
3. **A foreign Item that publishes `raster:bands` without declaring the Raster schema.** `create_header` must still read it, without changing the Item. Pinned in Task 2 (`test_reader_uses_the_selected_band_without_mutating_the_item`; its fixture declares no schema).
4. **A legend whose class name carries `/` or a space.** Publishing it makes an invalid Item; a person expects a clear error naming the class. Pinned in Task 3 (`test_a_class_name_stac_refuses_raises`).
5. **A single-date COG.** Its Item must carry `datetime`, not a day-long range, because `gs.timespan` widens a date label to its day. Pinned in Task 1 and Task 3 (`test_one_file_per_scene_is_one_item_per_instant`).

---

### Task 1: `gs.times` reads a scalar time coordinate

A COG of one date opens with `time` as a scalar coordinate, and `gs.times` raises `TypeError` on it. Items need that label.

**Files:**
- Modify: `src/geosave_engine/geodata/core/base.py` (the `times` property, about line 265)
- Test: `tests/geodata/core/test_base.py`

**Interfaces:**
- Produces: `raster.gs.times -> pd.DatetimeIndex | None`, one label for a scalar `time` coordinate.

- [ ] **Step 1: Write the failing test.** Append to `tests/geodata/core/test_base.py`:

```python
def test_times_reads_a_scalar_time_coordinate() -> None:
    from tests.geodata.conftest import build_raster

    one_date = build_raster(times=2).isel(time=0)

    assert one_date.gs.times.strftime("%Y%m%d").tolist() == [
        build_raster(times=2).gs.times[0].strftime("%Y%m%d")
    ]
```

- [ ] **Step 2: Run it and see it fail.**

Run: `uv run pytest tests/geodata/core/test_base.py::test_times_reads_a_scalar_time_coordinate -q`
Expected: FAIL with `TypeError: DatetimeIndex(...) must be called with a collection of some kind`.

- [ ] **Step 3: Fix the property.** In `times`, replace the last line:

```python
        return pd.DatetimeIndex(np.atleast_1d(coords[TIME_COORDINATE].values))
```

- [ ] **Step 4: Run the file.**

Run: `uv run pytest tests/geodata/core/test_base.py -q`
Expected: all pass.

---

### Task 2: Read STAC bands into attrs through PySTAC classes

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/headers/stac.py` (whole file replaced)
- Modify: `tests/geodata/attrs/headers/test_stac.py` (whole file replaced)
- Modify: `tests/geodata/stac/test_dataflow.py` (two tests removed; Task 4 writes their successors)
- Modify: `tests/geodata/stac/test_source.py` (asset field names)

`attrs/extensions/` stays until Task 3, because `stac/asset.py` still imports it.

**Interfaces:**
- Consumes: `raster.gs.attrs.data_vars[name].get(Model)`, `raster.gs.variables`.
- Produces:
  - `create_header(items, collection, loaded, *, groupby, item_properties=(), asset_fields=None, stac_cfg=None) -> AttrsHeader` (unchanged signature)
  - `read_bands(asset) -> list[tuple[RasterBand, Band]]`
  - `band_attrs(raster: RasterBand, spectral: Band) -> FlatAttrs`
  - `read_asset_fields(asset, *, band_index=1) -> dict[str, object]`, band keys in Raster/EO v1.1.0 spelling (`scale`, `common_name`).

- [ ] **Step 1: Replace the test file** `tests/geodata/attrs/headers/test_stac.py` with:

```python
from __future__ import annotations

import subprocess
import sys
from datetime import datetime as dt, timezone

import pytest
import pystac
import xarray as xr
from pystac.extensions.eo import Band
from pystac.extensions.raster import RasterBand

from geosave_engine.geodata.attrs import (
    CFVariable,
    Nodata,
    Packing,
    Spectral,
    StacMetadata,
)
from geosave_engine.geodata.attrs.headers.stac import (
    band_attrs,
    create_header,
    read_asset_fields,
    read_bands,
)
from geosave_engine.geodata.warnings import DroppedAttrsWarning


def _item(id: str, timestamp: dt, fields: dict[str, object]) -> pystac.Item:
    item = pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=timestamp.replace(tzinfo=timezone.utc),
        properties={},
    )
    item.add_asset(
        "red",
        pystac.Asset(
            href=f"https://example.com/{id}.tif",
            media_type=pystac.MediaType.GEOTIFF,
            extra_fields={"raster:bands": [fields]},
        ),
    )
    return item


def _collection() -> pystac.Collection:
    return pystac.Collection(
        id="example",
        description="Example collection",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180, -90, 180, 90]]),
            pystac.TemporalExtent([[dt(2025, 1, 1, tzinfo=timezone.utc), None]]),
        ),
        license="CC-BY-4.0",
    )


def test_importing_attrs_does_not_import_stac() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import geosave_engine.geodata.attrs; "
            "assert 'geosave_engine.geodata.stac' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_reading_uses_band_description_not_common_name() -> None:
    timestamp = dt(2025, 1, 1)
    item = _item("scene-1", timestamp, {"unit": "1"})
    item.assets["red"].extra_fields["eo:bands"] = [
        {"name": "B04", "common_name": "red", "description": "Surface reflectance"}
    ]

    header = create_header(
        [item], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    cf = header.data_vars["red"].get(CFVariable)
    assert cf == CFVariable(long_name="Surface reflectance", units="1")


@pytest.mark.parametrize("second", [{}, {"scale": 0.0002}])
def test_disagreeing_packing_drops_with_a_warning(second) -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"scale": 0.0001}),
        _item("scene-2", dt(2025, 1, 2), second),
    ]

    with pytest.warns(DroppedAttrsWarning, match="scale_factor") as warned:
        header = create_header(
            items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
        )

    assert header.data_vars["red"].get(Packing) is None
    assert "'scene-2'" in str(warned[0].message)


def test_an_item_without_the_asset_drops_its_fields_with_a_warning() -> None:
    first = _item("scene-1", dt(2025, 1, 1), {"unit": "1"})
    second = _item("scene-2", dt(2025, 1, 2), {"unit": "1"})
    del second.assets["red"]

    with pytest.warns(DroppedAttrsWarning, match="units"):
        header = create_header(
            [first, second],
            _collection(),
            xr.Dataset({"red": ("x", [1])}),
            groupby="id",
        )

    assert header.data_vars["red"].get(CFVariable) is None


def test_reading_writes_no_packing_when_no_item_publishes_it() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {"unit": "1"}),
    ]

    header = create_header(
        items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(Packing) is None


def test_reading_drops_a_label_missing_from_one_item() -> None:
    items = [
        _item("scene-1", dt(2025, 1, 1), {"unit": "1"}),
        _item("scene-2", dt(2025, 1, 2), {}),
    ]

    with pytest.warns(DroppedAttrsWarning, match="units"):
        header = create_header(
            items, _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
        )

    assert header.data_vars["red"].get(CFVariable) is None


def test_reading_rejects_an_empty_item_sequence() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        create_header([], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id")


def test_loader_nodata_is_authoritative_over_source_values() -> None:
    items = [
        _item("first", dt(2025, 1, 1), {"nodata": 0}),
        _item("second", dt(2025, 1, 2), {"nodata": 1}),
    ]
    loaded = xr.Dataset({"red": xr.DataArray([1.0], attrs={"nodata": -9999})})

    header = create_header(items, _collection(), loaded, groupby="id")

    assert header.data_vars["red"].get(Nodata).fill_value == -9999


def test_reader_uses_the_selected_band_without_mutating_the_item() -> None:
    item = _item("multi", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "raster:bands": [{"unit": "1", "scale": 0.1}, {"unit": "K", "offset": 0}],
        "eo:bands": [{"description": "Reflectance"}, {"description": "Temperature"}],
    }
    before = item.to_dict()

    header = create_header(
        [item],
        _collection(),
        xr.Dataset({"red.1": ("x", [1]), "red.2": ("x", [2])}),
        groupby="id",
    )

    assert header.data_vars["red.1"].get(CFVariable) == CFVariable(
        units="1", long_name="Reflectance"
    )
    assert header.data_vars["red.1"].get(Packing) == Packing(scale_factor=0.1)
    assert header.data_vars["red.2"].get(CFVariable) == CFVariable(
        units="K", long_name="Temperature"
    )
    assert header.data_vars["red.2"].get(Packing) == Packing(add_offset=0)
    assert item.to_dict() == before


@pytest.mark.parametrize("selection", [None, (), ("unit", "missing")])
def test_captured_properties_select_only_requested_fields(selection) -> None:
    item = _item("scene", dt(2025, 1, 1), {"unit": "1", "scale": 0.1})
    item.properties["platform"] = "example"

    header = create_header(
        [item],
        _collection(),
        xr.Dataset({"red": ("x", [1])}),
        groupby="id",
        item_properties=("platform", "missing"),
        asset_fields=selection,
    )

    metadata = header.root.get(StacMetadata)
    assert metadata.stac_items[0].properties == {"platform": "example"}
    expected = (
        {"unit": "1", "scale": 0.1}
        if selection is None
        else ({} if not selection else {"unit": "1"})
    )
    assert metadata.stac_items[0].assets == ({"red": expected} if expected else {})


def test_asset_reader_keeps_core_band_precedence() -> None:
    asset = pystac.Asset(
        "https://example.com/a.tif",
        extra_fields={
            "unit": "asset",
            "description": "asset",
            "raster:bands": [{"unit": "raster"}],
            "eo:bands": [{"description": "EO"}],
            "bands": [{"unit": "core", "description": "Core"}],
        },
    )

    assert read_asset_fields(asset) == {"unit": "core", "description": "Core"}


def test_loader_packing_overrides_the_agreed_source_packing() -> None:
    item = _item("scene", dt(2025, 1, 1), {"scale": 0.1, "offset": 2})
    loaded = xr.Dataset({"red": xr.DataArray([1], attrs={"scale_factor": 1.0})})

    header = create_header([item], _collection(), loaded, groupby="id")

    assert header.data_vars["red"].get(Packing) == Packing(scale_factor=1, add_offset=2)


def test_spectral_facts_reach_the_variable() -> None:
    item = _item("scene", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "eo:bands": [{"name": "B04", "common_name": "red", "center_wavelength": 0.665}]
    }

    header = create_header(
        [item], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(Spectral) == Spectral(
        common_name="red", center_wavelength=0.665
    )


def test_core_bands_read_like_legacy_ones() -> None:
    item = _item("scene", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "bands": [{"name": "B04", "unit": "1", "raster:scale": 0.0001}]
    }

    header = create_header(
        [item], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id"
    )

    assert header.data_vars["red"].get(CFVariable) == CFVariable(units="1")
    assert header.data_vars["red"].get(Packing) == Packing(scale_factor=0.0001)


def test_a_band_states_units_and_packing_as_attrs() -> None:
    raster = RasterBand.create(nodata=0, unit="1", scale=0.0001)
    spectral = Band.create(name="red", description="Red")

    assert band_attrs(raster, spectral) == {
        "units": "1",
        "long_name": "Red",
        "scale_factor": 0.0001,
    }


def test_a_bare_band_states_no_attrs() -> None:
    assert band_attrs(RasterBand({"data_type": "float32"}), Band({"name": "dem"})) == {}


def test_legacy_listings_pair_by_index() -> None:
    asset = pystac.Asset(
        "a.tif",
        extra_fields={
            "raster:bands": [{"data_type": "uint16", "nodata": 0, "scale": 0.0001}],
            "eo:bands": [{"name": "B04", "common_name": "red"}, {"name": "B08"}],
        },
    )

    (first, red), (second, nir) = read_bands(asset)

    assert (first.scale, red.common_name) == (0.0001, "red")
    assert (second.to_dict(), nir.name) == ({}, "B08")


def test_core_bands_read_without_parquet_nulls() -> None:
    asset = pystac.Asset(
        "a.tif",
        extra_fields={
            "bands": [{"name": "label", "unit": None, "raster:scale": None}],
            "raster:bands": [{"unit": "ignored"}],
        },
    )

    ((raster, spectral),) = read_bands(asset)

    assert (raster.to_dict(), spectral.to_dict()) == ({}, {"name": "label"})


def test_an_asset_without_bands_reads_as_none() -> None:
    assert read_bands(pystac.Asset("thumbnail.png")) == []
```

- [ ] **Step 2: Run it and see it fail.**

Run: `uv run pytest tests/geodata/attrs/headers/test_stac.py -q`
Expected: failures; `read_bands` still returns mappings and `band_attrs` takes one argument.

- [ ] **Step 3: Replace** `src/geosave_engine/geodata/attrs/headers/stac.py` with:

```python
"""Create an attrs header from one STAC load."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import odc.stac
from pystac.extensions.classification import RasterBandClassificationExtension
from pystac.extensions.eo import AssetEOExtension, Band
from pystac.extensions.raster import AssetRasterExtension, RasterBand

from geosave_engine import __path__ as _package_paths
from geosave_engine.geodata.utils.datetime import naive_utc
from geosave_engine.geodata.warnings import DroppedAttrsWarning

from ..header import AttrsHeader
from ..model import AttrsModel, FlatAttrs
from ..models import CFVariable, Legend, Packing, Spectral
from ..namespace import AttrsNamespace

if TYPE_CHECKING:
    import pystac
    import xarray as xr


# Asset fields that list bands, in the spelling each STAC generation used.
_BAND_LISTINGS = ("bands", "raster:bands", "eo:bands")


def create_header(
    items: Sequence[pystac.Item],
    collection: pystac.Collection,
    loaded: xr.Dataset,
    *,
    groupby: str,
    item_properties: Sequence[str] | None = (),
    asset_fields: Sequence[str] | None = None,
    stac_cfg: dict[str, Any] | None = None,
) -> AttrsHeader:
    """Create the attrs header described by one STAC load.

    The collection describes the whole raster and the items record where it
    came from, so both land on the root; each asset describes the variable it
    loads into, where a field the items publish differently is dropped.

    Args:
        items: Items making up one load, in search order.
        collection: Collection the items belong to.
        loaded: Output Dataset whose loader attrs describe effective pixels.
        groupby: Grouping mode `odc.stac.load` was asked to apply.
        item_properties: Item property names to record. Empty records identity
            only; None records every property.
        asset_fields: Asset field names to record. Empty records none; None
            records every field an asset publishes.
        stac_cfg: The same ODC conversion settings used for loading.

    Returns:
        Header to rebase onto the loaded Dataset.

    Raises:
        ValueError: `items` is empty, or an item publishes no instant to record.

    Warns:
        DroppedAttrsWarning: The items publish a field differently, or only
            some of them publish it.

    Examples:
        >>> header = create_header(matched, collection, loaded, groupby="solar_day")
        >>> header.data_vars["B04"].to_attrs()
        {'units': '1', 'long_name': 'Red', '_FillValue': 0, 'nodata': 0,
         'scale_factor': 0.0001, 'add_offset': -0.1, 'common_name': 'red'}
    """
    if not items:
        raise ValueError("creating a STAC header needs at least one item")

    parsed = list(odc.stac.parse_items(items, cfg=stac_cfg))
    providers = collection.providers or []
    rows = [
        {
            "id": item.id,
            "datetime": naive_utc(entry.nominal_datetime),
            "properties": _selected(item.properties, item_properties),
            "assets": _item_assets(item, asset_fields),
        }
        for entry, item in zip(parsed, items, strict=True)
    ]
    root: dict[str, Any] = {
        "id": collection.id,
        "title": collection.title,
        "summary": collection.description,
        "keywords": ", ".join(collection.keywords) if collection.keywords else None,
        "institution": providers[0].name if providers else None,
        "license": collection.license,
        "stac_groupby": groupby,
        "stac_items": rows,
    }

    variables = {}
    for name, variable in loaded.data_vars.items():
        published = []
        for item, entry in zip(items, parsed, strict=True):
            asset_name, band_index = entry.collection.band_key(str(name))
            asset = item.assets.get(asset_name)
            bands = read_bands(asset) if asset else []
            # An item may lack the asset, or list fewer bands than the file holds.
            if len(bands) >= band_index:
                published.append(band_attrs(*bands[band_index - 1]))
            else:
                published.append({})
        merged, dropped = AttrsNamespace.merge(
            [AttrsNamespace.from_attrs(fields, "variable") for fields in published],
            conflicts="drop",
        )
        disputed = sorted(dropped)
        if disputed:
            values = {
                key: {
                    item.id: item_attrs.get(key)
                    for item, item_attrs in zip(items, published, strict=True)
                }
                for key in disputed
            }
            warnings.warn(
                f"STAC items publish {disputed} differently for {str(name)!r}, so "
                f"those source attrs are dropped: {values}",
                DroppedAttrsWarning,
                skip_file_prefixes=tuple(_package_paths),
            )
        # The loader's own attrs describe the pixels it actually produced.
        variables[str(name)] = {**merged.to_attrs(), **variable.attrs}
    return AttrsHeader.from_attrs(
        root={
            **loaded.attrs,
            **{key: value for key, value in root.items() if value is not None},
        },
        data_vars=variables,
    )


def read_asset_fields(asset: pystac.Asset, *, band_index: int = 1) -> dict[str, object]:
    """Read one asset's fields, merging the band description it nests.

    Args:
        asset: Asset to read.
        band_index: One-based source band index.

    Returns:
        {
            "<field>": its value,
        }
        The asset's own fields without its band listings, overlaid with the
        selected band as the Raster and EO extensions spell it, as `"scale"`.

    Examples:
        >>> read_asset_fields(asset)
        {'gsd': 10, 'unit': '1', 'scale': 0.0001, 'name': 'B04', 'common_name': 'red'}
    """
    found = {
        key: value
        for key, value in asset.extra_fields.items()
        if key not in _BAND_LISTINGS
    }
    bands = read_bands(asset)
    if len(bands) >= band_index:
        raster, spectral = bands[band_index - 1]
        found.update(raster.to_dict())
        found.update(spectral.to_dict())
    return found


def read_bands(asset: pystac.Asset) -> list[tuple[RasterBand, Band]]:
    """Return an asset's bands as PySTAC's Raster and EO bands, in file order.

    Args:
        asset: Asset listing its bands as `raster:bands` and `eo:bands`, or
            as the STAC 1.1 `bands` PySTAC has no class for.

    Returns:
        One (raster band, EO band) pair per band either listing reaches,
        without the nulls Parquet fills absent keys with. Empty where the
        asset lists no bands.

    Examples:
        >>> raster, spectral = read_bands(asset)[0]
        >>> raster.scale, spectral.common_name
        (0.0001, 'red')
    """
    core = asset.extra_fields.get("bands")
    if core is not None:
        pairs = [_legacy(band) for band in core]
    else:
        stored = AssetRasterExtension(asset).bands or []
        spectral = AssetEOExtension(asset).bands or []
        pairs = [
            (
                stored[index].properties if index < len(stored) else {},
                spectral[index].properties if index < len(spectral) else {},
            )
            for index in range(max(len(stored), len(spectral)))
        ]
    return [(RasterBand(_stated(raster)), Band(_stated(eo))) for raster, eo in pairs]


def _legacy(band: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split one STAC 1.1 band into the Raster and EO bands PySTAC models.

    Args:
        band: One entry of an asset's `bands`.

    Returns:
        (raster fields, EO fields). `raster:` and `eo:` keys lose their
        prefix; `name` and `description` name the EO band; every other key
        describes the stored values and stays with the raster band.
    """
    raster: dict[str, Any] = {}
    spectral: dict[str, Any] = {}
    for key, value in band.items():
        if key.startswith("eo:"):
            spectral[key.removeprefix("eo:")] = value
        elif key in ("name", "description"):
            spectral[key] = value
        else:
            raster[key.removeprefix("raster:")] = value
    return raster, spectral


def _stated(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the nulls Parquet fills a key other rows carry with."""
    return {key: value for key, value in fields.items() if value is not None}


def band_attrs(raster: RasterBand, spectral: Band) -> FlatAttrs:
    """Translate one band into the variable attrs it states.

    `nodata` and `data_type` describe the file, and the loader that opens it
    states them, so neither is translated.

    Args:
        raster: The band as the Raster extension describes it.
        spectral: The band as the EO extension describes it.

    Returns:
        {
            "<attr key>": its value,
        }
        The keys of `CFVariable`, `Packing`, `Spectral` and `Legend` the band
        states; empty where it states none.

    Examples:
        >>> band_attrs(RasterBand.create(unit="1", scale=0.0001), Band({}))
        {'units': '1', 'scale_factor': 0.0001}
    """
    models: list[AttrsModel] = []
    if raster.unit is not None or spectral.description is not None:
        models.append(CFVariable(units=raster.unit, long_name=spectral.description))
    if raster.scale is not None or raster.offset is not None:
        models.append(Packing(scale_factor=raster.scale, add_offset=raster.offset))
    facts = {
        "common_name": spectral.common_name,
        "center_wavelength": spectral.center_wavelength,
        "full_width_half_max": spectral.full_width_half_max,
    }
    if any(value is not None for value in facts.values()):
        models.append(Spectral(**facts))
    classes = RasterBandClassificationExtension(raster).classes
    if classes:
        legend = Legend(class_map={entry.value: entry.name for entry in classes})
        colors = {
            entry.value: f"#{entry.color_hint}"
            for entry in classes
            if entry.color_hint is not None
        }
        if colors:
            legend = Legend(class_map=legend.class_map, color_map=colors)
        models.append(legend)

    attrs: FlatAttrs = {}
    for model in models:
        attrs.update(model.to_attrs())
    return attrs


def _item_assets(
    item: pystac.Item, asset_fields: Sequence[str] | None
) -> dict[str, dict[str, object]]:
    """Read each asset's captured fields, keyed by asset name."""
    captured: dict[str, dict[str, object]] = {}
    for name, asset in item.assets.items():
        selected = _selected(read_asset_fields(asset), asset_fields)
        if selected:
            captured[name] = selected
    return captured


def _selected(
    source: Mapping[str, object], names: Sequence[str] | None
) -> dict[str, object]:
    """Keep the named keys of a mapping, None keeping every key."""
    if names is None:
        return dict(source)
    return {key: source[key] for key in names if key in source}
```

- [ ] **Step 4: Run the header tests.**

Run: `uv run pytest tests/geodata/attrs/headers/test_stac.py -q`
Expected: 22 passed.

- [ ] **Step 5: Remove the two tests that read the old band format.** In `tests/geodata/stac/test_dataflow.py` delete `test_described_rasters_round_trip_through_the_table` and `test_described_items_validate_against_their_schemas`, the `_described_rasters` helper, and the imports only they use (`pq`, `stac_table_to_items`, `band_attrs`, `read_bands`, `CFVariable`, `Legend`, `Spectral`, `create_items`, `pystac` if unused). Keep `test_observations_to_prepared_index_to_reusable_dataset`.

- [ ] **Step 6: Update asset field names in** `tests/geodata/stac/test_source.py`. Run `uv run pytest tests/geodata/stac/test_source.py -q`; where a test expects a captured or listed asset field named `raster:scale`, `raster:offset`, `eo:common_name`, `eo:center_wavelength` or `eo:full_width_half_max`, drop the prefix (`scale`, `offset`, `common_name`, `center_wavelength`, `full_width_half_max`). Change nothing else.

- [ ] **Step 7: Run the affected suites.**

Run: `uv run pytest tests/geodata/attrs tests/geodata/stac/test_source.py tests/geodata/stac/test_dataflow.py -q -m "not integration"`
Expected: all pass. `tests/geodata/attrs/extensions/` still passes; it is deleted in Task 3.

---

### Task 3: Extension modules and Items

Each extension is one module that writes its schema onto an asset from a saved raster, through that schema's PySTAC class. PySTAC builds the Item and adds the asset; the modules only write.

**Files:**
- Create: `src/geosave_engine/geodata/stac/extensions/{raster,eo,classification,geosave}.py`
- Modify: `src/geosave_engine/geodata/stac/extensions/{__init__,projection,zarr}.py` (whole files replaced)
- Create: `docs/schemas/geosave/v0.1.0/schema.json`
- Modify: `src/geosave_engine/geodata/stac/item.py` (whole file replaced)
- Modify: `src/geosave_engine/geodata/stac/__init__.py` (drop `asset` from the imports and `__all__`)
- Delete: `src/geosave_engine/geodata/stac/asset.py`, `src/geosave_engine/geodata/stac/extensions/{types,datacube,cf}.py`
- Delete: `src/geosave_engine/geodata/attrs/extensions/` (the package)
- Delete: `tests/geodata/attrs/extensions/`, `tests/geodata/stac/test_asset.py`, `tests/geodata/stac/extensions/test_{cf,datacube,schemas}.py`
- Create: `tests/geodata/stac/extensions/test_{raster,eo,classification,geosave}.py`
- Modify: `tests/geodata/stac/extensions/test_{projection,zarr}.py` (whole files replaced)
- Modify: `tests/geodata/stac/test_item.py`, `tests/geodata/core/test_stack.py`

**Interfaces:**
- Consumes: `gs.times` from Task 1.
- Produces:
  - `extensions.projection.write(asset, raster)`, `extensions.raster.write(asset, raster)`, `extensions.eo.write(asset, raster)`, `extensions.classification.write(asset, raster)`, `extensions.zarr.write(asset, raster)`; each takes an asset already added to its Item and returns None.
  - `extensions.ASSET`: those five modules in write order (Raster before Classification).
  - `ZarrExtension.ext(asset, add_if_missing=False)`, `.apply(*, zarr_format, node_type, consolidated)`, properties `zarr_format`, `node_type`, `consolidated`
  - `GeosaveExtension.ext(item, add_if_missing=False)`, property `stack`; module constant `STACK_PROP = "geosave:stack"`
  - `item.default_key(raster) -> str`; `create_collection`, `create_item`, `create_items`, `create_stack_items` with unchanged signatures. `item.STACK_ID` and the `stac.asset` module are gone.

- [ ] **Step 1: Write the module tests.** Create or replace these six files under `tests/geodata/stac/extensions/`.

`test_projection.py`:

```python
"""The Projection module writes a raster's grid."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import ProjectionExtension

from geosave_engine.geodata import raster
from geosave_engine.geodata.stac.extensions import projection


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_grid_is_written_through_the_projection_class(scene) -> None:
    asset = _asset()

    projection.write(asset, scene)

    grid = ProjectionExtension.ext(asset)
    assert (grid.code, grid.shape) == ("EPSG:32633", [64, 64])
    assert grid.transform == [10.0, 0.0, 300000.0, 0.0, -10.0, 5000640.0]
    assert asset.owner.stac_extensions == [ProjectionExtension.get_schema_uri()]


def test_a_grid_without_an_epsg_code_states_its_wkt() -> None:
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    grid = GeoBox.from_bbox((0, 0, 640, 640), crs, resolution=10)
    plain = raster({"height": (("y", "x"), np.zeros((64, 64), "float32"))}, grid)
    asset = _asset()

    projection.write(asset, plain)

    stated = ProjectionExtension.ext(asset)
    assert stated.code is None
    assert "Sinusoidal" in stated.wkt2


def test_a_raster_without_a_grid_refuses() -> None:
    bare = xr.Dataset({"height": (("y", "x"), np.zeros((2, 2), "float32"))})

    with pytest.raises(ValueError, match="no locatable grid"):
        projection.write(_asset(), bare)
```

`test_raster.py`:

```python
"""The Raster module writes how each band is typed, filled and packed."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import dask
import numpy as np
import pytest
from pystac.extensions.raster import RasterExtension

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.attrs import CFVariable
from geosave_engine.geodata.stac.extensions import raster

from tests.geodata.conftest import build_raster


def _refuse(*args, **kwargs):
    raise AssertionError("writing the Raster fields computed pixels")


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_saved_cog_states_what_it_stores(scene, tmp_path) -> None:
    path = scene.gs.to_cog(tmp_path / "forest")[0]
    asset = _asset(str(path))

    with read_raster(path) as saved, dask.config.set(scheduler=_refuse):
        raster.write(asset, saved)

    bands = RasterExtension.ext(asset).bands
    assert [band.data_type for band in bands] == ["uint16", "uint16"]
    assert bands[0].nodata == 0
    assert asset.owner.stac_extensions == [RasterExtension.get_schema_uri()]


def test_packing_and_units_reach_the_band() -> None:
    scene = build_raster(packed=True).gs.rebase(CFVariable(units="1"), target="red")
    asset = _asset()

    raster.write(asset, scene)

    red = RasterExtension.ext(asset).bands[0]
    assert (red.scale, red.offset, red.unit) == (pytest.approx(1e-4), 0.0, "1")


def test_a_plain_raster_states_only_its_stored_type() -> None:
    dem = build_raster()[["nir"]].astype("float32").rename(nir="height")
    asset = _asset()

    raster.write(asset, dem)

    assert [band.to_dict() for band in RasterExtension.ext(asset).bands] == [
        {"data_type": "float32"}
    ]


def test_a_dtype_stac_cannot_name_is_other() -> None:
    mask = build_raster()[["nir"]].astype(bool).rename(nir="mask")
    asset = _asset()

    raster.write(asset, mask)

    assert RasterExtension.ext(asset).bands[0].data_type == "other"


@pytest.mark.parametrize(
    ("fill", "expected"), [(np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf")]
)
def test_a_nonfinite_fill_uses_stac_strings(fill, expected) -> None:
    dem = build_raster()[["nir"]].astype("float32").rename(nir="height")
    dem["height"].attrs["_FillValue"] = fill
    asset = _asset()

    raster.write(asset, dem)

    assert RasterExtension.ext(asset).bands[0].nodata == expected
```

`test_eo.py`:

```python
"""The EO module names spectral bands and nothing else."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
from pystac.extensions.eo import EOExtension

from geosave_engine.geodata.attrs import CFVariable, Spectral
from geosave_engine.geodata.stac.extensions import eo

from tests.geodata.conftest import build_raster


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_spectral_raster_names_every_band() -> None:
    scene = build_raster().gs.rebase(
        CFVariable(long_name="Red"),
        Spectral(common_name="red", center_wavelength=0.665),
        target="red",
    )
    asset = _asset()

    eo.write(asset, scene)

    red, nir = EOExtension.ext(asset).bands
    assert (red.name, red.common_name, red.center_wavelength) == ("red", "red", 0.665)
    assert red.description == "Red"
    assert (nir.name, nir.common_name) == ("nir", None)
    assert asset.owner.stac_extensions == [EOExtension.get_schema_uri()]


def test_a_raster_with_no_spectral_facts_writes_nothing() -> None:
    dem = build_raster()[["nir"]].astype("float32").rename(nir="height")
    asset = _asset()

    eo.write(asset, dem)

    assert asset.extra_fields == {}
    assert asset.owner.stac_extensions == []
```

`test_classification.py`:

```python
"""The Classification module writes legends onto Raster bands."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest
from pystac.extensions.classification import ClassificationExtension
from pystac.extensions.raster import RasterExtension

from geosave_engine.geodata.attrs import Legend
from geosave_engine.geodata.stac.extensions import classification, raster

from tests.geodata.conftest import build_raster


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def _label(legend: Legend):
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    return label.gs.rebase(legend, target="label")


def test_a_legend_becomes_classes_on_its_band() -> None:
    label = _label(
        Legend(class_map={0: "background", 1: "forest"}, color_map={1: "#00ff00"})
    )
    asset = _asset()
    raster.write(asset, label)

    classification.write(asset, label)

    band = RasterExtension.ext(asset).bands[0]
    classes = ClassificationExtension.ext(band).classes
    assert [(entry.value, entry.name) for entry in classes] == [
        (0, "background"),
        (1, "forest"),
    ]
    assert classes[1].color_hint == "00FF00"
    assert ClassificationExtension.get_schema_uri() in asset.owner.stac_extensions


def test_a_raster_without_a_legend_declares_nothing(scene) -> None:
    asset = _asset()
    raster.write(asset, scene)

    classification.write(asset, scene)

    assert asset.owner.stac_extensions == [RasterExtension.get_schema_uri()]


def test_a_class_name_stac_refuses_raises() -> None:
    label = _label(Legend(class_map={1: "tree/shrub"}))
    asset = _asset()
    raster.write(asset, label)

    with pytest.raises(ValueError, match="tree/shrub"):
        classification.write(asset, label)


def test_bit_masks_write_no_classes() -> None:
    flags = build_raster()[["nir"]].astype("uint8").rename(nir="qa")
    flags["qa"].attrs.update(flag_masks=[1, 2], flag_meanings="cloud shadow")
    asset = _asset()
    raster.write(asset, flags)

    classification.write(asset, flags)

    assert RasterExtension.ext(asset).bands[0].to_dict() == {"data_type": "uint8"}


def test_classes_need_the_raster_bands_written_first() -> None:
    label = _label(Legend(class_map={1: "forest"}))
    asset = _asset()
    RasterExtension.add_to(asset.owner)

    with pytest.raises(ValueError):
        classification.write(asset, label)
```

`test_zarr.py`:

```python
"""The Zarr module states a store's layout through GeoSave's own class."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.stac.extensions import ZarrExtension, zarr


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_store_states_its_layout(scene, tmp_path) -> None:
    path = scene.gs.to_zarr(tmp_path / "cube.zarr")
    asset = _asset(str(path))

    with read_raster(path) as saved:
        zarr.write(asset, saved)

    store = ZarrExtension.ext(asset)
    assert (store.zarr_format, store.node_type, store.consolidated) == (
        3,
        "group",
        False,
    )
    assert asset.owner.stac_extensions == [ZarrExtension.get_schema_uri()]


def test_a_raster_not_opened_from_a_store_writes_nothing(scene) -> None:
    asset = _asset()

    zarr.write(asset, scene)

    assert asset.extra_fields == {}
    assert asset.owner.stac_extensions == []


def test_fields_read_back_through_the_class() -> None:
    asset = _asset("cube.zarr")
    ZarrExtension.ext(asset, add_if_missing=True).apply(
        zarr_format=2, node_type="group", consolidated=True
    )

    restored = pystac.Item.from_dict(asset.owner.to_dict()).assets["image"]

    assert asset.extra_fields == {
        "zarr:zarr_format": 2,
        "zarr:node_type": "group",
        "zarr:consolidated": True,
    }
    assert ZarrExtension.ext(restored).consolidated is True


def test_an_undeclared_extension_refuses() -> None:
    with pytest.raises(pystac.ExtensionNotImplemented):
        ZarrExtension.ext(_asset())
```

`test_geosave.py`:

```python
"""GeoSave's own Item properties go through its extension class."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import pytest

from geosave_engine.geodata.stac.extensions import GeosaveExtension


def _item() -> pystac.Item:
    return pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})


def test_the_stack_is_an_item_property() -> None:
    item = _item()

    GeosaveExtension.ext(item, add_if_missing=True).stack = "s0"

    assert item.properties == {"geosave:stack": "s0"}
    assert item.stac_extensions == [GeosaveExtension.get_schema_uri()]
    assert GeosaveExtension.ext(pystac.Item.from_dict(item.to_dict())).stack == "s0"


def test_an_undeclared_extension_refuses() -> None:
    with pytest.raises(pystac.ExtensionNotImplemented):
        GeosaveExtension.ext(_item())
```

- [ ] **Step 2: Run them and see them fail.**

Run: `uv run pytest tests/geodata/stac/extensions -q`
Expected: collection errors such as `ImportError: cannot import name 'GeosaveExtension'`.

- [ ] **Step 3: Write the Projection module.** Replace `src/geosave_engine/geodata/stac/extensions/projection.py` with:

```python
"""Projection extension: the grid a file's pixels sit on."""

from __future__ import annotations

import pystac
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import ProjectionExtension


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write a raster's grid as Projection fields.

    Args:
        asset: Asset already added to its Item.
        raster: Raster carrying a grid with a CRS.

    Raises:
        ValueError: The raster carries no locatable grid.

    Examples:
        >>> write(asset, scene)
        >>> ProjectionExtension.ext(asset).code
        'EPSG:32749'
    """
    geobox = raster.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError(
            f"{asset.href} carries no locatable grid, which a STAC asset states; "
            f"keep it in an ordinary reference table instead"
        )
    epsg = geobox.crs.epsg
    ProjectionExtension.ext(asset, add_if_missing=True).apply(
        code=None if epsg is None else f"EPSG:{epsg}",
        wkt2=geobox.crs.to_wkt() if epsg is None else None,
        shape=list(geobox.shape),
        transform=list(geobox.transform)[:6],
    )
```

- [ ] **Step 4: Write the Raster module.** Create `src/geosave_engine/geodata/stac/extensions/raster.py`:

```python
"""Raster extension: how each band's stored numbers are typed, filled and packed."""

from __future__ import annotations

from math import isfinite

import pystac
import xarray as xr
from pystac.extensions.raster import (
    DataType,
    NoDataStrings,
    RasterBand,
    RasterExtension,
)

from geosave_engine.geodata.attrs import CFVariable, Nodata, Packing


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write one Raster band per variable, in file order.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `read_raster` opens it, holding stored values. No
            pixel is read.

    Examples:
        >>> write(asset, scene)
        >>> RasterExtension.ext(asset).bands[0].scale
        0.0001
    """
    header = raster.gs.attrs
    bands = []
    for name in raster.gs.variables:
        variable = header.data_vars[name]
        nodata = variable.get(Nodata)
        packing = variable.get(Packing)
        semantics = variable.get(CFVariable)

        fill = None if nodata is None else nodata.fill_value
        if isinstance(fill, float) and not isfinite(fill):
            fill = NoDataStrings(str(fill))
        dtype = raster[name].dtype.name
        bands.append(
            RasterBand.create(
                nodata=fill,
                data_type=DataType(dtype)
                if dtype in list(DataType)
                else DataType.OTHER,
                scale=None if packing is None else packing.scale_factor,
                offset=None if packing is None else packing.add_offset,
                unit=None if semantics is None else semantics.units,
            )
        )
    RasterExtension.ext(asset, add_if_missing=True).apply(bands)
```

- [ ] **Step 5: Write the EO module.** Create `src/geosave_engine/geodata/stac/extensions/eo.py`:

```python
"""EO extension: what each band is called and where in the spectrum it sits."""

from __future__ import annotations

import pystac
import xarray as xr
from pystac.extensions.eo import Band, EOExtension

from geosave_engine.geodata.attrs import CFVariable, Spectral


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write one EO band per variable where the raster is spectral.

    A label or a DEM is not an optical band, so a raster none of whose
    variables carries `Spectral` writes nothing; its names stay in the file.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `read_raster` opens it.

    Examples:
        >>> write(asset, scene)
        >>> EOExtension.ext(asset).bands[0].common_name
        'red'
    """
    header = raster.gs.attrs
    variables = {name: header.data_vars[name] for name in raster.gs.variables}
    if all(variable.get(Spectral) is None for variable in variables.values()):
        return

    bands = []
    for name, variable in variables.items():
        facts = variable.get(Spectral)
        semantics = variable.get(CFVariable)
        bands.append(
            Band.create(
                name=name,
                common_name=None if facts is None else facts.common_name,
                center_wavelength=None if facts is None else facts.center_wavelength,
                full_width_half_max=None
                if facts is None
                else facts.full_width_half_max,
                description=None if semantics is None else semantics.long_name,
            )
        )
    EOExtension.ext(asset, add_if_missing=True).apply(bands)
```

- [ ] **Step 6: Write the Classification module.** Create `src/geosave_engine/geodata/stac/extensions/classification.py`:

```python
"""Classification extension: the classes a band's pixel values name."""

from __future__ import annotations

import re
from typing import cast

import pystac
import xarray as xr
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.raster import RasterExtension

from geosave_engine.geodata.attrs import Legend
from geosave_engine.geodata.utils.color import parse_color

# What the schema lets a class be named.
_CLASS_NAME = re.compile(r"[0-9A-Za-z_-]+")


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write each variable's legend as classes on its Raster band.

    Classes sit inside the band they describe, so the Raster extension must
    have written the asset's bands first.

    Args:
        asset: Asset already added to its Item, carrying its Raster bands.
        raster: Raster as `read_raster` opens it. A variable with no legend,
            or one that lists bit masks, writes nothing.

    Raises:
        ValueError: A class name carries a character the schema refuses.

    Examples:
        >>> write(asset, label)
        >>> band = RasterExtension.ext(asset).bands[0]
        >>> ClassificationExtension.ext(band).classes[1].name
        'forest'
    """
    header = raster.gs.attrs
    bands = RasterExtension.ext(asset).bands or []
    for name, band in zip(raster.gs.variables, bands, strict=True):
        legend = header.data_vars[name].get(Legend)
        # CF masks do not state the bit offset, length and names a STAC bitfield needs.
        if legend is None or legend.class_map is None or legend.flag_masks is not None:
            continue
        ClassificationExtension.ext(band).apply(classes=_classes(legend))
        # A band has no owner to declare the schema on.
        ClassificationExtension.add_to(cast("pystac.Item", asset.owner))


def _classes(legend: Legend) -> list[Classification]:
    """Spell a legend's classes as Classification classes."""
    colors = legend.color_map or {}
    classes = []
    for value, name in (legend.class_map or {}).items():
        if not _CLASS_NAME.fullmatch(name):
            raise ValueError(
                f"class name {name!r} for value {value} carries a character STAC "
                f"refuses; use only letters, digits, '-' and '_'"
            )
        hint = None
        if value in colors:
            red, green, blue = parse_color(colors[value])
            hint = f"{red:02X}{green:02X}{blue:02X}"
        classes.append(Classification.create(value=value, name=name, color_hint=hint))
    return classes
```

- [ ] **Step 7: Write the Zarr module.** Replace `src/geosave_engine/geodata/stac/extensions/zarr.py` with:

```python
"""Zarr extension: the store metadata needed to open a saved hierarchy."""

from __future__ import annotations

from typing import Literal

import pystac
import xarray as xr
from pystac.extensions.base import ExtensionManagementMixin, PropertiesExtension

SCHEMA_URI = "https://stac-extensions.github.io/zarr/v1.1.0/schema.json"

ZARR_FORMAT_PROP = "zarr:zarr_format"
NODE_TYPE_PROP = "zarr:node_type"
CONSOLIDATED_PROP = "zarr:consolidated"


class ZarrExtension(PropertiesExtension, ExtensionManagementMixin[pystac.Item]):
    """Zarr store metadata on one asset.

    PySTAC ships no class for the Zarr extension, so this one follows the
    shape of its own: build it with `ext`, write with `apply`.

    Args:
        asset: Asset pointing at a Zarr store.

    Examples:
        >>> zarr = ZarrExtension.ext(asset, add_if_missing=True)
        >>> zarr.apply(zarr_format=3, node_type="group", consolidated=False)
        >>> zarr.zarr_format
        3
    """

    name: Literal["zarr"] = "zarr"

    def __init__(self, asset: pystac.Asset) -> None:
        """Wrap the fields of one asset."""
        self.asset = asset
        self.properties = asset.extra_fields

    def apply(self, *, zarr_format: int, node_type: str, consolidated: bool) -> None:
        """State how the store is laid out.

        Args:
            zarr_format: Zarr specification version the store follows, 2 or 3.
            node_type: What the href points at, `"group"` or `"array"`.
            consolidated: Whether the store carries consolidated metadata.
        """
        self.zarr_format = zarr_format
        self.node_type = node_type
        self.consolidated = consolidated

    @property
    def zarr_format(self) -> int | None:
        """Return the Zarr specification version the store follows."""
        return self._get_property(ZARR_FORMAT_PROP, int)

    @zarr_format.setter
    def zarr_format(self, value: int | None) -> None:
        self._set_property(ZARR_FORMAT_PROP, value)

    @property
    def node_type(self) -> str | None:
        """Return whether the href points at a group or an array."""
        return self._get_property(NODE_TYPE_PROP, str)

    @node_type.setter
    def node_type(self, value: str | None) -> None:
        self._set_property(NODE_TYPE_PROP, value)

    @property
    def consolidated(self) -> bool | None:
        """Return whether the store carries consolidated metadata."""
        return self._get_property(CONSOLIDATED_PROP, bool)

    @consolidated.setter
    def consolidated(self, value: bool | None) -> None:
        self._set_property(CONSOLIDATED_PROP, value)

    @classmethod
    def get_schema_uri(cls) -> str:
        """Return the schema an Item declares when an asset uses these fields."""
        return SCHEMA_URI

    @classmethod
    def ext(cls, obj: pystac.Asset, add_if_missing: bool = False) -> ZarrExtension:
        """Extend an asset with the Zarr fields.

        Args:
            obj: Asset already added to its Item.
            add_if_missing: Declare the schema on the Item when it is absent.

        Returns:
            The extension over the asset's fields.

        Raises:
            pystac.ExtensionNotImplemented: The Item does not declare the
                schema and `add_if_missing` is false.
        """
        cls.ensure_owner_has_extension(obj, add_if_missing)
        return cls(obj)


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write the layout of the Zarr store a raster was opened from.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `read_raster` opens it. One not opened from a Zarr
            store writes nothing.

    Examples:
        >>> write(asset, read_raster("samples/forest.zarr"))
        >>> ZarrExtension.ext(asset).node_type
        'group'
    """
    store = raster.encoding.get("zarr")
    if store is None:
        return
    ZarrExtension.ext(asset, add_if_missing=True).apply(
        zarr_format=store["zarr_format"],
        node_type=store["node_type"],
        consolidated=store["consolidated"],
    )
```

- [ ] **Step 8: Write the GeoSave extension.** Create `src/geosave_engine/geodata/stac/extensions/geosave.py`:

```python
"""GeoSave extension: the fields GeoSave adds to a STAC Item."""

from __future__ import annotations

from typing import Literal

import pystac
from pystac.extensions.base import ExtensionManagementMixin, PropertiesExtension

SCHEMA_URI = (
    "https://weedkat.github.io/geosave-engine/schemas/geosave/v0.1.0/schema.json"
)

STACK_PROP = "geosave:stack"


class GeosaveExtension(PropertiesExtension, ExtensionManagementMixin[pystac.Item]):
    """GeoSave's own properties on one Item.

    Args:
        item: Item to read or write.

    Examples:
        >>> GeosaveExtension.ext(item, add_if_missing=True).stack = "s0"
        >>> GeosaveExtension.ext(item).stack
        's0'
    """

    name: Literal["geosave"] = "geosave"

    def __init__(self, item: pystac.Item) -> None:
        """Wrap the properties of one Item."""
        self.item = item
        self.properties = item.properties

    @property
    def stack(self) -> str | None:
        """Return the stack this Item was saved from, shared by its groups."""
        return self._get_property(STACK_PROP, str)

    @stack.setter
    def stack(self, value: str | None) -> None:
        self._set_property(STACK_PROP, value)

    @classmethod
    def get_schema_uri(cls) -> str:
        """Return the schema an Item declares when it uses these fields."""
        return SCHEMA_URI

    @classmethod
    def ext(cls, obj: pystac.Item, add_if_missing: bool = False) -> GeosaveExtension:
        """Extend an Item with GeoSave's properties.

        Args:
            obj: Item to extend.
            add_if_missing: Declare the schema on the Item when it is absent.

        Returns:
            The extension over the Item's properties.

        Raises:
            pystac.ExtensionNotImplemented: The Item does not declare the
                schema and `add_if_missing` is false.
        """
        cls.ensure_has_extension(obj, add_if_missing)
        return cls(obj)
```

- [ ] **Step 9: Add the schema the class points at.** Create `docs/schemas/geosave/v0.1.0/schema.json` (not run against a validator):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://weedkat.github.io/geosave-engine/schemas/geosave/v0.1.0/schema.json",
  "title": "GeoSave Extension",
  "description": "Fields GeoSave adds to a STAC Item.",
  "type": "object",
  "required": ["stac_extensions", "type", "properties"],
  "properties": {
    "type": {"const": "Feature"},
    "stac_extensions": {
      "type": "array",
      "contains": {
        "const": "https://weedkat.github.io/geosave-engine/schemas/geosave/v0.1.0/schema.json"
      }
    },
    "properties": {
      "type": "object",
      "properties": {
        "geosave:stack": {
          "title": "Stack this Item was saved from, shared by its groups",
          "type": "string",
          "minLength": 1
        }
      }
    }
  }
}
```

- [ ] **Step 10: List the modules.** Replace `src/geosave_engine/geodata/stac/extensions/__init__.py` with:

```python
"""STAC extensions GeoSave writes, one module per schema.

Each module writes its schema's fields onto an asset from a saved raster,
through that schema's PySTAC class, or through GeoSave's own class where
PySTAC ships none.
"""

from . import classification, eo, projection, raster, zarr
from .geosave import GeosaveExtension
from .zarr import ZarrExtension

# Written onto every asset, in this order: classes sit inside the bands the
# Raster extension stores, so it comes before Classification.
ASSET = (projection, raster, eo, classification, zarr)

__all__ = [
    "ASSET",
    "GeosaveExtension",
    "ZarrExtension",
    "classification",
    "eo",
    "projection",
    "raster",
    "zarr",
]
```

- [ ] **Step 11: Replace** `src/geosave_engine/geodata/stac/item.py` with:

```python
"""Build native STAC Items and Collections from saved rasters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC
from datetime import datetime as DateTime
from os import PathLike
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from shapely import union_all
from shapely.geometry import mapping

from geosave_engine.geodata.io.readers import read_raster
from geosave_engine.geodata.io.storage import absolute_location

from . import extensions
from .extensions import GeosaveExtension

type ItemTime = DateTime | tuple[DateTime, DateTime]
type Saved = Mapping[str, xr.Dataset]

# What STAC calls a saved raster, by its suffix.
_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".zarr": pystac.MediaType.ZARR,
    ".nc": pystac.MediaType.NETCDF,
    ".nc4": pystac.MediaType.NETCDF,
    ".cdf": pystac.MediaType.NETCDF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
}


def create_collection(
    id: str,
    *,
    description: str,
    license: str = "other",
    title: str | None = None,
    providers: Sequence[pystac.Provider] | None = None,
) -> pystac.Collection:
    """Build a Collection that Items are created into.

    Args:
        id: Collection identity, which its Items carry.
        description: What the Collection holds.
        license: SPDX license identifier, or `"other"`.
        title: Short human-readable name.
        providers: Organizations that produced, processed or host the data.

    Returns:
        Collection covering the whole globe over an unbounded time range,
        until a table states what its rows cover.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> create_items(paths, collection=forest)[0].collection_id
        'forest'
    """
    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
        pystac.TemporalExtent([[None, None]]),
    )
    return pystac.Collection(
        id=id,
        description=description,
        extent=extent,
        title=title,
        license=license,
        providers=None if providers is None else list(providers),
    )


def create_item(
    assets: Mapping[str, str | PathLike[str]],
    *,
    id: str,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> pystac.Item:
    """Build one Item from saved rasters the caller names.

    Args:
        assets: Asset key mapped to the file or store holding it, as
            `{"label": "s0/label.tif"}`.
        id: Item identity.
        collection: Collection the Item belongs to; the Item takes its id.
            None states none.
        datetime: Time for timeless files, as one instant or a start and end.
            None takes the instants of the files' `time` coordinate, widened
            to what a `TimeSpec` on it says each label covers.
        **options: Read options passed to `read_raster` for every asset.

    Returns:
        Item over the files' footprint, dated by one instant or by a start
        and end, declaring the extensions its assets use. No pixel is read.

    Raises:
        ValueError: An asset has no locatable grid or names no raster format,
            or no asset is dated and `datetime` is None.

    Examples:
        >>> create_item({"label": "s0/label.tif"}, id="s0").id
        's0'
    """
    with ExitStack() as opened:
        named = {key: _open(opened, path, options) for key, path in assets.items()}
        return _item(named, id=id, collection=_identity(collection), when=datetime)


def create_items(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    id: str | None = None,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> tuple[pystac.Item, ...]:
    """Build the Items of one saved raster.

    Args:
        paths: What a raster's `to_cog`, `to_zarr` or `to_netcdf` returned.
        id: `GeoAnchor.format` template filled from each Item's own raster,
            as `"{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}"`. None names each Item
            after its file, or the folder its files share.
        collection: Collection the Items belong to; they take its id. None
            states none.
        datetime: Time for timeless files.
        **options: Read options passed to `read_raster` for every file.

    Returns:
        One Item per instant the files hold, in the order given, with one
        asset per file; a store is one Item. No pixel is read.

    Raises:
        ValueError: No path is given, two files of one instant claim the same
            asset key, `id` gives two Items the same id, or a file is
            timeless and `datetime` is None.

    Examples:
        >>> [item.id for item in create_items(ds.gs.to_cog("samples/forest"))]
        ['forest_20250601T103031', 'forest_20250611T103031']
    """
    files = [paths] if isinstance(paths, (str, PathLike)) else list(paths)
    if not files:
        raise ValueError("create_items needs the paths a writer returned")
    with ExitStack() as opened:
        saved = dict(_open(opened, path, options) for path in files)
        return _items(saved, id=id, collection=_identity(collection), when=datetime)


def create_stack_items(
    paths: Mapping[str, str | PathLike[str] | Sequence[str | PathLike[str]]],
    *,
    name: str,
    datetime: ItemTime | None = None,
    **options: Any,
) -> tuple[pystac.Item, ...]:
    """Build the Items of every group of a saved stack.

    Args:
        paths: What a stack's `to_cog`, `to_zarr` or `to_netcdf` returned:
            group names mapped to the files, or the one store, holding each.
        name: The stack's identity, shared by its Items.
        datetime: Time for timeless groups. None gives them the span the
            dated groups cover.
        **options: Read options passed to `read_raster` for every file.

    Returns:
        Each group's Items, groups in the order `paths` lists them. A group's
        name is its collection, ids are prefixed with `name`, and every Item
        carries `geosave:stack`.

    Raises:
        ValueError: A group is timeless, no group is dated and `datetime` is
            None.

    Examples:
        >>> items = create_stack_items(sample.gs.to_cog("samples/s0"), name="s0")
        >>> [item.id for item in items]
        ['s0/optical_20250601T103031', 's0/optical_20250611T103031', 's0/label']
        >>> items[0].collection_id
        'optical'
    """
    with ExitStack() as opened:
        groups: dict[str, Saved] = {}
        for group, saved in paths.items():
            files = [saved] if isinstance(saved, (str, PathLike)) else list(saved)
            groups[group] = dict(_open(opened, path, options) for path in files)

        # A timeless group is dated by what the dated groups cover.
        spans = [
            raster.gs.timespan
            for files in groups.values()
            for raster in files.values()
            if raster.gs.timespan is not None
        ]
        fallback = datetime
        if fallback is None and spans:
            fallback = (min(start for start, _ in spans), max(end for _, end in spans))

        items: list[pystac.Item] = []
        for group, files in groups.items():
            dated = any(raster.gs.timespan is not None for raster in files.values())
            built = _items(
                files,
                id=None,
                collection=group,
                when=None if dated else fallback,
                prefix=f"{name}/",
            )
            for each in built:
                GeosaveExtension.ext(each, add_if_missing=True).stack = name
            items.extend(built)
        return tuple(items)


def default_key(raster: xr.Dataset) -> str:
    """Name the asset holding a whole raster.

    Args:
        raster: Raster one file or store holds.

    Returns:
        The raster's variable where it has exactly one, else `"image"`.

    Examples:
        >>> default_key(scene[["red"]]), default_key(scene)
        ('red', 'image')
    """
    variables = raster.gs.variables
    if len(variables) == 1:
        return variables[0]
    return "image"


def _media_type(href: str) -> str:
    """Name the media type of the raster file or store an href points at."""
    found = _MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if found is None:
        raise ValueError(
            f"{href} names no raster file or store; pass a path a writer returned"
        )
    return found


def _identity(collection: pystac.Collection | None) -> str | None:
    """Return the id an Item of this Collection carries, None for no Collection."""
    return None if collection is None else collection.id


def _open(
    opened: ExitStack, path: str | PathLike[str], options: Mapping[str, Any]
) -> tuple[str, xr.Dataset]:
    """Open one saved raster lazily, to be closed when `opened` exits."""
    raster = opened.enter_context(read_raster(path, **options))
    return absolute_location(path), raster


def _items(
    saved: Saved,
    *,
    id: str | None,
    collection: str | None,
    when: ItemTime | None,
    prefix: str = "",
) -> tuple[pystac.Item, ...]:
    """Build one Item per scene the saved files hold.

    Args:
        saved: Absolute href mapped to the raster opened from it, files of
            one raster in the order its writer returned them.
        id: `GeoAnchor.format` template, or None for a name from the path.
        collection: Collection id the Items carry, or None.
        when: Time for timeless files.
        prefix: Text put before every id.

    Returns:
        One Item per scene, in the order the scenes first appear.

    Raises:
        ValueError: Two files of one scene claim the same asset key, or two
            Items share an id.
    """
    # Files covering the same time are one scene; a store covers its own axis.
    scenes: dict[object, dict[str, xr.Dataset]] = {}
    for href, raster in saved.items():
        scenes.setdefault(raster.gs.timespan, {})[href] = raster

    items = []
    for files in scenes.values():
        named: dict[str, tuple[str, xr.Dataset]] = {}
        for href, raster in files.items():
            key = default_key(raster)
            if key in named:
                raise ValueError(
                    f"{href} and {named[key][0]} cover the same time and would "
                    f"both be the asset {key!r}; build them as separate Items"
                )
            named[key] = (href, raster)

        # One file is named after itself; several after the folder they share.
        first_href, first = next(iter(files.items()))
        written = PurePosixPath(first_href)
        name = written.stem if len(files) == 1 else written.parent.name
        item_id = name if id is None else first.gs.anchor.format(id)
        items.append(
            _item(named, id=f"{prefix}{item_id}", collection=collection, when=when)
        )

    ids = [each.id for each in items]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"id template {id!r} gives several scenes the same id {ids}; add a "
            f"time field such as {{start:%Y%m%d}}"
        )
    return tuple(items)


def _item(
    named: Mapping[str, tuple[str, xr.Dataset]],
    *,
    id: str,
    collection: str | None,
    when: ItemTime | None,
) -> pystac.Item:
    """Build one Item from opened rasters keyed by asset name.

    Args:
        named: Asset key mapped to the href and the raster opened from it.
        id: Item identity.
        collection: Collection id the Item carries, or None.
        when: Time for timeless rasters.

    Returns:
        Item with one asset per raster, over their joined footprint.

    Raises:
        ValueError: A raster has no locatable grid, or none is dated and
            `when` is None.
    """
    rasters = [raster for _, raster in named.values()]
    if isinstance(when, tuple):
        start, end = when
    elif when is not None:
        start, end = when, when
    else:
        start, end = _covered(rasters, list(named))

    grids = [raster.gs.geobox for raster in rasters]
    if any(grid is None or grid.crs is None for grid in grids):
        footprint = None
    else:
        footprint = union_all([grid.geographic_extent.geom for grid in grids])

    # STAC dates an Item by one instant, or by a start and an end, never both.
    ranged = start != end
    item = pystac.Item(
        id,
        None if footprint is None else mapping(footprint),
        None if footprint is None else list(footprint.bounds),
        None if ranged else start,
        {},
        start_datetime=start if ranged else None,
        end_datetime=end if ranged else None,
        collection=collection,
    )
    for key, (href, raster) in named.items():
        # An extension declares its schema on the asset's owner, so add first.
        item.add_asset(
            key, pystac.Asset(href, media_type=_media_type(href), roles=["data"])
        )
        for extension in extensions.ASSET:
            extension.write(item.assets[key], raster)
    return item


def _covered(
    rasters: Sequence[xr.Dataset], keys: Sequence[str]
) -> tuple[DateTime, DateTime]:
    """Return the first and last instant the rasters of one Item cover.

    Args:
        rasters: Rasters of one Item, as opened from their files.
        keys: Their asset keys, for the error.

    Returns:
        (first, last) in UTC. Rasters holding one time label between them
        cover that instant; rasters holding several cover what `gs.timespan`
        says, so a monthly label covers its month.

    Raises:
        ValueError: No raster is dated.
    """
    labels = [raster.gs.times for raster in rasters if raster.gs.times is not None]
    spans = [raster.gs.timespan for raster in rasters if raster.gs.timespan is not None]
    if not spans:
        raise ValueError(
            f"none of the assets {keys} is dated, and a STAC Item needs a "
            f"time; pass datetime="
        )
    instants = pd.DatetimeIndex(
        np.concatenate([each.values for each in labels])
    ).unique()
    if len(instants) == 1:
        instant = instants[0].tz_localize("UTC").to_pydatetime()
        return instant, instant
    start = min(start for start, _ in spans).replace(tzinfo=UTC)
    end = max(end for _, end in spans).replace(tzinfo=UTC)
    return start, end
```

- [ ] **Step 12: Delete what nothing imports now**, and drop `asset` from `src/geosave_engine/geodata/stac/__init__.py` (both the `from . import asset, item, table` line and `__all__`).

```bash
rm src/geosave_engine/geodata/stac/asset.py \
   src/geosave_engine/geodata/stac/extensions/types.py \
   src/geosave_engine/geodata/stac/extensions/datacube.py \
   src/geosave_engine/geodata/stac/extensions/cf.py
rm -r src/geosave_engine/geodata/attrs/extensions tests/geodata/attrs/extensions
rm tests/geodata/stac/test_asset.py tests/geodata/stac/extensions/test_cf.py \
   tests/geodata/stac/extensions/test_datacube.py tests/geodata/stac/extensions/test_schemas.py
grep -rn "attrs.extensions\|stac.asset\|stac import asset\|STACK_ID\|band_fields\|from_raster" src
```

Expected from the `grep`: only `ml/segmentation/supervised/data.py` (its `STACK_ID` import; fixed in Task 4) and prose in READMEs or docstrings naming `stac.asset.from_raster`, which Task 5 rewrites.

- [ ] **Step 13: Run the module tests.**

Run: `uv run pytest tests/geodata/stac/extensions -q`
Expected: 23 passed.

- [ ] **Step 14: Adapt** `tests/geodata/stac/test_item.py`. Apply these rules, then run the file:

| Old | New |
| --- | --- |
| `from geosave_engine.geodata.attrs.extensions import classification` and `from geosave_engine.geodata.stac.extensions import projection` | `from pystac.extensions.classification import ClassificationExtension`, `from pystac.extensions.projection import ProjectionExtension`, `from pystac.extensions.raster import RasterExtension` |
| `projection.SCHEMA` / `classification.SCHEMA` | `ProjectionExtension.get_schema_uri()` / `ClassificationExtension.get_schema_uri()` |
| `asset.extra_fields["bands"][i][...]` | `RasterExtension.ext(asset).bands[i].<property>` (`data_type`, `nodata`, `scale`, `unit`) |
| `asset.extra_fields["proj:code"]` and other `proj:` keys | `ProjectionExtension.ext(asset).code` and the matching property |
| `item.properties["geosave:stack"]` | `GeosaveExtension.ext(item).stack` |
| expected `stac_extensions` of a plain scene | `[ProjectionExtension.get_schema_uri(), RasterExtension.get_schema_uri()]` |
| expected `stac_extensions` of a label with a legend | Projection, Raster, Classification URIs, compared as a set |
| a store's `end_datetime` of `2025-06-11T00:00:00Z` | `2025-06-11T23:59:59.999999Z` (a date label covers its day, as `gs.timespan` reads it) |
| any assertion on `cube:` or `cf:` fields | delete the assertion |

`test_one_file_per_scene_is_one_item_per_instant` must assert each Item's `datetime` is set and `start_datetime` is absent; add that assertion if it is missing:

```python
    assert [each.datetime.isoformat() for each in items] == [
        "2025-06-01T00:00:00+00:00",
        "2025-06-11T00:00:00+00:00",
    ]
    assert all("start_datetime" not in each.properties for each in items)
```

Add one test that the Item builder runs every module, so a module left out of `ASSET` is caught:

```python
def test_every_extension_module_writes_onto_each_asset(scene, tmp_path) -> None:
    from geosave_engine.geodata.stac.extensions import ZarrExtension

    (store,) = stac.create_items(scene.gs.to_zarr(tmp_path / "cube.zarr"))

    asset = store.assets["image"]
    assert ProjectionExtension.ext(asset).code == "EPSG:32633"
    assert len(RasterExtension.ext(asset).bands) == 2
    assert ZarrExtension.ext(asset).node_type == "group"
    assert asset.media_type == "application/vnd+zarr"
    assert asset.roles == ["data"]
```

Tests in `test_item.py` that call `table` stay failing until Task 4.

- [ ] **Step 15: Run** `tests/geodata/core/test_stack.py`: `item.properties["geosave:stack"]` stays valid as written; no change expected.

- [ ] **Step 16: Run the suites this task owns.**

Run: `uv run pytest tests/geodata/stac/extensions tests/geodata/stac/test_item.py tests/geodata/attrs -q -m "not integration"`
Expected: all pass except tests in `test_item.py` that go through `table`, which Task 4 owns.

---

### Task 4: The table, its client, and the manifest

**Files:**
- Modify: `src/geosave_engine/geodata/stac/table.py` (whole file replaced)
- Modify: `src/geosave_engine/geodata/stac/client.py` (`StacTableClient` replaced, imports trimmed)
- Modify: `src/geosave_engine/ml/segmentation/supervised/data.py`
- Modify: `tests/geodata/stac/test_table.py` (whole file replaced)
- Modify: `tests/geodata/stac/test_native_items.py`, `tests/geodata/stac/test_table_client.py`, `tests/geodata/stac/test_dataflow.py`, `tests/geodata/stac/test_item.py`, `tests/geodata/core/test_stack.py`, `tests/geodata/io/test_remote.py`, `tests/ml/segmentation/supervised/test_data.py`, `tests/cli/core/test_workspace.py`

**Interfaces:**
- Consumes: Items from Task 3; `GeosaveExtension`, `STACK_PROP`.
- Produces:
  - `table.from_items(items) -> GeoDataFrame`
  - `table.to_items(rows) -> list[pystac.Item]`
  - `table.write(items_or_rows, path, *, collections=None, overwrite=False, storage_options=None) -> Path | str`
  - `table.read(path, **options) -> GeoDataFrame`
  - `table.read_collections(path, *, storage_options=None) -> dict[str, pystac.Collection]`
  - `table.parse_collections(value: bytes | str) -> dict[str, pystac.Collection]`
  - `table.load(items, *, assets=None, **options) -> Dataset`
  - `StacTableClient(path, *, duckdb=None)` over one file.

- [ ] **Step 1: Replace** `tests/geodata/stac/test_table.py` with:

```python
"""Item tables: Items to rows, to one file, and back to pixels."""

from __future__ import annotations

import shutil

import dask
import numpy as np
import pyarrow.parquet as pq
import pytest
from pystac.extensions.raster import RasterExtension

from geosave_engine.geodata import GeoVector, stac, stack
from geosave_engine.geodata.stac import table
from geosave_engine.geodata.stac.extensions import GeosaveExtension


def _refuse(*args, **kwargs):
    raise AssertionError("the table computed pixels")


def _cog_items(scene, root, name="forest", **options):
    return stac.create_items(scene.gs.to_cog(root / name), **options)


def _stack_items(scene, label, root, name="s0"):
    sample = stack({"optical": scene, "label": label})
    return stac.create_stack_items(sample.gs.to_cog(root / name), name=name)


def _collections():
    return [
        stac.create_collection(name, description=f"{name} rows", license="CC-BY-4.0")
        for name in ("optical", "label")
    ]


def test_items_become_rows_in_wgs84(scene, tmp_path) -> None:
    rows = table.from_items(_cog_items(scene, tmp_path))

    assert rows["id"].tolist() == ["forest_20250601T000000", "forest_20250611T000000"]
    assert rows.crs.to_epsg() == 4326
    assert "bbox" not in rows


def test_no_items_refuses() -> None:
    with pytest.raises(ValueError, match="at least one STAC Item"):
        table.from_items([])


def test_rows_become_the_items_they_came_from(scene, tmp_path) -> None:
    items = _cog_items(scene, tmp_path)

    restored = table.to_items(table.from_items(items))

    assert [item.id for item in restored] == [item.id for item in items]
    assert restored[0].bbox == pytest.approx(items[0].bbox)
    assert restored[0].assets["image"].href == items[0].assets["image"].href
    assert RasterExtension.ext(restored[0].assets["image"]).bands[0].nodata == 0


def test_filtered_rows_keep_only_the_assets_they_have(scene, label, tmp_path) -> None:
    rows = table.from_items(_stack_items(scene, label, tmp_path))

    labels = table.to_items(rows[rows["collection"] == "label"])

    assert [list(item.assets) for item in labels] == [["label"]]
    assert GeosaveExtension.ext(labels[0]).stack == "s0"


def test_a_written_table_is_stac_geoparquet(scene, tmp_path) -> None:
    path = table.write(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    metadata = pq.read_metadata(path).metadata
    assert b"stac-geoparquet" in metadata
    assert b"geo" in metadata


def test_rows_load_back_as_the_raster_that_was_written(scene, tmp_path) -> None:
    path = table.write(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    with dask.config.set(scheduler=_refuse):
        restored = table.load(table.to_items(table.read(path)))

    assert dict(restored.sizes) == {"time": 2, "y": 64, "x": 64}
    np.testing.assert_array_equal(restored.red.values, scene.red.values)
    restored.close()


def test_read_rows_carry_hrefs_that_open(scene, tmp_path) -> None:
    items = _cog_items(scene, tmp_path)
    path = table.write(items, tmp_path / "catalog" / "items.parquet")

    rows = table.read(path)

    assert rows["assets"][0]["image"]["href"] == items[0].assets["image"].href


def test_a_table_follows_its_assets_when_moved(scene, tmp_path) -> None:
    root = tmp_path / "dataset"
    table.write(_cog_items(scene, root), root / "items.parquet")

    shutil.move(root, tmp_path / "moved")

    rows = table.read(tmp_path / "moved" / "items.parquet")
    restored = table.load(table.to_items(rows))
    np.testing.assert_array_equal(restored.red.values, scene.red.values)
    restored.close()


def test_each_group_of_a_stack_loads_alone(scene, label, tmp_path) -> None:
    path = table.write(_stack_items(scene, label, tmp_path), tmp_path / "items.parquet")
    rows = table.read(path)

    optical = table.load(table.to_items(rows[rows["collection"] == "optical"]))
    classes = table.load(table.to_items(rows[rows["collection"] == "label"]))

    assert optical.gs.variables == ("red", "nir")
    assert classes.gs.variables == ("label",)
    optical.close()
    classes.close()


def test_a_named_asset_no_item_has_is_a_key_error(scene, tmp_path) -> None:
    with pytest.raises(KeyError, match="missing"):
        table.load(_cog_items(scene, tmp_path), assets="missing")


def test_only_data_assets_load(scene, tmp_path) -> None:
    import pystac

    (item, *_) = _cog_items(scene, tmp_path)
    item.add_asset("thumbnail", pystac.Asset("preview.png", roles=["thumbnail"]))

    restored = table.load([item])

    assert restored.gs.variables == ("red", "nir")
    restored.close()


def test_an_existing_table_is_not_replaced_unasked(scene, tmp_path) -> None:
    items = _cog_items(scene, tmp_path)
    path = table.write(items, tmp_path / "items.parquet")

    with pytest.raises(FileExistsError):
        table.write(items, path)


def test_collections_ride_in_the_table_with_the_extent_of_their_rows(
    scene, label, tmp_path
) -> None:
    given = _collections()
    path = table.write(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=given,
    )

    stored = table.read_collections(path)

    assert sorted(stored) == ["label", "optical"]
    assert stored["optical"].license == "CC-BY-4.0"
    assert stored["optical"].extent.temporal.to_dict()["interval"] == [
        ["2025-06-01T00:00:00Z", "2025-06-11T00:00:00Z"]
    ]
    # The Collections handed in keep the open extent they came with.
    assert given[0].extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]


def test_items_of_a_stored_collection_link_to_it(scene, label, tmp_path) -> None:
    path = table.write(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections(),
    )

    (item, *_) = table.to_items(table.read(path))

    assert [link.rel for link in item.links] == ["collection"]


def test_a_row_naming_an_absent_collection_refuses(scene, label, tmp_path) -> None:
    with pytest.raises(ValueError, match="'label'"):
        table.write(
            _stack_items(scene, label, tmp_path),
            tmp_path / "items.parquet",
            collections=_collections()[:1],
        )


def test_a_table_without_collections_reads_none(scene, tmp_path) -> None:
    path = table.write(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    assert table.read_collections(path) == {}


def test_a_batch_concatenated_onto_a_table_reads_back_whole(
    scene, label, tmp_path
) -> None:
    path = table.write(
        _stack_items(scene, label, tmp_path, "s0"),
        tmp_path / "items.parquet",
        collections=_collections(),
    )
    more = table.from_items(_stack_items(scene, label, tmp_path, "s1"))

    grown = GeoVector.concat([table.read(path), more])
    table.write(
        grown,
        path,
        collections=table.read_collections(path).values(),
        overwrite=True,
    )

    rows = table.read(path)
    assert sorted(set(rows["geosave:stack"])) == ["s0", "s1"]
    assert sorted(table.read_collections(path)) == ["label", "optical"]
    restored = table.load(
        table.to_items(
            rows[(rows["geosave:stack"] == "s1") & (rows["collection"] == "optical")]
        )
    )
    np.testing.assert_array_equal(restored.red.values, scene.red.values)
    restored.close()


def test_a_rewrite_without_collections_stores_none(scene, label, tmp_path) -> None:
    path = table.write(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections(),
    )

    table.write(table.read(path), path, overwrite=True)

    assert table.read_collections(path) == {}


def test_a_bbox_filter_reads_only_matching_rows(scene, tmp_path) -> None:
    path = table.write(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    assert table.read(path, bbox=(0.0, 0.0, 1.0, 1.0)).empty
```

- [ ] **Step 2: Run it and see it fail.**

Run: `uv run pytest tests/geodata/stac/test_table.py -q`
Expected: failures such as `AttributeError: module ... has no attribute 'to_items'`.

- [ ] **Step 3: Replace** `src/geosave_engine/geodata/stac/table.py` with:

```python
"""Keep STAC Items as a table: in memory, on disk, and back to pixels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pystac
import rustac
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

from geosave_engine.geodata.io import geoparquet, storage
from geosave_engine.geodata.io.readers import read_raster, read_vector
from geosave_engine.geodata.io.storage import absolute_location

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset, GeoDataFrame
    from geosave_engine.geodata.io.storage import StorageOptions


def from_items(items: Iterable[pystac.Item]) -> GeoDataFrame:
    """Build a table with one row per Item.

    Args:
        items: Items to tabulate. They are not changed.

    Returns:
        GeoDataFrame in WGS84 with the Items' properties as columns and their
        assets as mappings.

    Raises:
        ValueError: No Item is given.

    Examples:
        >>> from_items(create_items(paths))["id"].tolist()
        ['forest_20250601T000000', 'forest_20250611T000000']
    """
    records = [item.to_dict(include_self_link=False) for item in items]
    if not records:
        raise ValueError("from_items needs at least one STAC Item")
    table = parse_stac_items_to_arrow(records).read_all()
    frame = gpd.GeoDataFrame.from_arrow(table)
    # geopandas drops this covering column on read, so no item table carries it.
    return cast("GeoDataFrame", frame.drop(columns="bbox"))


def to_items(rows: gpd.GeoDataFrame) -> list[pystac.Item]:
    """Build the Item each row of an item table states.

    Args:
        rows: Rows of an item table, as `from_items` or `read` returns them,
            filtered or concatenated like any GeoDataFrame.

    Returns:
        One Item per row, in row order, holding only the assets its row has.

    Examples:
        >>> [item.id for item in to_items(rows[rows["collection"] == "label"])]
        ['s0/label']
    """
    found = rustac.from_arrow(_arrow(rows))
    return [pystac.Item.from_dict(fields) for fields in found["features"]]


def write(
    items: Iterable[pystac.Item] | gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    collections: Iterable[pystac.Collection] | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> Path | str:
    """Write Items, or an item table, as one stac-geoparquet file.

    Asset hrefs are stored relative to the table, so a table and its assets
    move together. Add a batch by concatenating its rows onto the table's and
    writing again, passing the Collections the table already holds.

    Args:
        items: Items to store, or a table of them as `read` returns it.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        collections: Collections the rows belong to, stored in the file's
            metadata with the extent their rows cover. The objects are not
            changed. None stores none.
        overwrite: Replace an existing file when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Local writes return a path; URL writes return the supplied URL.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: No Item is given, the suffix is wrong, or `collections`
            is given and a row names a collection absent from it.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> items = create_items(paths, collection=forest)
        >>> write(items, "data/catalog.parquet", collections=[forest])
        PosixPath('data/catalog.parquet')
    """
    location = absolute_location(path)
    given = to_items(items) if isinstance(items, gpd.GeoDataFrame) else list(items)

    described: dict[str, pystac.Collection] = {}
    for collection in collections or ():
        clone = collection.clone()
        # A Collection stored in a table has no document of its own to link.
        clone.clear_links()
        clone.set_self_href(location)
        described[clone.id] = clone

    stored = []
    for item in given:
        clone = item.clone()
        clone.set_self_href(location)
        clone.make_asset_hrefs_relative()
        if described and clone.collection_id is not None:
            if clone.collection_id not in described:
                raise ValueError(
                    f"item {clone.id!r} names the collection "
                    f"{clone.collection_id!r}, which is not among the collections "
                    f"given {sorted(described)}; pass it too"
                )
            # The Item schema requires a link beside a collection id.
            clone.set_collection(described[clone.collection_id])
        stored.append(clone)

    table = _arrow(from_items(stored))
    fields = {}
    for collection_id, collection in described.items():
        members = [item for item in stored if item.collection_id == collection_id]
        if members:
            collection.extent = pystac.Extent.from_items(members)
        collection.clear_links()
        fields[collection_id] = collection.to_dict(include_self_link=False)

    return storage.write(
        path,
        lambda target: to_parquet(table, target, collections=fields or None),
        suffixes=geoparquet.FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )


def read(path: str | PathLike[str], **options: Any) -> GeoDataFrame:
    """Read an item table.

    Args:
        path: Parquet file, as a path or URL.
        **options: GeoParquet read options. `bbox` and `filters` are applied
            while reading.

    Returns:
        GeoDataFrame whose asset hrefs open directly, each row holding only
        the assets it has.

    Examples:
        >>> rows = read("data/catalog.parquet", bbox=(12.0, 45.0, 13.0, 46.0))
        >>> rows.gs.query(anchor)["id"].tolist()
        ['forest_20250601T000000']
    """
    frame = read_vector(path, **options)
    if "assets" not in frame or frame.empty:
        return frame
    location = absolute_location(path)
    items = to_items(frame)
    for item in items:
        item.set_self_href(location)
        item.make_asset_hrefs_absolute()
    return from_items(items)


def read_collections(
    path: str | PathLike[str], *, storage_options: StorageOptions | None = None
) -> dict[str, pystac.Collection]:
    """Return the Collections an item table carries, keyed by id.

    Args:
        path: Parquet file, as a path or URL.
        storage_options: Options for the filesystem a URL names.

    Returns:
        {
            "<collection id>": its Collection, with the extent of its rows,
        }
        Empty where the file carries none.

    Examples:
        >>> read_collections("data/catalog.parquet")["forest"].license
        'CC-BY-4.0'
    """
    filesystem, target = storage.filesystem_path(path, storage_options)
    with filesystem.open(target, "rb") as handle:
        metadata = pq.read_metadata(handle).metadata or {}
    return parse_collections(metadata.get(b"stac-geoparquet", b"{}"))


def parse_collections(value: bytes | str) -> dict[str, pystac.Collection]:
    """Decode the Collections one file's `stac-geoparquet` metadata states.

    Args:
        value: The metadata value, as JSON.

    Returns:
        {
            "<collection id>": its Collection,
        }

    Examples:
        >>> parse_collections(b'{"version": "1.0.0"}')
        {}
    """
    stored = orjson.loads(value).get("collections", {})
    return {
        collection_id: pystac.Collection.from_dict(fields)
        for collection_id, fields in stored.items()
    }


def load(
    items: Iterable[pystac.Item],
    *,
    assets: str | Sequence[str] | None = None,
    **options: Any,
) -> Dataset:
    """Open the rasters Items point at as one lazy Dataset.

    Args:
        items: Items holding one raster: a store, or the scenes of one
            product.
        assets: Asset name or names to read. None reads every data asset.
        **options: Raster read options. Chunks default to an empty mapping.

    Returns:
        Lazy Dataset; scenes join along `time`.

    Raises:
        KeyError: A named asset is in no Item.
        ValueError: No Item has a data asset, or the files do not form one
            raster.

    Examples:
        >>> load(to_items(rows[rows["collection"] == "optical"])).sizes["time"]
        2
    """
    wanted = [assets] if isinstance(assets, str) else assets
    hrefs = []
    found = set()
    for item in items:
        for key, asset in item.assets.items():
            if wanted is not None and key not in wanted:
                continue
            if asset.roles is not None and not asset.has_role("data"):
                continue
            hrefs.append(asset.get_absolute_href())
            found.add(key)

    for key in wanted or ():
        if key not in found:
            raise KeyError(key)
    if not hrefs:
        raise ValueError("load needs at least one Item with a data asset")
    source = hrefs[0] if len(hrefs) == 1 else hrefs
    return read_raster(source, **{"chunks": {}, **options})


def _arrow(rows: gpd.GeoDataFrame) -> pa.Table:
    """Spell rows as the Arrow table the format states, bounds and all."""
    # A STAC bbox is its geometry's bounds; geopandas dropped the column on read.
    bounds = rows.geometry.bounds
    bounds.columns = ["xmin", "ymin", "xmax", "ymax"]
    boxes = pd.Series(bounds.to_dict("records"), index=rows.index)
    return pa.table(rows.assign(bbox=boxes).to_arrow())
```

- [ ] **Step 4: Replace `StacTableClient`** in `src/geosave_engine/geodata/stac/client.py` with the class below (not run). In `StacClient.open`, drop the `is_folder` branch so only a `.parquet` / `.geoparquet` suffix opens a table, then remove the imports nothing uses any more (`Path`, `storage`); `ruff check` names them.

```python
class StacTableClient:
    """Search one STAC GeoParquet table with rustac and build sources on it.

    Item properties are columns of a table, so a sort or filter names them
    without the `properties.` prefix a STAC API takes.

    Args:
        path: Parquet file as `stac.table.write` stores it, as a path or URL.
        duckdb: Configured rustac session, needed for a bucket behind
            credentials or a custom endpoint. None creates a session.

    Examples:
        >>> client = StacTableClient("samples/catalog.parquet")
        >>> client.collections()
        {'forest'}
        >>> source = client.source("forest")
    """

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        duckdb: rustac.DuckdbClient | None = None,
    ) -> None:
        """Bind the table, reading nothing yet.

        Args:
            path: Parquet file, as a path or URL.
            duckdb: Caller-configured rustac session, or None to create one.
        """
        self._location = absolute_location(path)
        self._duckdb = duckdb if duckdb is not None else rustac.DuckdbClient()

    def search(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]:
        """Run one search over the table.

        Args:
            query: Query object, or raw `rustac` search parameters.

        Returns:
            Every matching Item, its asset hrefs absolute.

        Examples:
            >>> query = StacQuery(collections=["forest"])
            >>> len(client.search(query.set_filter("eo:cloud_cover <= 10")))
            3
        """
        given = query.to_search_params() if isinstance(query, StacQuery) else query
        params = dict(given)

        # rustac takes one RFC 3339 string, a range as "start/end".
        when = params.get("datetime")
        if isinstance(when, tuple):
            start, end = when
            params["datetime"] = f"{datetime_to_str(start)}/{datetime_to_str(end)}"
        elif when is not None and not isinstance(when, str):
            params["datetime"] = datetime_to_str(when)

        items = []
        for fields in self._duckdb.search(self._location, **params):
            item = pystac.Item.from_dict(fields)
            # A table stores asset hrefs relative to itself.
            item.set_self_href(self._location)
            item.make_asset_hrefs_absolute()
            items.append(item)
        return items

    def collections(self) -> set[str]:
        """Name the collections this table's rows belong to.

        Returns:
            Collection IDs.
        """
        return set(self._described())

    def collection(self, collection: str) -> pystac.Collection:
        """Read one collection of this table.

        Args:
            collection: Collection ID.

        Returns:
            The Collection the table stores under that id, or the one rustac
            derives from its rows where the table stores none.

        Raises:
            CollectionNotFoundError: No stored Collection and no row carries
                `collection`.
        """
        described = self._described()
        if collection not in described:
            raise CollectionNotFoundError(
                f"collection {collection!r} not found in {self._location}; "
                f"call collections() to see what it holds"
            )
        return described[collection]

    def source(self, collection: str) -> StacSource:
        """Build a source for one collection of this table.

        Args:
            collection: Collection ID. Discover them with `collections`.

        Returns:
            Source for `collection`, on every default.

        Raises:
            CollectionNotFoundError: `collection` is not in this table.

        Examples:
            >>> source = client.source("forest").set_config(bands=["red", "nir"])
        """
        self.collection(collection)
        return StacSource(self, collection=collection)

    def _described(self) -> dict[str, pystac.Collection]:
        """Return every collection of the table, keyed by id.

        Returns:
            {
                "<collection id>": the Collection the table stores, or the
                    one derived from the rows carrying that id,
            }
        """
        described = {}
        for fields in self._duckdb.get_collections(self._location):
            described[fields["id"]] = pystac.Collection.from_dict(fields)
        # What a writer stored says more than what rows alone can; the same
        # session reads it, so one configuration reaches a remote table.
        metadata = self._duckdb.query_to_table(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'stac-geoparquet'",
            [self._location],
        )
        for row in pa.table(metadata).to_pylist():
            described.update(table.parse_collections(row["value"]))
        return described
```

- [ ] **Step 5: Move the manifest onto Items** in `src/geosave_engine/ml/segmentation/supervised/data.py` (not run):

Replace the two imports

```python
from geosave_engine.geodata.stac.item import STACK_ID
from stac_geoparquet import to_dict
```

with

```python
from geosave_engine.geodata.stac.extensions import GeosaveExtension
from geosave_engine.geodata.stac.extensions.geosave import STACK_PROP
```

Replace every remaining `STACK_ID` with `STACK_PROP` (the manifest column keeps its name).

In `_make_reference`, replace

```python
        rows = [self._source_rows[key] for key in reference.parent_id]
        reference["source_id"] = [row[STACK_ID] for row in rows]
        # Stored STAC fields describe the raw sample, not these prepared pixels.
        properties = [to_dict(row)["properties"] for row in rows]
```

with

```python
        records = [self._source_rows[key] for key in reference.parent_id]
        reference["source_id"] = [
            GeosaveExtension.ext(record).stack for record in records
        ]
        # Stored STAC fields describe the raw sample, not these prepared pixels.
        properties = [record.properties for record in records]
```

In `_prepare`, replace

```python
            sample = stack({name: table.load(part) for name, part in groups.items()})
            # The target's row is the sample's record: it carries the annotations.
            row = groups[self.target].iloc[0]
```

with

```python
            items = {name: table.to_items(part) for name, part in groups.items()}
            sample = stack({name: table.load(part) for name, part in items.items()})
            # The target's Item is the sample's record: it carries the annotations.
            record = items[self.target][0]
```

and `self._source_rows[parent_id] = row.to_dict()` with `self._source_rows[parent_id] = record`. If `_source_rows` has a type annotation, make it `dict[str, pystac.Item]` and import `pystac` under `TYPE_CHECKING`.

- [ ] **Step 6: Run the table tests.**

Run: `uv run pytest tests/geodata/stac/test_table.py -q`
Expected: all pass.

- [ ] **Step 7: Adapt the remaining tests** with these rules, running each file after editing it:

| Old | New |
| --- | --- |
| `table.load(rows)` / `table.load(rows[...], **options)` | `table.load(table.to_items(rows))` / `table.load(table.to_items(rows[...]), **options)` |
| `stac_table_to_items(pq.read_table(path))` followed by `pystac.Item.from_dict` | `table.to_items(table.read(path))` |
| `row["geosave:stack"]` on an Item | `GeosaveExtension.ext(item).stack` |
| a catalog fixture writing `part-0.parquet` and `part-1.parquet` into a folder | one `table.write([first, second], folder / "items.parquet", collections=[forest])`, and the client opened on that file |

Per file:

- `tests/geodata/stac/test_native_items.py`: delete `test_a_table_keeps_its_bbox_through_read_and_rewrite` (rows carry no `bbox` now; `test_rows_become_the_items_they_came_from` covers it). Keep `test_an_edited_geometry_changes_the_bbox_written`, asserting on `table.to_items(table.read(path))[0].bbox`.
- `tests/geodata/stac/test_table_client.py`: apply the fixture rule. Keep every test; in `test_table_client_reads_stored_collections_through_rustac` and `test_a_custom_s3_endpoint_reads_items_and_stored_collections` point the client at the file.
- `tests/geodata/stac/test_dataflow.py`: add the successors of the two tests Task 2 removed:

```python
def test_described_rasters_round_trip_through_the_table(tmp_path) -> None:
    optical = build_raster(times=2, packed=True)[["red"]]
    optical = optical.gs.rebase(
        CFVariable(units="1"),
        Spectral(common_name="red", center_wavelength=0.665),
        target="red",
    )
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    label = label.gs.rebase(
        Legend(class_map={0: "background", 1: "forest"}, color_map={1: "#00ff00"}),
        target="label",
    )
    when = datetime(2025, 6, 1, tzinfo=UTC)
    items = [
        *create_items(optical.gs.to_cog(tmp_path / "optical")),
        *create_items(label.gs.to_cog(tmp_path / "label"), datetime=when),
    ]

    saved = table.write(items, tmp_path / "items.parquet")

    restored = table.to_items(table.read(saved))
    red, spectral = read_bands(restored[0].assets["red"])[0]
    classes, named = read_bands(restored[-1].assets["label"])[0]
    assert red.data_type == "uint16"
    assert band_attrs(red, spectral) == {
        "units": "1",
        "long_name": "red",
        "scale_factor": pytest.approx(1e-4),
        "common_name": "red",
        "center_wavelength": 0.665,
    }
    legend = Legend.from_attrs(band_attrs(classes, named))
    assert legend.class_map == {0: "background", 1: "forest"}


@pytest.mark.integration
def test_described_items_validate_against_their_schemas(tmp_path) -> None:
    scene = build_raster(times=1, packed=True)

    (item,) = create_items(scene.gs.to_cog(tmp_path / "scene"))

    item.validate()
```

  with imports `from geosave_engine.geodata.attrs import CFVariable, Legend, Spectral`, `from geosave_engine.geodata.attrs.headers.stac import band_attrs, read_bands`, `from geosave_engine.geodata.stac import create_items, table`.
- `tests/geodata/stac/test_item.py`, `tests/geodata/core/test_stack.py`, `tests/geodata/io/test_remote.py`, `tests/cli/core/test_workspace.py`: apply the first two rules.
- `tests/ml/segmentation/supervised/test_data.py`: no change expected; the manifest is still `table.read(path)` and still has a `geosave:stack` column.

- [ ] **Step 8: Run everything STAC touches.**

Run: `uv run pytest tests/geodata tests/ml/segmentation tests/workflow tests/cli/core/test_workspace.py -q -m "not integration and not slow"`
Expected: all pass.

---

### Task 5: Guides and full verification

**Files:**
- Modify: `docs/guides/architecture.md` (the STAC block, about lines 122-165)
- Modify: `docs/guides/workflows.md` (about lines 43-46)

- [ ] **Step 1: Rewrite the STAC example** in `docs/guides/architecture.md` to:

```python
from geosave_engine.geodata import GeoVector, stac, stack

forest = stac.create_collection("forest", description="Forest samples")
paths = image.gs.to_cog("samples/forest")                     # exactly the files written
items = stac.create_items(paths, collection=forest)            # one Item per scene
items = stac.create_items(image.gs.to_zarr("samples/forest.zarr"))   # one store, one Item

# Items are stored as one stac-geoparquet file, Collections in its metadata.
stac.table.write(items, "samples/catalog.parquet", collections=[forest])

rows = stac.table.read("samples/catalog.parquet", bbox=bounds).gs.query(aoi)
forest_pixels = stac.table.load(stac.table.to_items(rows))     # lazy Dataset
collections = stac.table.read_collections("samples/catalog.parquet")

# A new batch is concatenated like any vector and the table written again.
more = stac.table.from_items(new_items)
stac.table.write(
    GeoVector.concat([stac.table.read("samples/catalog.parquet"), more]),
    "samples/catalog.parquet",
    collections=collections.values(),
    overwrite=True,
)

# A stack is one Item per group; its rows share `geosave:stack`.
sample = stack({"optical": optical, "label": label})
items = stac.create_stack_items(sample.gs.to_cog("samples/s0"), name="s0")
```

In the prose below it, replace the sentence about `stac.asset.from_raster` with: an asset's fields are written by the modules of `stac.extensions`, one per schema, each through that schema's PySTAC class. Remove any sentence about adding to a catalog by writing another part file.

- [ ] **Step 2: Check** `docs/guides/workflows.md` lines 43-46 still read correctly; `stac.create_stack_items(paths, name="s1")` and `stac.table.write` are unchanged. Edit only if a sentence mentions part files or `table.load(rows)`.

- [ ] **Step 3: Lint and run the whole suite.**

Run: `uv run ruff check . && uv run pytest -q -m "not integration and not slow"`
Expected: no lint errors; all tests pass.

- [ ] **Step 4: Run the integration tests that need the network.**

Run: `uv run pytest tests/geodata/stac -q -m integration`
Expected: pass. `Item.validate()` on a stack Item fails on the `geosave` schema until the docs site serves `docs/schemas/`; if that is the only failure, report it and leave it.

- [ ] **Step 5: Confirm the layering.**

Run: `grep -rn "extra_fields\[\|properties\[" src/geosave_engine/geodata/stac src/geosave_engine/geodata/attrs/headers/stac.py`
Expected: no match. (`read_bands` reads `extra_fields.get("bands")`, the one keyed read.)

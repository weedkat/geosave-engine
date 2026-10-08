# STAC Extension Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Publish STAC 1.1 `bands` and extension fields from one module per schema, and read them back into the same attrs models.

**Architecture:** `stac/band.py` owns core band fields beside `asset.py` and `item.py`. `stac/extensions/<name>.py` each own one schema as plain functions listed in the `BAND` and `ASSET` tuples. `stac/header.py` (moved out of `attrs/`) decodes bands through the same modules.

**Tech Stack:** xarray, PySTAC 1.14.3 (plain `extra_fields`, no extension wrappers), stac-geoparquet 0.8.2, pydantic attrs models.

**Spec:** `docs/superpowers/specs/2026-10-08-stac-extension-modules-design.md`

## Global Constraints

- Publish `bands` only; read `bands`, `raster:bands`, `eo:bands`.
- Schema versions: projection v2.0.0, raster v2.0.0, eo v2.0.0, classification v2.0.0, datacube v2.2.0, cf v0.2.0.
- No registry, no plugin entry points, no compatibility aliases.
- No pixel is computed and no file is opened while describing a raster.
- `geodata` imports no torch; `attrs/` imports nothing from `stac/`.
- No commits: the working tree carries an unrelated refactor.
- Run tests with `uv run pytest <path> -q`; baseline is 243 passing in `tests/geodata/stac tests/geodata/attrs`.

## Review Focus

1. Parquet pads bands and nested classes with `null`; reading must give the bands that were written. (Task 3)
2. A legacy asset with more `eo:bands` than `raster:bands` still yields one band per index. (Task 3)
3. A `xr.decode_cf` raster describes the stored dtype and packing, not float values. (Task 3)
4. A `Legend` carrying `flag_masks` publishes no classes and declares no Classification. (Task 2)
5. A label Item declares no EO or Raster schema; an Item whose assets use different extensions declares their union. (Task 4)
6. A CRS without an EPSG code states WKT2 in both Projection and Datacube. (Task 4)

---

### Task 1: Native numbers and the `Spectral` model

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/model.py` (`from_attrs`)
- Create: `src/geosave_engine/geodata/attrs/models/spectral.py`
- Modify: `src/geosave_engine/geodata/attrs/models/__init__.py`, `src/geosave_engine/geodata/attrs/__init__.py`
- Test: `tests/geodata/attrs/test_models.py`, `tests/geodata/attrs/models/test_spectral.py`

**Interfaces:**
- Produces: `Spectral(common_name: str | None, center_wavelength: float | None, full_width_half_max: float | None)`, registered under `variable` as `"spectral"`, exported from `geodata.attrs`.

- [x] Failing tests:

```python
def test_numpy_scalars_read_as_native_numbers() -> None:
    nodata = Nodata.from_attrs({"_FillValue": np.uint16(0)})
    packing = Packing.from_attrs({"scale_factor": np.float32(1e-4)})
    assert type(nodata.fill_value) is int
    assert packing.scale_factor == 0.0001


def test_spectral_round_trips_through_attrs() -> None:
    spectral = Spectral(common_name="red", center_wavelength=0.665)
    assert spectral.to_attrs() == {"common_name": "red", "center_wavelength": 0.665}
    assert Spectral.from_attrs({**spectral.to_attrs(), "units": "1"}) == spectral
```

- [x] `from_attrs` converts `np.integer`/`np.bool_` with `.item()` before validation. The float32 shortening was dropped: writers store the widened value, so a described raster would disagree with its own file.
- [x] Add `Spectral`, register it, run `uv run pytest tests/geodata/attrs -q`.

### Task 2: Band extension modules

**Files:**
- Create: `src/geosave_engine/geodata/stac/extensions/{__init__,types,raster,eo,classification}.py`
- Test: `tests/geodata/stac/extensions/test_{raster,eo,classification}.py`

**Interfaces:**
- Produces: `types.BandFields`, `types.AssetFields`, `types.BandExtension`, `types.AssetExtension`; each module's `PREFIX`, `SCHEMA`, `encode_band(attrs: FlatAttrs) -> BandFields`, `decode_band(band: BandFields) -> <Model> | None`; `extensions.BAND = (raster, eo, classification)`.

- [x] Failing tests, one file per module:

```python
def test_packing_spells_raster_fields_both_ways() -> None:
    assert raster.encode_band({"scale_factor": 0.0001, "add_offset": 0.0}) == {
        "raster:scale": 0.0001, "raster:offset": 0.0}
    assert raster.encode_band({"units": "1"}) == {}
    assert raster.decode_band({"name": "red", "raster:scale": 0.0001}) == Packing(scale_factor=0.0001)
    assert raster.decode_band({"name": "red"}) is None


def test_spectral_spells_eo_fields_both_ways() -> None:
    fields = {"eo:common_name": "red", "eo:center_wavelength": 0.665}
    assert eo.encode_band({"common_name": "red", "center_wavelength": 0.665}) == fields
    assert eo.decode_band(fields) == Spectral(common_name="red", center_wavelength=0.665)
    assert eo.encode_band({}) == {} and eo.decode_band({"name": "dem"}) is None


def test_a_legend_spells_classes_both_ways() -> None:
    legend = Legend(class_map={0: "background", 1: "forest"}, color_map={1: "#00ff00"})
    fields = classification.encode_band(legend.to_attrs())
    assert fields == {"classification:classes": [
        {"value": 0, "name": "background"},
        {"value": 1, "name": "forest", "color_hint": "00FF00"}]}
    decoded = classification.decode_band(fields)
    assert decoded.class_map == {0: "background", 1: "forest"}
    assert parse_color(decoded.color_map[1]) == (0, 255, 0)


def test_a_bit_mask_legend_publishes_no_classes() -> None:
    masks = Legend(flag_masks=[1, 2], flag_meanings="cloud shadow")
    assert classification.encode_band(masks.to_attrs()) == {}


def test_classes_padded_by_parquet_still_decode() -> None:
    band = {"classification:classes": [{"value": 0, "name": "water", "color_hint": None}]}
    assert classification.decode_band(band).class_map == {0: "water"}
```

- [x] Implement the three modules and `types.py`; run `uv run pytest tests/geodata/stac/extensions -q`.

### Task 3: `stac/band.py`

**Files:**
- Create: `src/geosave_engine/geodata/stac/band.py`
- Test: `tests/geodata/stac/test_band.py`

**Interfaces:**
- Consumes: `extensions.BAND`.
- Produces: `from_variable(variable: xr.DataArray) -> BandFields`, `to_attrs(band: BandFields) -> FlatAttrs`, `read(asset: pystac.Asset) -> list[BandFields]`.

- [x] Failing tests:

```python
def test_a_variable_becomes_one_band() -> None:
    red = build_raster(packed=True).red
    red.attrs.update(units="1", long_name="Red reflectance")
    assert band.from_variable(red) == {
        "name": "red", "data_type": "uint16", "nodata": 0, "unit": "1",
        "description": "Red reflectance", "raster:scale": 0.0001, "raster:offset": 0.0}


def test_a_decoded_variable_states_what_a_writer_stores() -> None:
    raw = build_raster(packed=True)
    assert band.from_variable(xr.decode_cf(raw).red) == band.from_variable(raw.red)


def test_a_band_states_units_and_packing_as_attrs() -> None:
    fields = {"name": "red", "data_type": "uint16", "nodata": 0, "unit": "1",
              "description": "Red", "raster:scale": 0.0001}
    assert band.to_attrs(fields) == {"units": "1", "long_name": "Red", "scale_factor": 0.0001}


def test_legacy_listings_read_as_core_bands() -> None:
    asset = pystac.Asset("a.tif", extra_fields={
        "raster:bands": [{"data_type": "uint16", "nodata": 0, "scale": 0.0001}],
        "eo:bands": [{"name": "B04", "common_name": "red"}, {"name": "B08"}]})
    assert band.read(asset) == [
        {"data_type": "uint16", "nodata": 0, "raster:scale": 0.0001,
         "name": "B04", "eo:common_name": "red"},
        {"name": "B08"}]


def test_core_bands_read_without_parquet_nulls() -> None:
    asset = pystac.Asset("a.tif", extra_fields={
        "bands": [{"name": "label", "unit": None, "raster:scale": None}],
        "raster:bands": [{"unit": "ignored"}]})
    assert band.read(asset) == [{"name": "label"}]
```

- [x] Implement. `from_variable` reads `{**variable.attrs, **variable.encoding}` minus `None` encoding values; `nodata` and `data_type` are not decoded by `to_attrs`, because the loader that opens the file states them.
- [x] Run `uv run pytest tests/geodata/stac/test_band.py -q`.

### Task 4: Asset extension modules and `schemas`

**Files:**
- Create: `src/geosave_engine/geodata/stac/extensions/{projection,datacube,cf}.py`
- Modify: `src/geosave_engine/geodata/stac/extensions/__init__.py`
- Test: `tests/geodata/stac/extensions/test_{projection,datacube,cf,schemas}.py`

**Interfaces:**
- Produces: each module's `PREFIX`, `SCHEMA`, `encode_asset(raster: xr.Dataset) -> AssetFields`; `extensions.ASSET = (projection, datacube, cf)`; `extensions.schemas(item: pystac.Item) -> list[str]`.

- [x] Failing tests:

```python
def test_a_grid_becomes_projection_fields() -> None:
    raster = build_raster()
    assert projection.encode_asset(raster) == {
        "proj:code": "EPSG:32749", "proj:shape": [2, 2],
        "proj:transform": list(raster.gs.geobox.transform)[:6]}


def test_a_cube_states_its_dimensions_and_variables() -> None:
    fields = datacube.encode_asset(build_raster(times=2))
    assert fields["cube:dimensions"] == {
        "x": {"type": "spatial", "axis": "x", "extent": [500005.0, 500015.0],
              "step": 10.0, "reference_system": 32749},
        "y": {"type": "spatial", "axis": "y", "extent": [9000005.0, 9000015.0],
              "step": -10.0, "reference_system": 32749},
        "time": {"type": "temporal",
                 "extent": ["2025-06-01T00:00:00Z", "2025-06-02T00:00:00Z"]}}
    assert fields["cube:variables"]["red"] == {
        "dimensions": ["time", "y", "x"], "type": "data"}


def test_a_single_scene_is_no_cube() -> None:
    assert datacube.encode_asset(build_raster()) == {}


def test_standard_names_become_cf_parameters() -> None:
    raster = build_raster()
    raster.red.attrs.update(standard_name="surface_albedo", units="1")
    assert cf.encode_asset(raster) == {
        "cf:parameter": [{"name": "surface_albedo", "unit": "1"}]}
    assert cf.encode_asset(build_raster()) == {}


def test_only_used_schemas_are_declared() -> None:
    item = _item({"proj:code": "EPSG:4326",
                  "bands": [{"name": "label", "classification:classes": []}]})
    assert schemas(item) == [projection.SCHEMA, classification.SCHEMA]
```

Plus a laea-grid case asserting `proj:wkt2` and a string `reference_system`.

- [x] Implement; run `uv run pytest tests/geodata/stac/extensions -q`.

### Task 5: Rewire `asset.py` and `item.py`

**Files:**
- Modify: `src/geosave_engine/geodata/stac/asset.py`, `src/geosave_engine/geodata/stac/item.py`
- Test: `tests/geodata/stac/test_asset.py`, `tests/geodata/stac/test_item.py`

**Interfaces:**
- Consumes: `band.from_variable`, `band.read`, `extensions.ASSET`, `extensions.schemas`.
- Produces: unchanged signatures for `asset.from_raster`, `asset.default_key`, `item.from_raster`.

- [x] Rewrite the existing assertions on `raster:bands` / `eo:bands` to `bands` (same facts: names, dtype, nodata, unit, scale, zero values kept, native JSON numbers). Add:

```python
def test_a_label_item_declares_only_what_it_uses(label, tmp_path) -> None:
    labelled = label.gs.rebase(Legend(class_map={1: "forest"}), target="label")
    built = item.from_raster(labelled, {"label": tmp_path / "l.tif"}, id="l",
                             datetime=DateTime(2025, 1, 1, tzinfo=timezone.utc))
    assert built.stac_extensions == [projection.SCHEMA, classification.SCHEMA]
```

- [x] Run to see them fail, then replace `_describe`'s wrapper code and `_band_fields`, and the three `add_to` calls. Delete the PySTAC extension imports.
- [x] Run `uv run pytest tests/geodata/stac -q`.

### Task 6: Move the header to `stac/header.py`

**Files:**
- Move: `src/geosave_engine/geodata/attrs/headers/stac.py` -> `src/geosave_engine/geodata/stac/header.py`
- Move: `tests/geodata/attrs/headers/test_stac.py` -> `tests/geodata/stac/test_header.py`
- Modify: `src/geosave_engine/geodata/stac/source.py`

**Interfaces:**
- Consumes: `band.read`, `band.to_attrs`.
- Produces: `header.create_header(...)` and `header.read_asset_fields(asset, *, band_index=1)` with unchanged signatures.

- [x] Update the moved tests: import path; captured asset fields read `{"unit": "1", "raster:scale": 0.1}`; replace the import-isolation test with one asserting `geosave_engine.geodata.stac` is not imported by `geosave_engine.geodata.attrs`. Add:

```python
def test_spectral_facts_reach_the_variable() -> None:
    item = _item("scene", dt(2025, 1, 1), {})
    item.assets["red"].extra_fields = {
        "eo:bands": [{"name": "B04", "common_name": "red", "center_wavelength": 0.665}]}
    header = create_header([item], _collection(), xr.Dataset({"red": ("x", [1])}), groupby="id")
    assert header.data_vars["red"].get(Spectral) == Spectral(
        common_name="red", center_wavelength=0.665)
```

- [x] Replace the hand-built `band_attrs` with `band.to_attrs(band.read(asset)[band_index - 1])`; rebuild `read_asset_fields` on `band.read`.
- [x] Run `uv run pytest tests/geodata/stac tests/geodata/attrs -q`.

### Task 7: End-to-end round trip and checks

**Files:**
- Modify: `tests/geodata/stac/test_dataflow.py`
- Modify: docs under `docs/guides/` that name `raster:bands`, `eo:bands` or `attrs.headers.stac`

- [x] Add a round trip: optical + label raster -> `to_cog` -> `to_items` -> `table.write` -> `stac_table_to_items` -> `band.read` -> `band.to_attrs` equals the source units, packing, legend. Add an `integration`-marked test calling `Item.validate()` on an optical, a label and a cube Item.
- [x] Run `uv run pytest tests/geodata -q`, `uv run pytest -m integration tests/geodata/stac -q`, `uv run ruff check src tests`.

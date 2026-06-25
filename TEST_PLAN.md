# Test Plan — pipeline/io, datasets/source, supervised template

## Scope

Cover:
- `src/geosave_engine/geodata/pipeline/io.py`
- `src/geosave_engine/geodata/datasets/source.py`
- `src/templates/semantic_segmentation/methods/supervised/lightning/data_source.py`
- `src/templates/semantic_segmentation/methods/supervised/lightning/data_module.py`

## Pre-test fixes (applied)

- `_make_filename(layer, name)` — prefix with layer name, use `layer.centroid` (was bottom-left corner), keep `-<res>m` suffix.
- `save_tile` derives layer name from `output_dir.name`.
- `RasterLayer.filename_regex` widened to optionally match `-<res>m` suffix.

## Stale tests — delete + rewrite

- `tests/geosave_engine/geodata/pipeline/test_io.py`
- `tests/templates/semantic_segmentation/methods/supervised/test_data_module.py`

## New / rewritten files

```
tests/conftest.py                                                  (extend)
tests/geosave_engine/geodata/datasets/__init__.py                  (new, empty)
tests/geosave_engine/geodata/datasets/test_source.py               (new)
tests/geosave_engine/geodata/pipeline/test_io.py                   (rewrite)
tests/templates/.../supervised/test_data_source.py                 (new)
tests/templates/.../supervised/test_data_module.py                 (rewrite)
```

## Shared fixtures (tests/conftest.py)

```python
@pytest.fixture
def make_geo_layer():
    """bbox, resolution, dt, with_data, bands, fill, dtype → GeoLayer"""

@pytest.fixture
def dw_tif_path():
    """tests/data/dw_-22.7491991582_15.9791703445-20190223.tif"""
```

Keep existing `pytest_configure` (.env loading).

## Test contents

### test_io.py

- **TestComputeClassPct**
  - basic mapping with class_dict
  - single class
  - unknown value → `class_{u}` fallback
  - decimal rounding honored
- **TestManifestWriter**
  - creates xlsx on `create_layer`
  - layers sheet columns + row
  - `_meta` sheet written with flattened columns + `id` row
  - upsert by name
  - reserved name raises
  - description column
  - `write_tile` creates `<layer>` sheet
  - filename format (`<name>_<lon>_<lat>-<date>-<res>m.tif`)
  - standard cols: filename, datetime, lat, lon, crs, bbox (no `split`)
  - centroid lat/lon match expected
  - extra_meta merged
  - upsert by filename
  - appends different tiles
  - raises on missing declared layer
  - raises on partial result
  - warns on extra layer
  - rehydrates layers + records
  - remove_layer clears records
- **TestDeclareLayers**
  - batch declares from schema
  - upsert behavior on re-declare
- **TestManifestWriterGeocode**
  - mock `Place.from_coordinate`; record contains place fields when `geocode=True`, absent when `False`
- **TestSaveTile**
  - raises on no data
  - writes tif
  - filename contains name + date + res
  - creates output dir
- **TestProcessedTracker**
  - non-`.csv` suffix raises
  - `is_done` / `is_error` / `is_processed` semantics distinct
  - `mark_done`, `mark_error(msg)` persist message
  - persists across instances
  - multiple keys

### test_source.py

- **TestRasterSourceInit**
  - root stored
- **TestIngestFromDir** (fake `DummySource` subclass + synthetic anchor tifs via `save_tile`)
  - auto-declares `LAYER_SCHEMA` to writer
  - iterates `.tif` recursively
  - calls `ingest` per anchor
  - calls `writer.write_tile` with `extra_meta` result
  - calls `tracker.mark_done` on success
  - skips when `tracker.is_processed` True
  - calls `tracker.mark_error` on exception, continues loop
  - works with `writer=None`
  - works with `tracker=None`
- **TestNotImplemented**
  - `ingest_from_bbox` raises
  - `ingest_from_geojson` raises

### test_data_source.py

- **TestLabelSourceIngest** (uses `dw_tif_path`)
  - remap correctness (output values in {0..7, 255})
  - returns `{"dynamicworld": GeoLayer}`
- **TestLabelSourceExtraMeta**
  - returns `{"dynamicworld": {class_name: pct}}`
  - percentages sum ≈ 1.0
  - class names from `CLASS_NAMES`
- **TestLabelSourceAsDataset**
  - `FileNotFoundError` when root missing
  - returns `RasterLayer` when populated
- **TestLabelSourceSchema**
  - `LAYER_SCHEMA["dynamicworld"]` present; meta keys ⊇ {0..7, 255}
- **TestGeoSourceAsDataset**
  - `FileNotFoundError` when root missing
  - returns intersected dataset when populated
- **TestGeoSourceSchema**
  - `LAYER_SCHEMA` has `sentinel_2_l1c` + `cloud_mask`
  - imagery meta keys == `SEL_BANDS`
- **TestGeoSourceIngest** `@pytest.mark.integration`
  - real CDSE fetch via `dw_tif_path` anchor
  - asserts layer keys, shapes, cloud_mask dtype

### test_data_module.py

- **TestInit**
  - geo_sources/label_sources created for train/val/test
  - predict_source created
  - stride defaults to patch_size
  - explicit stride overrides
  - batch_size / num_workers stored
- **TestPrepareDataErrors**
  - unknown split key raises `ValueError`
  - bbox set → `NotImplementedError` propagates
  - geojson set → `NotImplementedError` propagates
- **TestPrepareDataPredictShortCircuit**
  - bbox/geojson set → `ingest_dirs` ignored (monkeypatch `predict_source.ingest_from_bbox` to no-op, assert `geo_sources` untouched)
- **TestPrepareDataHappyPath** `@pytest.mark.integration`
  - real CDSE ingest for `train` split (1 anchor via dw fixture)
  - asserts manifest + tifs on disk
- **TestSetup** (pre-populated `tmp_path` with synthetic tifs matching widened regex)
  - `setup("fit")` builds train+val datasets + samplers
  - `setup("validate")` builds val only
  - `setup("test")` builds test only
  - `setup("predict")` honors `predict_sampler` literal
  - `setup("predict")` invalid `predict_sampler` value raises
  - `setup("bad")` raises `ValueError`
  - missing split dirs → propagates `FileNotFoundError`
- **TestDataloaders**
  - each `*_dataloader()` returns `DataLoader`
  - train has `drop_last=True`
  - others have `drop_last=False`
  - batch_size forwarded

## Markers

- Unmarked: pure unit + local fixture tests
- `@pytest.mark.integration`: CDSE-hitting paths (GeoSource ingest, DataModule prepare_data happy path)

Run unit only (default): `pytest`
Run integration: `pytest -m integration`

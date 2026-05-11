from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
import xarray as xr

from geosave_engine.geodata.pipeline import (
    Anchor,
    BandMeta,
    BaseSource,
    ClassMeta,
    Derived,
    ManifestWriter,
    Pipeline,
    Source,
    SourceData,
    compute_class_pcts,
    save_tile,
)
from geosave_engine.geodata.pipeline.anchor import ANCHOR_CACHE_KEY
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import Sentinel2L2AQuery
from geosave_engine.utils.geodata import extract_raster_scale_offset, extract_stac_attrs

DATA_DIR = Path(__file__).parents[3] / "data"
ANCHOR_TIFF = DATA_DIR / "dw_-22.7491991582_15.9791703445-20190223.tif"


def _make_mock_source(name: str, source_data: SourceData | None) -> MagicMock:
    """Return a mocked BaseSource whose load() returns source_data."""
    source = MagicMock(spec=BaseSource)
    source.name = name
    source.load.return_value = source_data
    return source


def _make_source_data(bands: dict[str, np.ndarray] | None = None) -> SourceData:
    """Return a minimal SourceData with a 2x2 xr.Dataset."""
    if bands is None:
        bands = {"B04": np.ones((1, 2, 2), dtype="float32"), "B08": np.full((1, 2, 2), 2.0, dtype="float32")}
    ds = xr.Dataset(
        {k: xr.DataArray(v, dims=["time", "y", "x"]) for k, v in bands.items()}
    )
    item = pystac.Item(
        id="test-item",
        geometry=None,
        bbox=None,
        datetime=datetime(2019, 2, 23),
        properties={},
    )
    return SourceData(ds=ds, items=[item])


def compute_ndvi(caches: dict[str, SourceData]) -> xr.DataArray:
    ds = caches["s2"].ds
    return ((ds["B08"] - ds["B04"]) / (ds["B08"] + ds["B04"])).median(dim="time")


def _make_da(**extra_attrs: object) -> xr.DataArray:
    da = xr.DataArray(np.zeros((3, 3)))
    da.attrs["datetime"] = datetime(2019, 2, 23)
    da.attrs["bbox"] = (-22.749199, 15.979170, -22.5, 16.1)
    da.attrs.update(extra_attrs)
    return da


def _make_stac_item(
    item_id: str = "test-item",
    properties: dict | None = None,
    assets: dict | None = None,
) -> pystac.Item:
    item = pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2019, 2, 23),
        properties=properties or {},
    )
    for asset_key, asset_data in (assets or {}).items():
        item.add_asset(asset_key, pystac.Asset(href="https://example.com/", extra_fields=asset_data))
    return item


# ---------------------------------------------------------------------------
# extract_stac_attrs
# ---------------------------------------------------------------------------

class TestExtractStacAttrs:
    def test_extracts_nested_property(self):
        item = _make_stac_item(properties={"eo:cloud_cover": 12.3})
        result = extract_stac_attrs(item, {"cloud_cover": "properties.eo:cloud_cover"})
        assert result == {"cloud_cover": 12.3}

    def test_extracts_multiple_paths(self):
        item = _make_stac_item(properties={"eo:cloud_cover": 5.0, "sun_azimuth": 145.0})
        result = extract_stac_attrs(
            item,
            {"cloud_cover": "properties.eo:cloud_cover", "azimuth": "properties.sun_azimuth"},
        )
        assert result == {"cloud_cover": 5.0, "azimuth": 145.0}

    def test_raises_on_missing_key(self):
        item = _make_stac_item(properties={})
        with pytest.raises(KeyError, match="eo:cloud_cover"):
            extract_stac_attrs(item, {"cloud_cover": "properties.eo:cloud_cover"})

    def test_raises_on_missing_segment(self):
        item = _make_stac_item()
        with pytest.raises(KeyError, match="nonexistent"):
            extract_stac_attrs(item, {"x": "nonexistent.key"})


# ---------------------------------------------------------------------------
# extract_raster_scale_offset
# ---------------------------------------------------------------------------

class TestExtractRasterScaleOffset:
    def test_extracts_scale_and_offset(self):
        item = _make_stac_item(
            assets={"B04": {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}}
        )
        scale, offset = extract_raster_scale_offset(item)
        assert scale == pytest.approx(0.0001)
        assert offset == pytest.approx(-0.1)

    def test_raises_when_no_raster_bands(self):
        item = _make_stac_item(assets={"B04": {"type": "image/tiff"}})
        with pytest.raises(ValueError, match="scale/offset"):
            extract_raster_scale_offset(item)

    def test_raises_when_no_assets(self):
        item = _make_stac_item()
        with pytest.raises(ValueError, match="scale/offset"):
            extract_raster_scale_offset(item)


# ---------------------------------------------------------------------------
# compute_class_pcts
# ---------------------------------------------------------------------------

class TestComputeClassPcts:
    def test_basic_classes(self):
        da = xr.DataArray(np.array([[0, 1], [1, 2]]))
        result = compute_class_pcts(da)
        assert result == pytest.approx({"class_0": 0.25, "class_1": 0.5, "class_2": 0.25})

    def test_single_class(self):
        da = xr.DataArray(np.zeros((4, 4), dtype=int))
        result = compute_class_pcts(da)
        assert result == {"class_0": 1.0}

    def test_empty_array(self):
        da = xr.DataArray(np.array([]))
        result = compute_class_pcts(da)
        assert result == {}

    def test_keys_have_class_prefix(self):
        da = xr.DataArray(np.array([0, 255]))
        result = compute_class_pcts(da)
        assert "class_0" in result
        assert "class_255" in result


# ---------------------------------------------------------------------------
# ManifestWriter
# ---------------------------------------------------------------------------

class TestManifestWriter:
    def _make_writer(self, tmp_path: Path) -> tuple[ManifestWriter, Path]:
        writer = ManifestWriter(tmp_path)
        writer.create_layer("ndvi", role="image", meta=[BandMeta(name="B04")])
        return writer, tmp_path / "manifest.xlsx"

    def test_creates_xlsx_on_create_layer(self, tmp_path):
        _, path = self._make_writer(tmp_path)
        assert path.exists()

    def test_layers_sheet_written(self, tmp_path):
        _, path = self._make_writer(tmp_path)
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        assert "layers" in sheets

    def test_create_layer_image_num_bands(self, tmp_path):
        _, path = self._make_writer(tmp_path)
        df = pd.read_excel(path, sheet_name="layers", engine="openpyxl")
        assert int(df.loc[df["name"] == "ndvi", "num_bands"].iloc[0]) == 1

    def test_create_layer_label_num_class(self, tmp_path):
        writer = ManifestWriter(tmp_path)
        writer.create_layer(
            "label", role="label",
            meta=[ClassMeta(class_id=0, class_name="bg"), ClassMeta(class_id=1, class_name="tree")],
        )
        df = pd.read_excel(tmp_path / "manifest.xlsx", sheet_name="layers", engine="openpyxl")
        assert int(df.loc[df["name"] == "label", "num_class"].iloc[0]) == 2

    def test_create_layer_upsert(self, tmp_path):
        writer = ManifestWriter(tmp_path)
        writer.create_layer("label", role="label", meta=[ClassMeta(class_id=0, class_name="bg")])
        writer.create_layer("label", role="label", meta=[ClassMeta(class_id=0, class_name="bg"), ClassMeta(class_id=1, class_name="tree")])
        df = pd.read_excel(tmp_path / "manifest.xlsx", sheet_name="layers", engine="openpyxl")
        assert len(df) == 1
        assert int(df["num_class"].iloc[0]) == 2

    def test_create_layer_meta_sheet_written(self, tmp_path):
        _, path = self._make_writer(tmp_path)
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        assert "ndvi_meta" in sheets

    def test_create_layer_band_meta_columns(self, tmp_path):
        writer = ManifestWriter(tmp_path)
        writer.create_layer(
            "image", role="image",
            meta=[
                BandMeta(name="B04", wavelength=0.665, resolution=10.0),
                BandMeta(name="B08", wavelength=0.842, resolution=10.0),
            ],
        )
        df = pd.read_excel(tmp_path / "manifest.xlsx", sheet_name="image_meta", engine="openpyxl")
        assert list(df["name"]) == ["B04", "B08"]
        assert df["wavelength"].iloc[0] == pytest.approx(0.665)

    def test_create_layer_class_meta_columns(self, tmp_path):
        writer = ManifestWriter(tmp_path)
        writer.create_layer(
            "label", role="label",
            meta=[
                ClassMeta(class_id=0, class_name="water"),
                ClassMeta(class_id=1, class_name="trees", color="#3a9e3a"),
            ],
        )
        df = pd.read_excel(tmp_path / "manifest.xlsx", sheet_name="label_meta", engine="openpyxl")
        assert len(df) == 2
        assert df["class_name"].tolist() == ["water", "trees"]

    def test_create_layer_reserved_name_raises(self, tmp_path):
        writer = ManifestWriter(tmp_path)
        with pytest.raises(ValueError, match="reserved"):
            writer.create_layer("layers", role="image", meta=[BandMeta(name="B04")])

    def test_record_tile_creates_data_sheet(self, tmp_path):
        writer, path = self._make_writer(tmp_path)
        writer.record_tile({"ndvi": _make_da()})
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        assert "ndvi" in sheets

    def test_record_tile_row_has_filename(self, tmp_path):
        writer, path = self._make_writer(tmp_path)
        writer.record_tile({"ndvi": _make_da()})
        df = pd.read_excel(path, sheet_name="ndvi", engine="openpyxl")
        assert "filename" in df.columns
        assert df["filename"].iloc[0] == "-22.749199_15.979170-20190223.tif"

    def test_record_tile_da_attrs_as_columns(self, tmp_path):
        writer, path = self._make_writer(tmp_path)
        writer.record_tile({"ndvi": _make_da(cloud_cover=12.3)})
        df = pd.read_excel(path, sheet_name="ndvi", engine="openpyxl")
        assert "cloud_cover" in df.columns
        assert df["cloud_cover"].iloc[0] == pytest.approx(12.3)

    def test_record_tile_upserts_by_filename(self, tmp_path):
        writer, path = self._make_writer(tmp_path)
        writer.record_tile({"ndvi": _make_da(cloud_cover=10.0)})
        writer.record_tile({"ndvi": _make_da(cloud_cover=5.0)})
        df = pd.read_excel(path, sheet_name="ndvi", engine="openpyxl")
        assert len(df) == 1
        assert df["cloud_cover"].iloc[0] == pytest.approx(5.0)

    def test_record_tile_appends_new_filename(self, tmp_path):
        writer, path = self._make_writer(tmp_path)
        da1 = _make_da()
        da1.attrs["bbox"] = (-22.0, 15.0, -21.5, 15.5)
        writer.record_tile({"ndvi": da1})
        da2 = _make_da()
        da2.attrs["bbox"] = (-23.0, 16.0, -22.5, 16.5)
        writer.record_tile({"ndvi": da2})
        df = pd.read_excel(path, sheet_name="ndvi", engine="openpyxl")
        assert len(df) == 2

    def test_record_tile_nested_attr_json_stringified(self, tmp_path):
        import json
        writer, path = self._make_writer(tmp_path)
        writer.record_tile({"ndvi": _make_da(label_map={0: "bg", 1: "tree"})})
        df = pd.read_excel(path, sheet_name="ndvi", engine="openpyxl")
        assert json.loads(df["label_map"].iloc[0])["0"] == "bg"

    def test_record_tile_raises_on_undeclared_layer(self, tmp_path):
        writer = ManifestWriter(tmp_path)
        writer.create_layer("ndvi", role="image", meta=[BandMeta(name="B04")])
        with pytest.raises(ValueError, match="missing declared"):
            writer.record_tile({"ghost": _make_da()})

    def test_record_tile_raises_on_partial_result(self, tmp_path):
        writer = ManifestWriter(tmp_path)
        writer.create_layer("image", role="image", meta=[BandMeta(name="B04")])
        writer.create_layer("label", role="label", meta=[ClassMeta(class_id=0, class_name="bg")])
        with pytest.raises(ValueError, match="missing declared"):
            writer.record_tile({"image": _make_da()})

    def test_record_tile_warns_on_extra_layer(self, tmp_path):
        writer, _ = self._make_writer(tmp_path)
        with pytest.warns(UserWarning):
            writer.record_tile({"ndvi": _make_da(), "extra": _make_da()})

    def test_is_written_false_before_record(self, tmp_path):
        writer, _ = self._make_writer(tmp_path)
        assert not writer.is_written("-22.749199_15.979170-20190223.tif")

    def test_is_written_true_after_record(self, tmp_path):
        writer, _ = self._make_writer(tmp_path)
        writer.record_tile({"ndvi": _make_da()})
        assert writer.is_written("-22.749199_15.979170-20190223.tif")

    def test_rehydrates_declared_layers(self, tmp_path):
        w1 = ManifestWriter(tmp_path)
        w1.create_layer("ndvi", role="image", meta=[BandMeta(name="B04")])
        w2 = ManifestWriter(tmp_path)
        assert any(lyr.name == "ndvi" for lyr in w2.sheets.layers)

    def test_rehydrates_written_filenames(self, tmp_path):
        w1 = ManifestWriter(tmp_path)
        w1.create_layer("ndvi", role="image", meta=[BandMeta(name="B04")])
        w1.record_tile({"ndvi": _make_da()})
        w2 = ManifestWriter(tmp_path)
        assert w2.is_written("-22.749199_15.979170-20190223.tif")

    def test_remove_layer_clears_records_and_processed(self, tmp_path):
        writer, _ = self._make_writer(tmp_path)
        writer.record_tile({"ndvi": _make_da()})
        assert writer.is_written("-22.749199_15.979170-20190223.tif")
        writer.remove_layer("ndvi")
        assert not any(lyr.name == "ndvi" for lyr in writer.sheets.layers)
        assert "ndvi" not in writer.sheets.records
        assert not writer.is_written("-22.749199_15.979170-20190223.tif")


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------

class TestAnchor:
    def test_from_tiff_loads_metadata(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF)
        assert anchor.crs is not None
        assert anchor.width == 510
        assert anchor.height == 510
        assert anchor.datetime.year == 2019
        assert anchor.datetime.month == 2
        assert anchor.datetime.day == 23

    def test_bbox_is_wgs84_bounds(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF)
        lon_min, lat_min, lon_max, lat_max = anchor.bbox
        assert lon_min < lon_max
        assert lat_min < lat_max
        assert -180.0 <= lon_min <= 180.0
        assert -90.0 <= lat_min <= 90.0
        assert -23.0 < lon_min < -22.5
        assert 15.9 < lat_min < 16.1

    def test_from_bbox_creates_valid_anchor(self):
        bbox = (15.97, -22.75, 16.05, -22.65)
        anchor = Anchor.from_bbox(bbox, crs="EPSG:4326", resolution=0.0001, datetime=datetime(2019, 2, 23))
        assert anchor.width > 0
        assert anchor.height > 0
        assert anchor.datetime == datetime(2019, 2, 23)
        assert "EPSG:4326" in anchor.crs or "4326" in anchor.crs

    def test_from_bbox_label_is_none(self):
        bbox = (15.97, -22.75, 16.05, -22.65)
        anchor = Anchor.from_bbox(bbox, crs="EPSG:4326", resolution=0.0001, datetime=datetime(2019, 2, 23))
        assert anchor.label is None


# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------

class TestDerived:
    def test_cache_keys_single_string(self):
        d = Derived(sources="s2", compute_fn=lambda c: c["s2"].ds["B04"].isel(time=0), name="x")
        assert d.cache_keys() == ("s2",)

    def test_cache_keys_list(self):
        d = Derived(sources=["s2", "dem"], compute_fn=lambda c: c["s2"].ds["B04"].isel(time=0), name="x")
        assert d.cache_keys() == ("s2", "dem")

    def test_compute_passes_declared_keys_only(self):
        received: dict = {}

        def fn(cache: dict) -> xr.DataArray:
            received.update(cache)
            return xr.DataArray(np.zeros((2, 2)))

        sd = _make_source_data()
        d = Derived(sources="s2", compute_fn=fn, name="x")
        d.compute({"s2": sd, "other": "should_not_appear"})
        assert set(received.keys()) == {"s2"}

    def test_compute_raises_on_missing_key(self):
        d = Derived(sources="missing", compute_fn=lambda c: c["missing"], name="x")
        with pytest.raises(KeyError):
            d.compute({})

    def test_from_cache_factory(self):
        d = Derived.from_cache(
            sources="s2",
            compute_fn=lambda c: xr.DataArray(np.zeros((2, 2))),
            name="ndvi",
        )
        assert d.name == "ndvi"
        assert d.cache_keys() == ("s2",)

    def test_label_from_anchor_injects_anchor_key(self):
        d = Derived.label_from_anchor(name="label")
        assert ANCHOR_CACHE_KEY in d.cache_keys()
        assert len(d.cache_keys()) == 1

    def test_label_from_anchor_default_fn_returns_label(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF, load_label=True)
        d = Derived.label_from_anchor(name="label")
        result = d.compute({ANCHOR_CACHE_KEY: anchor})
        assert isinstance(result, xr.DataArray)

    def test_label_from_anchor_raises_when_label_none(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF, load_label=False)
        d = Derived.label_from_anchor(name="label")
        with pytest.raises(ValueError, match="label"):
            d.compute({ANCHOR_CACHE_KEY: anchor})

    def test_label_from_anchor_remap_applies_mapping(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF, load_label=True)
        unique_vals = set(int(v) for v in np.unique(anchor.label.values))
        remap = {v: i for i, v in enumerate(sorted(unique_vals))}
        d = Derived.label_from_anchor(name="label", remap=remap)
        result = d.compute({ANCHOR_CACHE_KEY: anchor})
        assert set(int(v) for v in np.unique(result.values)) == set(remap.values())

    def test_label_from_anchor_remap_raises_on_unmapped(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF, load_label=True)
        d = Derived.label_from_anchor(name="label", remap={999: 0})
        with pytest.raises(ValueError, match="Unmapped label values"):
            d.compute({ANCHOR_CACHE_KEY: anchor})

    def test_image_from_source_stacks_all_bands_by_default(self):
        sd = _make_source_data()
        d = Derived.image_from_source(name="image", sources="s2")
        result = d.compute({"s2": sd})
        assert isinstance(result, xr.DataArray)
        assert "band" in result.dims
        assert list(result.coords["band"].values) == ["B04", "B08"]

    def test_image_from_source_selects_subset_bands(self):
        sd = _make_source_data()
        d = Derived.image_from_source(name="image", sources="s2", bands=["B04"])
        result = d.compute({"s2": sd})
        assert list(result.coords["band"].values) == ["B04"]

    def test_image_from_source_reduce_median_drops_time(self):
        sd = _make_source_data()
        d = Derived.image_from_source(name="image", sources="s2", reduce="median")
        result = d.compute({"s2": sd})
        assert "time" not in result.dims
        assert result.dims == ("band", "y", "x")

    def test_image_from_source_reduce_none_keeps_time(self):
        sd = _make_source_data()
        d = Derived.image_from_source(name="image", sources="s2", reduce=None)
        result = d.compute({"s2": sd})
        assert result.dims == ("time", "band", "y", "x")

    def test_image_from_source_reduce_first(self):
        sd = _make_source_data()
        d = Derived.image_from_source(name="image", sources="s2", reduce="first")
        result = d.compute({"s2": sd})
        assert "time" not in result.dims

    def test_image_from_source_reduce_mean(self):
        sd = _make_source_data()
        d = Derived.image_from_source(name="image", sources="s2", reduce="mean")
        result = d.compute({"s2": sd})
        assert "time" not in result.dims


# ---------------------------------------------------------------------------
# Pipeline unit (mocked sources, no network)
# ---------------------------------------------------------------------------

class TestPipelineUnit:
    @pytest.fixture
    def anchor(self):
        return Anchor.from_tiff(ANCHOR_TIFF)

    def _ndvi_fn(self, cache: dict) -> xr.DataArray:
        ds = cache["s2"].ds
        return ((ds["B08"] - ds["B04"]) / (ds["B08"] + ds["B04"])).isel(time=0)

    def test_source_none_skips_derived(self, anchor):
        source = _make_mock_source("s2", None)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, [source], [derived]).run()
        assert "ndvi" not in result

    def test_run_computes_derived(self, anchor):
        sd = _make_source_data()
        source = _make_mock_source("s2", sd)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, [source], [derived]).run()
        assert "ndvi" in result
        assert isinstance(result["ndvi"], xr.DataArray)

    def test_run_tags_datetime_bbox(self, anchor):
        sd = _make_source_data()
        source = _make_mock_source("s2", sd)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, [source], [derived]).run()
        assert "datetime" in result["ndvi"].attrs
        assert "bbox" in result["ndvi"].attrs

    def test_run_tags_split(self, anchor):
        sd = _make_source_data()
        source = _make_mock_source("s2", sd)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, [source], [derived]).run(split="train")
        assert result["ndvi"].attrs["split"] == "train"

    def test_run_no_split_attr_when_omitted(self, anchor):
        sd = _make_source_data()
        source = _make_mock_source("s2", sd)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, [source], [derived]).run()
        assert "split" not in result["ndvi"].attrs

    def test_run_collects_stac_item_ids(self, anchor):
        sd = _make_source_data()
        source = _make_mock_source("s2", sd)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, [source], [derived]).run()
        ids = result["ndvi"].attrs["stac_item_ids"]
        assert len(ids) == 1
        assert isinstance(ids[0], str)

    def test_run_skips_derived_missing_source(self, anchor):
        source = _make_mock_source("s2", None)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        with pytest.warns(UserWarning):
            result = Pipeline(anchor, [source], [derived]).run()
        assert "ndvi" not in result

    def test_accepts_single_source_not_list(self, anchor):
        sd = _make_source_data()
        source = _make_mock_source("s2", sd)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, source, [derived]).run()
        assert "ndvi" in result

    def test_accepts_single_derived_not_list(self, anchor):
        sd = _make_source_data()
        source = _make_mock_source("s2", sd)
        derived = Derived(sources="s2", compute_fn=self._ndvi_fn, name="ndvi")
        result = Pipeline(anchor, [source], derived).run()
        assert "ndvi" in result

    def test_anchor_derived_uses_label(self, anchor):
        label_derived = Derived.label_from_anchor(name="label")
        result = Pipeline(anchor, [], label_derived).run()
        assert "label" in result


# ---------------------------------------------------------------------------
# Pipeline validation
# ---------------------------------------------------------------------------

class TestPipelineValidation:
    def test_raises_on_zero_anchors(self):
        with pytest.raises(TypeError):
            Pipeline()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# save_layer validation
# ---------------------------------------------------------------------------

class TestSaveTileValidation:
    def test_raises_on_missing_datetime_attr(self, tmp_path):
        da = xr.DataArray([1.0, 2.0])
        da.attrs["bbox"] = (0.0, 0.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="datetime"):
            save_tile(da, tmp_path)

    def test_raises_on_missing_bbox_attr(self, tmp_path):
        da = xr.DataArray([1.0, 2.0])
        da.attrs["datetime"] = datetime(2019, 2, 23)
        with pytest.raises(ValueError, match="bbox"):
            save_tile(da, tmp_path)


# ---------------------------------------------------------------------------
# Integration: full pipeline with manifest
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end: real CDSE STAC search + pixel download + save + manifest.

    Requires tests/.env with CDSE credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.).
    Run with: pytest -m integration
    """

    @pytest.fixture(scope="class")
    def ingested(self, tmp_path_factory):
        output_dir = tmp_path_factory.mktemp("pipeline_output")
        manifest_path = output_dir / "manifest.xlsx"
        anchor = Anchor.from_tiff(ANCHOR_TIFF)
        query = Sentinel2L2AQuery().max_cloud_cover(30)
        source = Source.sentinel_2_l2a(
            name="s2",
            client=StacClient.cdse(),
            query=query,
            time_range=timedelta(days=30),
            bands=["B04", "B08"],
        )
        ndvi = Derived(
            sources="s2",
            compute_fn=compute_ndvi,
            name="ndvi",
        )
        manifest = ManifestWriter(output_dir)
        manifest.create_layer("ndvi", role="image", meta=[BandMeta(name="B04"), BandMeta(name="B08")])
        result = Pipeline(anchor, [source], [ndvi]).run()
        manifest.write_tile(result)
        return result, output_dir, output_dir / "manifest.xlsx"

    def test_result_contains_ndvi_layer(self, ingested):
        result, _, _ = ingested
        assert "ndvi" in result
        assert isinstance(result["ndvi"], xr.DataArray)

    def test_result_has_pipeline_attrs(self, ingested):
        result, _, _ = ingested
        da = result["ndvi"]
        assert "datetime" in da.attrs
        assert "bbox" in da.attrs

    def test_result_has_stac_item_ids(self, ingested):
        result, _, _ = ingested
        da = result["ndvi"]
        assert "stac_item_ids" in da.attrs
        assert len(da.attrs["stac_item_ids"]) > 0
        assert isinstance(da.attrs["stac_item_ids"][0], str)

    def test_saved_file_exists(self, ingested):
        _, output_dir, _ = ingested
        tif_files = list((output_dir / "ndvi").glob("*.tif"))
        assert len(tif_files) == 1

    def test_saved_file_is_valid_geotiff(self, ingested):
        _, output_dir, _ = ingested
        tif_files = list((output_dir / "ndvi").glob("*.tif"))
        with rasterio.open(tif_files[0]) as src:
            assert src.count >= 1
            assert src.crs is not None
            assert src.width > 0
            assert src.height > 0

    def test_saved_file_has_valid_ndvi_range(self, ingested):
        _, output_dir, _ = ingested
        tif_files = list((output_dir / "ndvi").glob("*.tif"))
        with rasterio.open(tif_files[0]) as src:
            data = src.read(1)
            valid = data[data != src.nodata] if src.nodata is not None else data.flatten()
            assert float(valid.min()) >= -1.0
            assert float(valid.max()) <= 1.0

    def test_manifest_exists(self, ingested):
        _, _, manifest_path = ingested
        assert manifest_path.exists()

    def test_manifest_has_ndvi_sheet(self, ingested):
        _, _, manifest_path = ingested
        sheets = pd.read_excel(manifest_path, sheet_name=None, engine="openpyxl")
        assert "ndvi" in sheets

    def test_manifest_row_has_correct_filename(self, ingested):
        result, output_dir, manifest_path = ingested
        da = result["ndvi"]
        bbox = da.attrs["bbox"]
        dt: datetime = da.attrs["datetime"]
        expected_filename = f"{bbox[0]:.6f}_{bbox[1]:.6f}-{dt.strftime('%Y%m%d')}.tif"
        df = pd.read_excel(manifest_path, sheet_name="ndvi", engine="openpyxl")
        assert expected_filename in df["filename"].values

    def test_manifest_has_datetime_column(self, ingested):
        _, _, manifest_path = ingested
        df = pd.read_excel(manifest_path, sheet_name="ndvi", engine="openpyxl")
        assert "datetime" in df.columns

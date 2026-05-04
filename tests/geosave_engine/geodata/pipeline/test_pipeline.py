from datetime import timedelta
from pathlib import Path

import pytest
import rasterio
import xarray as xr

from geosave_engine.geodata.pipeline import Anchor, Derived, Pipeline, Source, SourceData, save_layer
from geosave_engine.geodata.pipeline.source.base import SourceData
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import Sentinel2L2AQuery

DATA_DIR = Path(__file__).parents[3] / "data"
ANCHOR_TIFF = DATA_DIR / "dw_-22.7491991582_15.9791703445-20190223.tif"


def compute_ndvi(caches: dict[str, SourceData]) -> xr.DataArray:
    ds = caches["s2"].ds
    return ((ds["B08"] - ds["B04"]) / (ds["B08"] + ds["B04"])).median(dim="time")


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
        # WGS84 range check
        assert -180.0 <= lon_min <= 180.0
        assert -90.0 <= lat_min <= 90.0
        # Area is West Africa: ~22.7°W, ~16°N
        assert -23.0 < lon_min < -22.5
        assert 15.9 < lat_min < 16.1


class TestPipelineValidation:
    def test_raises_on_zero_anchors(self):
        with pytest.raises(ValueError, match="exactly one Anchor"):
            Pipeline()

    def test_raises_on_two_anchors(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF)
        with pytest.raises(ValueError, match="exactly one Anchor"):
            Pipeline(anchor, anchor)

    def test_raises_on_missing_cache_key(self):
        anchor = Anchor.from_tiff(ANCHOR_TIFF)
        derived = Derived(
            need_caches={"nonexistent": ["B04"]},
            compute_fn=compute_ndvi,
            layer_name="ndvi",
        )
        # No source loaded, so "nonexistent" not in cache
        with pytest.raises(KeyError, match="nonexistent"):
            Pipeline(anchor, derived).run()


class TestSaveLayerValidation:
    def test_raises_on_missing_datetime_attr(self, tmp_path):
        da = xr.DataArray([1.0, 2.0])
        da.attrs["bbox"] = (0.0, 0.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="datetime"):
            save_layer({"layer": da}, tmp_path)

    def test_raises_on_missing_bbox_attr(self, tmp_path):
        from datetime import datetime
        da = xr.DataArray([1.0, 2.0])
        da.attrs["datetime"] = datetime(2019, 2, 23)
        with pytest.raises(ValueError, match="bbox"):
            save_layer({"layer": da}, tmp_path)


@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end: real CDSE STAC search + pixel download + save.

    Requires tests/.env with CDSE credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.).
    Run with: pytest -m integration
    """

    @pytest.fixture(scope="class")
    def ingested(self, tmp_path_factory):
        output_dir = tmp_path_factory.mktemp("pipeline_output")
        anchor = Anchor.from_tiff(ANCHOR_TIFF)
        query = Sentinel2L2AQuery().max_cloud_cover(30)
        source = Source.sentinel_2_l2a(
            layer_name="s2",
            client=StacClient.cdse(),
            query=query,
            time_range=timedelta(days=30),
            bands=["B04", "B08"],
        )
        ndvi = Derived(
            need_caches={"s2": ["B04", "B08"]},
            compute_fn=compute_ndvi,
            layer_name="ndvi",
        )
        result = Pipeline(anchor, source, ndvi).run()
        save_layer(result, output_dir)
        return result, output_dir

    def test_result_contains_ndvi_layer(self, ingested):
        result, _ = ingested
        assert "ndvi" in result
        assert isinstance(result["ndvi"], xr.DataArray)

    def test_result_has_pipeline_attrs(self, ingested):
        result, _ = ingested
        da = result["ndvi"]
        assert "datetime" in da.attrs
        assert "bbox" in da.attrs

    def test_saved_file_exists(self, ingested):
        _, output_dir = ingested
        ndvi_dir = output_dir / "ndvi"
        assert ndvi_dir.exists()
        tif_files = list(ndvi_dir.glob("*.tif"))
        assert len(tif_files) == 1

    def test_saved_file_is_valid_geotiff(self, ingested):
        _, output_dir = ingested
        tif_files = list((output_dir / "ndvi").glob("*.tif"))
        with rasterio.open(tif_files[0]) as src:
            assert src.count >= 1
            assert src.crs is not None
            assert src.width > 0
            assert src.height > 0

    def test_saved_file_has_valid_ndvi_range(self, ingested):
        _, output_dir = ingested
        tif_files = list((output_dir / "ndvi").glob("*.tif"))
        with rasterio.open(tif_files[0]) as src:
            data = src.read(1)
            valid = data[data != src.nodata] if src.nodata is not None else data.flatten()
            assert float(valid.min()) >= -1.0
            assert float(valid.max()) <= 1.0

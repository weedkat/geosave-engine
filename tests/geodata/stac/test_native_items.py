"""Native STAC Items integrate with the GeoDataFrame accessor."""

from datetime import UTC, datetime
from pathlib import Path

import pystac
import pytest
from shapely.geometry import mapping

from geosave_engine.geodata import GeoVector, io
from tests.geodata.conftest import build_raster


def _item(tmp_path, name="scene"):
    source = build_raster(times=1).isel(time=0)
    saved = io.geotiff.write_cog(source, tmp_path / f"{name}.tif")
    shape = source.gs.geobox.extent.to_crs("EPSG:4326").geom
    item = pystac.Item(
        name,
        mapping(shape),
        list(shape.bounds),
        datetime(2025, 6, 1, tzinfo=UTC),
        {"eo:cloud_cover": 12.5, "provider:quality": "good"},
    )
    item.add_asset(
        "reflectance",
        pystac.Asset(str(saved), media_type=pystac.MediaType.COG, roles=["data"]),
    )
    item.add_asset(
        "thumbnail",
        pystac.Asset(
            "https://example.com/thumb.jpg",
            media_type=pystac.MediaType.JPEG,
            roles=["thumbnail"],
        ),
    )
    return item, source


def test_native_item_insertion_does_not_require_xarray_metadata(tmp_path):
    entry, _ = _item(tmp_path)
    frame = GeoVector.from_items([entry])
    updated = frame.gs.upsert(entry, on="id")
    assert updated.id.tolist() == [entry.id]
    assert updated.iloc[0]["provider:quality"] == "good"
    assert "raster_metadata" not in frame
    assert not any(name.startswith("geosave:") for name in frame.columns)
    assert (
        frame.iloc[0].assets["reflectance"]["href"] == entry.assets["reflectance"].href
    )


def test_relative_assets_resolve_without_mutating_the_item(tmp_path):
    entry, _ = _item(tmp_path)
    entry.assets["reflectance"].href = "scene.tif"
    entry.set_self_href(str(tmp_path / "scene.json"))
    frame = GeoVector.from_items([entry])
    assert entry.assets["reflectance"].href == "scene.tif"
    assert frame.iloc[0].assets["reflectance"]["href"] == str(tmp_path / "scene.tif")


def test_explicit_stac_serializer_retains_item_and_asset_properties(tmp_path):
    import pyarrow.parquet as pq
    from stac_geoparquet.arrow import stac_table_to_items
    from pandas.testing import assert_frame_equal

    first, source = _item(tmp_path, "first")
    second, _ = _item(tmp_path, "second")
    bands = [
        {"name": name, "data_type": source[name].dtype.name}
        for name in source.gs.variables
    ]
    first.assets["reflectance"].extra_fields["bands"] = bands
    second.properties["new:property"] = 7
    second.assets["other"] = second.assets.pop("reflectance")
    frame = GeoVector.from_items([first, second]).iloc[::-1]
    original = frame.copy(deep=True)
    saved = frame.gs.to_geoparquet(tmp_path / "items.parquet")
    assert_frame_equal(frame, original)
    schema = pq.read_schema(saved)
    assert b"stac-geoparquet" in schema.metadata
    restored = list(stac_table_to_items(pq.read_table(saved)))
    assert [entry["id"] for entry in restored] == ["second", "first"]
    assert restored[0]["properties"]["new:property"] == 7
    assert restored[1]["assets"]["reflectance"]["bands"] == bands


def test_a_stac_table_keeps_its_bbox_through_read_and_rewrite(tmp_path):
    import pyarrow.parquet as pq
    import pytest
    from stac_geoparquet.arrow import stac_table_to_items

    from geosave_engine.geodata import read_vector

    entry, _ = _item(tmp_path)
    first = GeoVector.from_items([entry]).gs.to_geoparquet(tmp_path / "catalog.parquet")
    second = read_vector(first).gs.to_geoparquet(tmp_path / "again.parquet")

    for path in (first, second):
        (restored,) = stac_table_to_items(pq.read_table(path))
        assert restored["bbox"] == pytest.approx(entry.bbox)


def test_an_edited_geometry_changes_the_bbox_written(tmp_path):
    import pyarrow.parquet as pq
    from stac_geoparquet.arrow import stac_table_to_items

    entry, _ = _item(tmp_path)
    frame = GeoVector.from_items([entry])
    frame = frame.set_geometry(frame.geometry.translate(xoff=1.0))

    (restored,) = stac_table_to_items(
        pq.read_table(frame.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    )

    assert restored["bbox"][0] == pytest.approx(entry.bbox[0] + 1.0)


@pytest.mark.parametrize("folder", ["dataset", "data set"])
def test_a_stac_table_follows_its_assets_when_moved(tmp_path, folder):
    import shutil

    import pyarrow.parquet as pq

    from geosave_engine.geodata import read_vector

    home = tmp_path / folder
    home.mkdir()
    entry, _ = _item(home)
    table = GeoVector.from_items([entry]).gs.to_geoparquet(home / "catalog.parquet")
    stored = pq.read_table(table).column("assets")[0].as_py()
    assert stored["reflectance"]["href"] == "./scene.tif"
    assert stored["thumbnail"]["href"] == "https://example.com/thumb.jpg"

    moved = tmp_path / "moved"
    shutil.move(home, moved)

    href = read_vector(moved / "catalog.parquet").iloc[0].assets["reflectance"]["href"]
    assert href == str(moved / "scene.tif")
    assert Path(href).is_file()

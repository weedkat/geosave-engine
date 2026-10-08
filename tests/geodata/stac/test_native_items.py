"""Native STAC Items from other producers fit the item table."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pystac
import pytest
from shapely.geometry import mapping
from stac_geoparquet.arrow import stac_table_to_items

from geosave_engine.geodata import io
from geosave_engine.geodata.stac import table
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


def test_a_native_item_needs_no_geosave_fields(tmp_path):
    entry, _ = _item(tmp_path)

    rows = table.from_items([entry])

    assert rows.id.tolist() == [entry.id]
    assert rows.iloc[0]["provider:quality"] == "good"
    assert not any(name.startswith("geosave:") for name in rows.columns)
    assert (
        rows.iloc[0].assets["reflectance"]["href"] == entry.assets["reflectance"].href
    )


def test_relative_assets_resolve_without_mutating_the_item(tmp_path):
    entry, _ = _item(tmp_path)
    entry.assets["reflectance"].href = "scene.tif"
    entry.set_self_href(str(tmp_path / "scene.json"))

    rows = table.from_items([entry])

    assert entry.assets["reflectance"].href == "scene.tif"
    assert rows.iloc[0].assets["reflectance"]["href"] == str(tmp_path / "scene.tif")


def test_a_written_table_retains_item_and_asset_properties(tmp_path):
    first, source = _item(tmp_path, "first")
    second, _ = _item(tmp_path, "second")
    bands = [
        {"name": name, "data_type": source[name].dtype.name}
        for name in source.gs.variables
    ]
    first.assets["reflectance"].extra_fields["bands"] = bands
    second.properties["new:property"] = 7
    second.assets["other"] = second.assets.pop("reflectance")

    saved = table.write([second, first], tmp_path / "items.parquet")

    restored = list(stac_table_to_items(pq.read_table(saved)))
    assert [entry["id"] for entry in restored] == ["second", "first"]
    assert restored[0]["properties"]["new:property"] == 7
    assert restored[1]["assets"]["reflectance"]["bands"] == bands


def test_only_data_assets_load(tmp_path):
    entry, source = _item(tmp_path)

    loaded = table.load([entry])

    assert list(loaded.data_vars) == list(source.data_vars)
    loaded.close()


def test_a_table_keeps_its_bbox_through_read_and_rewrite(tmp_path):
    entry, _ = _item(tmp_path)
    first = table.write([entry], tmp_path / "catalog.parquet")
    second = table.write(table.read(first), tmp_path / "again.parquet")

    for path in (first, second):
        (restored,) = stac_table_to_items(pq.read_table(path))
        assert restored["bbox"] == pytest.approx(entry.bbox)


def test_an_edited_geometry_changes_the_bbox_written(tmp_path):
    entry, _ = _item(tmp_path)
    rows = table.from_items([entry])
    rows = rows.set_geometry(rows.geometry.translate(xoff=1.0))

    (restored,) = stac_table_to_items(
        pq.read_table(table.write(rows, tmp_path / "catalog.parquet"))
    )

    assert restored["bbox"][0] == pytest.approx(entry.bbox[0] + 1.0)


@pytest.mark.parametrize("folder", ["dataset", "data set"])
def test_a_table_follows_its_assets_when_moved(tmp_path, folder):
    home = tmp_path / folder
    home.mkdir()
    entry, _ = _item(home)
    saved = table.write([entry], home / "catalog.parquet")
    stored = pq.read_table(saved).column("assets")[0].as_py()
    assert stored["reflectance"]["href"] == "./scene.tif"
    assert stored["thumbnail"]["href"] == "https://example.com/thumb.jpg"

    moved = tmp_path / "moved"
    shutil.move(home, moved)

    href = table.read(moved / "catalog.parquet").iloc[0].assets["reflectance"]["href"]
    assert href == str(moved / "scene.tif")
    assert Path(href).is_file()

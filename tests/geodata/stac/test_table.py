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


def test_a_table_that_is_not_stac_is_returned_as_read(tmp_path) -> None:
    from shapely.geometry import Point

    plain = GeoVector.from_geometry(
        Point(0, 0),
        properties={"id": "a", "assets": {"label": {"href": "a.tif"}}, "quality": 0.97},
    )
    path = plain.gs.to_geoparquet(tmp_path / "labels.parquet")

    rows = table.read(path)

    assert rows.loc[0, "quality"] == 0.97
    assert rows.loc[0, "assets"]["label"]["href"] == "a.tif"

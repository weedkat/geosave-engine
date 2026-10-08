"""Items built from the rasters a writer saved."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import dask
import numpy as np
import pytest
from odc.geo.geobox import GeoBox
from pystac.extensions.classification import ClassificationExtension
from pystac.extensions.datacube import DatacubeExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension

from geosave_engine.geodata import io, raster, stac, stack
from geosave_engine.geodata.attrs import Legend
from geosave_engine.geodata.stac import table
from geosave_engine.geodata.stac.extensions import GeosaveExtension, ZarrExtension

from tests.geodata.conftest import build_raster

IDS = ["forest_20250601T000000", "forest_20250602T000000"]


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


@pytest.mark.integration
def test_items_validate_against_the_stac_schemas(tmp_path: Path) -> None:
    source = build_raster(times=2)
    for built in stac.create_items(source.gs.to_cog(tmp_path / "forest")):
        built.validate()


def test_a_footprint_is_built_for_a_grid_without_an_epsg_code(tmp_path: Path) -> None:
    crs = "+proj=laea +lat_0=12.34 +lon_0=56.78 +datum=WGS84 +units=m"
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs=crs, resolution=10)
    label = raster(
        {"class": (("y", "x"), np.ones((4, 4), dtype="uint8"))}, grid
    ).assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    path = io.geotiff.write_cog(label, tmp_path / "laea.tif")

    built = stac.create_item({"label": path}, id="laea")

    assert built.bbox[0] == pytest.approx(56.78, abs=0.01)
    assert built.bbox[1] == pytest.approx(12.34, abs=0.01)


def test_items_survive_the_table(tmp_path: Path) -> None:
    source = build_raster(times=2)
    items = stac.create_items(source.gs.to_cog(tmp_path / "forest", split_bands=True))

    saved = table.write(items, tmp_path / "catalog.parquet")

    restored = table.to_items(table.read(saved))
    assert [entry.id for entry in restored] == IDS
    assert sorted(restored[0].assets) == ["nir", "red"]
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].data_type == "uint16"


def test_a_store_is_one_item_with_no_collection(scene, tmp_path):
    path = scene.gs.to_zarr(tmp_path / "forest.zarr")

    (built,) = stac.create_items(path)

    assert built.id == "forest"
    assert built.collection_id is None
    assert list(built.assets) == ["image"]
    assert built.assets["image"].href == str(path)
    assert built.properties["start_datetime"] == "2025-06-01T00:00:00Z"


def test_one_file_per_scene_is_one_item_per_instant(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest")

    built = stac.create_items(paths)

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert [list(each.assets) for each in built] == [["image"], ["image"]]
    assert built[1].assets["image"].href == str(paths[1])
    assert built[1].datetime == datetime(2025, 6, 11, tzinfo=UTC)
    assert [each.datetime.isoformat() for each in built] == [
        "2025-06-01T00:00:00+00:00",
        "2025-06-11T00:00:00+00:00",
    ]
    assert all("start_datetime" not in each.properties for each in built)


def test_split_bands_become_one_asset_per_variable(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest", split_bands=True)

    built = stac.create_items(paths)

    assert [each.id for each in built] == [
        "forest_20250601T000000",
        "forest_20250611T000000",
    ]
    assert list(built[0].assets) == ["red", "nir"]
    assert built[1].assets["nir"].href == str(paths[3])


def test_items_take_the_id_of_their_collection_without_a_link(scene, tmp_path):
    forest = stac.create_collection("forest", description="Forest samples")

    built = stac.create_items(scene.gs.to_cog(tmp_path / "forest"), collection=forest)

    assert [each.collection_id for each in built] == ["forest", "forest"]
    assert [each.links for each in built] == [[], []]


def test_two_files_claiming_one_asset_key_refuse(scene, tmp_path):
    first = io.geotiff.write_cog(scene.isel(time=0), tmp_path / "a.tif")
    second = io.geotiff.write_cog(scene.isel(time=0), tmp_path / "b.tif")

    with pytest.raises(ValueError, match="same time"):
        stac.create_items([first, second])


def test_an_item_states_what_the_file_stores(tmp_path):
    source = build_raster(times=2)[["red"]]
    path = source.gs.to_netcdf(tmp_path / "small.nc", encoding={"red": {"dtype": "u1"}})

    (built,) = stac.create_items(path)

    assert source.red.dtype == "uint16"
    assert RasterExtension.ext(built.assets["red"]).bands[0].data_type == "uint8"


def test_no_paths_refuse():
    with pytest.raises(ValueError, match="paths a writer returned"):
        stac.create_items([])


def test_an_id_template_is_filled_per_scene(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest")

    built = stac.create_items(paths, id="{lat:.2f}N_{start:%Y%m%d}")

    assert [each.id for each in built] == ["45.13N_20250601", "45.13N_20250611"]
    assert built[0].assets["image"].href == str(paths[0])


def test_an_id_template_that_repeats_refuses(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest")

    with pytest.raises(ValueError, match="same id"):
        stac.create_items(paths, id="{lat:.2f}N")


def test_building_items_computes_no_pixels(scene, tmp_path):
    paths = scene.gs.to_cog(tmp_path / "forest")

    with dask.config.set(scheduler=_refuse):
        built = stac.create_items(paths, chunks={})

    assert len(built) == 2


def test_a_timeless_raster_needs_a_time(label, tmp_path):
    (path,) = label.gs.to_cog(tmp_path / "label")

    with pytest.raises(ValueError, match="pass datetime="):
        stac.create_items(path)

    (built,) = stac.create_items(path, datetime=datetime(2025, 6, 1, tzinfo=UTC))
    assert built.datetime == datetime(2025, 6, 1, tzinfo=UTC)


def test_the_caller_names_the_assets_of_one_item(scene, label, tmp_path):
    optical = scene.gs.to_zarr(tmp_path / "optical.zarr")
    (classes,) = label.gs.to_cog(tmp_path / "label")

    built = stac.create_item({"optical": optical, "label": classes}, id="s0")

    assert built.id == "s0"
    assert list(built.assets) == ["optical", "label"]
    assert built.assets["label"].href == str(classes)
    # The timeless label rides on the span its dated sibling covers.
    assert built.properties["start_datetime"] == "2025-06-01T00:00:00Z"


def test_a_stack_of_cogs_is_one_item_per_group_scene(scene, label, tmp_path):
    tree = stack({"optical": scene, "label": label})
    paths = tree.gs.to_cog(tmp_path / "s0")

    built = stac.create_stack_items(paths, name="s0")

    assert [each.id for each in built] == [
        "s0/optical_20250601T000000",
        "s0/optical_20250611T000000",
        "s0/label",
    ]
    assert [each.collection_id for each in built] == ["optical", "optical", "label"]
    assert {GeosaveExtension.ext(each).stack for each in built} == {"s0"}
    # The timeless label takes the span its dated sibling covers.
    assert built[2].properties["start_datetime"] == "2025-06-01T00:00:00Z"


def test_a_stack_of_stores_is_one_item_per_group(scene, label, tmp_path):
    tree = stack({"optical": scene, "label": label})
    paths = tree.gs.to_zarr(tmp_path / "s0")

    optical, classes = stac.create_stack_items(paths, name="s0")

    assert (optical.id, classes.id) == ("s0/optical", "s0/label")
    assert (optical.collection_id, classes.collection_id) == ("optical", "label")
    assert optical.assets["image"].href == str(tmp_path / "s0/optical.zarr")
    assert "xarray:open_kwargs" not in optical.assets["image"].extra_fields
    assert classes.assets["label"].href == str(tmp_path / "s0/label.zarr")
    assert classes.properties["start_datetime"] == "2025-06-01T00:00:00Z"


def test_a_timeless_stack_needs_a_time(label, tmp_path):
    tree = stack({"label": label})
    paths = tree.gs.to_cog(tmp_path / "s0")

    with pytest.raises(ValueError, match="pass datetime="):
        stac.create_stack_items(paths, name="s0")


def test_a_label_item_declares_only_what_it_uses(label, tmp_path: Path) -> None:
    classes = label.gs.rebase(Legend(class_map={1: "forest"}), target="label")
    (path,) = classes.gs.to_cog(tmp_path / "label")

    built = stac.create_item(
        {"label": path}, id="label", datetime=datetime(2025, 1, 1, tzinfo=UTC)
    )

    assert set(built.stac_extensions) == {
        ProjectionExtension.get_schema_uri(),
        RasterExtension.get_schema_uri(),
        ClassificationExtension.get_schema_uri(),
    }


def test_a_plain_scene_declares_its_grid_bands_and_time(scene, tmp_path: Path) -> None:
    (path, _) = scene.gs.to_cog(tmp_path / "scene")

    built = stac.create_item({"image": path}, id="scene")

    assert built.stac_extensions == [
        ProjectionExtension.get_schema_uri(),
        RasterExtension.get_schema_uri(),
        DatacubeExtension.get_schema_uri(),
    ]


def test_a_collection_starts_with_an_open_extent() -> None:
    forest = stac.create_collection(
        "forest", description="Forest samples", license="CC-BY-4.0"
    )

    assert (forest.id, forest.license) == ("forest", "CC-BY-4.0")
    assert forest.extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]
    assert forest.extent.temporal.intervals == [[None, None]]


def test_every_extension_module_writes_onto_each_asset(scene, tmp_path) -> None:
    (store,) = stac.create_items(scene.gs.to_zarr(tmp_path / "cube.zarr"))

    asset = store.assets["image"]
    assert ProjectionExtension.ext(asset).code == "EPSG:32633"
    assert len(RasterExtension.ext(asset).bands) == 2
    assert ZarrExtension.ext(asset).node_type == "group"
    assert asset.media_type == "application/vnd+zarr"
    assert asset.roles == ["data"]

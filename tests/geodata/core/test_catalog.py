"""A catalog built from the Items a raster describes itself with."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave_engine.geodata import GeoVector, read_raster, read_vector

from tests.geodata.conftest import build_raster


def test_an_unsaved_raster_is_saved_as_cogs_and_lists_one_row_per_scene(
    tmp_path: Path,
) -> None:
    # Catalogs key assets by band, so indexing splits bands unless told not to.
    items = build_raster(times=2).gs.to_items(tmp_path / "forest")

    catalog = GeoVector.from_items(items)

    assert catalog["id"].tolist() == [
        "forest_20250601T000000",
        "forest_20250602T000000",
    ]
    assert catalog.crs == "EPSG:4326"
    assert sorted(catalog.iloc[0]["assets"]) == ["nir", "red"]
    red = tmp_path / "forest/forest_20250601T000000/red.tif"
    assert catalog.iloc[0]["assets"]["red"]["href"] == str(red)


def test_a_name_with_a_dot_stays_whole_in_the_scene_id(tmp_path: Path) -> None:
    for split_bands in (False, True):
        items = build_raster(times=2).gs.to_items(
            tmp_path / str(split_bands) / "scene.v2", split_bands=split_bands
        )
        assert items[0].id == "scene.v2_20250601T000000"
        assert items[0].collection_id == "scene.v2"


def test_each_scene_is_dated_and_bounded(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    items = build_raster(times=2).gs.to_items(tmp_path / "forest")

    assert [item.datetime for item in items] == [
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 6, 2, tzinfo=UTC),
    ]
    assert all(len(item.bbox) == 4 for item in items)


def test_rows_of_one_raster_share_its_name_as_their_collection(tmp_path: Path) -> None:
    cube = build_raster(times=2)

    scenes = GeoVector.from_items(cube.gs.to_items(tmp_path / "forest"))
    lone = GeoVector.from_items(cube.isel(time=[0]).gs.to_items(tmp_path / "water"))
    store = GeoVector.from_items(
        cube.gs.to_items(tmp_path / "copy.zarr", driver="zarr")
    )

    assert scenes["collection"].tolist() == ["forest", "forest"]
    assert lone["collection"].tolist() == ["water"]
    assert store["collection"].tolist() == ["copy"]


@pytest.mark.parametrize(
    ("driver", "name"), [("zarr", "forest.zarr"), ("netcdf", "forest.nc")]
)
def test_a_store_driver_saves_one_store_as_one_row(tmp_path, driver, name) -> None:
    (item,) = build_raster(times=2).gs.to_items(tmp_path / name, driver=driver)

    assert item.id == "forest"
    assert item.assets["image"].href == str(tmp_path / name)


def test_a_saved_raster_describes_itself_from_where_it_was_read(tmp_path: Path) -> None:
    store = build_raster(times=2).gs.to_zarr(tmp_path / "forest.zarr")

    with read_raster(store) as saved:
        (item,) = saved.gs.to_items()

    assert item.id == "forest"
    assert item.collection_id == "forest"
    assert item.assets["image"].href == str(store)


def test_a_raster_read_from_a_folder_of_cogs_is_indexed_when_written(tmp_path) -> None:
    build_raster(times=2).gs.to_cog(tmp_path / "scenes")

    with read_raster(tmp_path / "scenes") as saved:
        with pytest.raises(ValueError, match="index it when writing"):
            saved.gs.to_items()


def test_a_collection_is_named_by_the_caller_or_by_the_path(tmp_path: Path) -> None:
    cube = build_raster(times=1)

    (named,) = cube.gs.to_items(tmp_path / "a", collection="sentinel-2")
    (pathed,) = cube.gs.to_items(tmp_path / "b")

    assert named.collection_id == "sentinel-2"
    assert pathed.collection_id == "b"


def test_a_scalar_instant_raster_is_one_scene_named_by_the_path(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    (item,) = build_raster(times=1).isel(time=0).gs.to_items(tmp_path / "label")

    assert (item.id, item.datetime) == ("label", datetime(2025, 6, 1, tzinfo=UTC))
    assert sorted(item.assets) == ["nir", "red"]


def test_one_file_per_scene_is_keyed_as_image(tmp_path: Path) -> None:
    items = build_raster(times=2).gs.to_items(tmp_path / "forest", split_bands=False)

    assert [sorted(item.assets) for item in items] == [["image"], ["image"]]
    assert items[0].assets["image"].href == str(
        tmp_path / "forest/forest_20250601T000000.tif"
    )


def test_a_timeless_raster_refuses_and_names_the_way_out(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="datetime="):
        build_raster().gs.to_items(tmp_path / "dem")


def test_writer_options_reach_the_writer(tmp_path: Path) -> None:
    import rasterio

    (item,) = build_raster(times=1).gs.to_items(tmp_path / "forest", compress="ZSTD")

    with rasterio.open(item.assets["red"].href) as src:
        assert src.compression.name == "zstd"


def test_an_unsaved_raster_without_a_path_refuses(tmp_path: Path) -> None:
    store = build_raster(times=2).gs.to_zarr(tmp_path / "forest.zarr")

    with pytest.raises(ValueError, match="not saved"):
        build_raster(times=2).gs.to_items()
    # Pixels changed since the read are no longer what the store holds.
    with read_raster(store) as saved, pytest.raises(ValueError, match="not saved"):
        saved.gs.to_nan().gs.to_items()


def test_a_band_indexes_as_the_raster_it_converts_to(tmp_path: Path) -> None:
    source = build_raster(times=2)

    items = source["red"].gs.to_items(tmp_path / "red", collection="forest")

    assert [item.id for item in items] == ["red_20250601T000000", "red_20250602T000000"]
    assert [sorted(item.assets) for item in items] == [["red"], ["red"]]
    assert {item.collection_id for item in items} == {"forest"}
    restored = GeoVector.from_items(items).gs.to_raster()
    np.testing.assert_array_equal(restored.red, source.red)


def test_rasters_register_into_one_catalog(tmp_path: Path) -> None:
    cube = build_raster(times=1)
    forest = GeoVector.from_items(cube.gs.to_items(tmp_path / "forest"))
    water = GeoVector.from_items(
        cube.gs.to_items(tmp_path / "water.zarr", driver="zarr")
    )

    catalog = forest.gs.upsert(water, on="id")

    assert sorted(catalog["id"]) == ["forest_20250601T000000", "water"]


def _catalog(tmp_path: Path, source: xr.Dataset, **options) -> Path:
    items = source.gs.to_items(tmp_path / "forest", **options)
    return GeoVector.from_items(items).gs.to_geoparquet(tmp_path / "catalog.parquet")


@pytest.mark.parametrize("split_bands", [False, True])
def test_a_registered_raster_reads_back_through_a_query(tmp_path, split_bands) -> None:
    source = build_raster(times=2, packed=True)
    table = _catalog(tmp_path, source, split_bands=split_bands)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        restored = read_vector(table).gs.query(source).gs.to_raster()

    assert started == []
    assert restored.red.chunks is not None
    with read_raster(tmp_path / "forest", chunks={}) as expected:
        xr.testing.assert_identical(restored, expected)
    np.testing.assert_array_equal(restored.red, source.red)


def test_a_one_date_target_selects_one_scene(tmp_path: Path) -> None:
    source = build_raster(times=2)
    catalog = read_vector(_catalog(tmp_path, source))

    restored = catalog.gs.query(source.isel(time=[0])).gs.to_raster()

    np.testing.assert_array_equal(restored.time, source.time[:1])


def test_a_store_row_reads_back(tmp_path: Path) -> None:
    source = build_raster(times=2)
    store = tmp_path / "forest.zarr"
    items = source.gs.to_items(store, driver="zarr")
    table = GeoVector.from_items(items).gs.to_geoparquet(tmp_path / "catalog.parquet")

    with read_raster(store, chunks={}) as expected:
        xr.testing.assert_identical(read_vector(table).gs.to_raster(), expected)


def test_an_empty_selection_refuses(tmp_path: Path) -> None:
    catalog = read_vector(_catalog(tmp_path, build_raster(times=1)))

    with pytest.raises(ValueError, match="at least one source"):
        catalog.iloc[0:0].gs.to_raster()


def test_one_raster_registered_in_two_formats_refuses_to_read_as_one(tmp_path) -> None:
    source = build_raster(times=2)
    scenes = GeoVector.from_items(source.gs.to_items(tmp_path / "forest"))
    store = GeoVector.from_items(
        source.gs.to_items(tmp_path / "copy.zarr", driver="zarr")
    )

    with pytest.raises(ValueError, match="one variable at one instant"):
        GeoVector.concat([scenes, store]).gs.to_raster()


def test_a_written_catalog_loads_in_odc_stac(tmp_path: Path) -> None:
    import odc.stac
    import stac_geoparquet

    source = build_raster(times=2)
    catalog = read_vector(_catalog(tmp_path, source, split_bands=True))

    loaded = odc.stac.load(list(stac_geoparquet.to_item_collection(catalog)), chunks={})

    assert set(loaded.data_vars) == {"red", "nir"}
    assert loaded.odc.geobox == source.odc.geobox
    np.testing.assert_array_equal(loaded.time, source.time)
    np.testing.assert_array_equal(loaded.red, source.red)


def test_the_loop_runs_on_a_bucket(bucket: str) -> None:
    source = build_raster(times=2)
    store = f"{bucket}/samples/forest.zarr"

    catalog = GeoVector.from_items(source.gs.to_items(store, driver="zarr"))
    table = catalog.gs.to_geoparquet(f"{bucket}/samples/catalog.parquet")
    restored = read_vector(table).gs.query(source).gs.to_raster()

    assert catalog["id"].tolist() == ["forest"]
    assert catalog.iloc[0]["assets"]["image"]["href"] == store
    np.testing.assert_array_equal(restored.red, source.red)
    np.testing.assert_array_equal(restored.time, source.time)


def test_remote_assets_keep_urls_in_a_local_table(bucket: str, tmp_path: Path) -> None:
    import fsspec
    import pyarrow.parquet as pq

    store = f"{bucket}/forest.zarr"
    catalog = GeoVector.from_items(
        build_raster(times=2).gs.to_items(store, driver="zarr")
    )

    elsewhere = catalog.gs.to_geoparquet(tmp_path / "catalog.parquet")
    catalog.gs.to_geoparquet(f"{bucket}/catalog.parquet")
    beside = pq.read_table(
        f"{bucket}/catalog.parquet".removeprefix("memory:/"),
        filesystem=fsspec.filesystem("memory"),
    )

    # A table on another filesystem keeps the URL; one beside the assets shortens it.
    stored = pq.read_table(elsewhere).column("assets")[0].as_py()
    assert stored["image"]["href"] == store
    assert beside.column("assets")[0].as_py()["image"]["href"] == "./forest.zarr"


@pytest.fixture
def hf_bucket():
    """Yield a unique prefix in the Hugging Face test bucket and empty it afterwards."""
    import uuid

    import fsspec
    from huggingface_hub import get_token

    if get_token() is None:
        pytest.skip("HF_TOKEN or `hf auth login` is required")
    prefix = f"hf://buckets/fatmur/test/geosave-tests/{uuid.uuid4()}"
    filesystem, path = fsspec.core.url_to_fs(prefix)
    yield prefix
    if filesystem.exists(path):
        filesystem.rm(path, recursive=True)


@pytest.mark.integration
def test_a_zarr_raster_round_trips_through_a_hugging_face_bucket(
    hf_bucket: str,
) -> None:
    source = build_raster(times=2)
    items = source.gs.to_items(f"{hf_bucket}/forest.zarr", driver="zarr")

    table = GeoVector.from_items(items).gs.to_geoparquet(f"{hf_bucket}/catalog.parquet")
    restored = read_vector(table).gs.to_raster()

    np.testing.assert_array_equal(restored.red, source.red)


@pytest.mark.integration
def test_cogs_round_trip_through_a_hugging_face_bucket(hf_bucket: str) -> None:
    import os

    from geosave_engine.geodata import configure_gdal

    if "AWS_ACCESS_KEY_ID" not in os.environ:
        pytest.skip("S3 credentials generated from the HF token are required")
    # The gateway serves no ListObjectsV1, which GDAL's open-time listing uses.
    configure_gdal(
        aws_s3_endpoint="s3.hf.co",
        aws_virtual_hosting=False,
        aws_default_region="us-east-1",
        gdal_disable_readdir_on_open=True,
    )
    source = build_raster(times=2, packed=True)

    items = source.gs.to_items(f"{hf_bucket}/forest", split_bands=True)
    table = GeoVector.from_items(items).gs.to_geoparquet(f"{hf_bucket}/catalog.parquet")
    restored = read_vector(table).gs.query(source).gs.to_raster()

    assert sum(len(item.assets) for item in items) == 4
    np.testing.assert_array_equal(restored.red, source.red)
    np.testing.assert_array_equal(restored.time, source.time)


def test_a_local_asset_keeps_its_path_in_a_remote_table(bucket: str, tmp_path) -> None:
    items = build_raster(times=2).gs.to_items(tmp_path / "forest.zarr", driver="zarr")

    table = GeoVector.from_items(items).gs.to_geoparquet(f"{bucket}/catalog.parquet")

    href = read_vector(table).iloc[0]["assets"]["image"]["href"]
    assert href == str(tmp_path / "forest.zarr")


def test_an_item_table_is_written_as_stac_geoparquet(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from geosave_engine.geodata import GeoVector as Vector

    table = _catalog(tmp_path, build_raster(times=1))
    plain = Vector.from_geometry("POINT (13 52)").gs.to_geoparquet(
        tmp_path / "plain.parquet"
    )

    assert b"stac-geoparquet" in pq.read_schema(table).metadata
    assert b"stac-geoparquet" not in pq.read_schema(plain).metadata

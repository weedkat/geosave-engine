"""Raster writers publish matching GeoVector records only after writing pixels."""

import shutil

from dask import delayed
from dask.delayed import Delayed
import numpy as np
import pyarrow.parquet as pq
import pytest

from geosave_engine.geodata import (
    GeoVector,
    read_raster,
    read_stack,
    read_vector,
    stack,
)
from tests.geodata.conftest import build_raster


@pytest.mark.parametrize("entry", ["from_assets", "from_xarray", "writer"])
def test_dataset_registration_names_metadata_after_the_saved_asset(tmp_path, entry):
    source = build_raster(times=1).isel(time=0)
    saved = source.gs.to_cog(tmp_path / "optical.tif")
    if entry == "from_assets":
        row = GeoVector.from_assets(saved).iloc[0]
    elif entry == "from_xarray":
        with read_raster(saved) as opened:
            row = GeoVector.from_xarray(opened).iloc[0]
    else:
        source.gs.to_cog(saved, catalog=tmp_path / "catalog.parquet", overwrite=True)
        row = read_vector(tmp_path / "catalog.parquet").iloc[0]
    assert set(row.assets) == set(row.raster_metadata) == {"optical"}
    assert list(row.raster_metadata["optical"]["bands"]) == ["red", "nir"]
    with row.gs.to_xarray() as reopened:
        np.testing.assert_array_equal(reopened["optical"].red.values, source.red.values)


@pytest.mark.parametrize("entry", ["from_assets", "from_xarray", "writer"])
def test_registration_reads_saved_metadata_once(tmp_path, monkeypatch, entry):
    from dask.callbacks import Callback
    from geosave_engine.geodata import io

    source = build_raster(times=1).isel(time=0)
    saved = source.gs.to_cog(tmp_path / "image.tif")
    opened = read_raster(saved)
    reads = []
    pixel_tasks = []
    read = io.read_raster

    def observe(path, **options):
        reads.append(path)
        return read(path, **options)

    monkeypatch.setattr(io, "read_raster", observe)
    try:
        with Callback(pretask=lambda key, *_: pixel_tasks.append(key)):
            if entry == "from_assets":
                GeoVector.from_assets(saved)
            elif entry == "from_xarray":
                GeoVector.from_xarray(opened)
            else:
                io.assets.write_catalog(saved, tmp_path / "catalog.parquet")
        assert len(reads) == 1
        assert pixel_tasks == []
    finally:
        opened.close()


@pytest.mark.parametrize("writer", ["to_cog", "to_zarr"])
def test_dataset_writer_returns_path_and_registers_changed_pixels(tmp_path, writer):
    original = build_raster(times=1).isel(time=0)
    source = original.gs.to_zarr(tmp_path / "source.zarr")
    changed = read_raster(source, chunks="auto")
    changed = changed.assign(red=changed.red + 7)
    destination = tmp_path / ("image.tif" if writer == "to_cog" else "image.zarr")
    saved = getattr(changed.gs, writer)(
        destination, catalog=tmp_path / "catalog.parquet", id="image"
    )

    assert saved == destination
    row = read_vector(tmp_path / "catalog.parquet").iloc[0]
    assert row.id == "image"
    restored = row.gs.to_xarray().gs.rasters["image"]
    np.testing.assert_array_equal(restored.red.values, original.red.values + 7)
    assert restored.red.chunks is not None
    assert restored.gs.geobox == original.gs.geobox
    stored = pq.read_table(tmp_path / "catalog.parquet").column("assets")[0].as_py()
    assert stored["image"]["href"] == destination.name


@pytest.mark.parametrize("writer", ["to_cog", "to_zarr"])
def test_stack_catalog_and_assets_can_move_together(tmp_path, writer):
    optical = build_raster(times=2)
    sample = stack(
        {"optical": optical, "label": optical[["red"]].isel(time=0, drop=True)}
    )
    root = tmp_path / "original"
    root.mkdir()
    destination = root / ("sample" if writer == "to_cog" else "sample.zarr")
    options = {"split_bands": True} if writer == "to_cog" else {}
    saved = getattr(sample.gs, writer)(
        destination, catalog=root / "catalog.parquet", id="sample", **options
    )
    assert saved == destination
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    row = read_vector(moved / "catalog.parquet").iloc[0]
    restored = row.gs.to_xarray()

    for name, raster in sample.gs.rasters.items():
        actual = restored.gs.rasters[name]
        assert actual.gs.geobox == raster.gs.geobox
        for variable in raster.data_vars:
            np.testing.assert_array_equal(
                actual[variable].values, raster[variable].values
            )
            assert actual[variable].chunks is not None
    np.testing.assert_array_equal(restored["optical"].time.values, optical.time.values)
    if writer == "to_zarr":
        assert row.assets["optical"]["group"] == "optical"
        assert row.assets["label"]["group"] == "label"
        assert row.assets["optical"]["href"] == row.assets["label"]["href"]


def test_read_stack_can_register_one_zarr_store(tmp_path):
    optical = build_raster(times=2)
    source = stack(
        {"optical": optical, "label": optical[["red"]].isel(time=0, drop=True)}
    )
    saved = source.gs.to_zarr(tmp_path / "sample.zarr")

    record = GeoVector.from_xarray(read_stack(saved), id="sample")
    row = record.iloc[0]
    assert row.assets["optical"]["group"] == "optical"
    np.testing.assert_array_equal(
        row.gs.to_xarray()["optical"].red.values, optical.red.values
    )


def test_zarr_group_pointer_selects_only_that_group(tmp_path):
    source = build_raster(times=2)
    store = stack({"optical": source, "other": source[["red"]]}).gs.to_zarr(
        tmp_path / "sample.zarr"
    )
    row = GeoVector.from_assets(
        {"image": {"href": store, "group": "optical"}}, id="sample"
    ).iloc[0]
    restored = row.gs.to_xarray()
    assert restored.gs.groups == ("image",)
    np.testing.assert_array_equal(restored["image"].time.values, source.time.values)


@pytest.mark.parametrize("as_stack", [False, True])
def test_deferred_zarr_write_publishes_catalog_after_compute(tmp_path, as_stack):
    source = build_raster(times=2).chunk()
    data = stack({"image": source}) if as_stack else source
    catalog = tmp_path / "catalog.parquet"
    saved = data.gs.to_zarr(
        tmp_path / "image.zarr", catalog=catalog, id="image", compute=False
    )
    assert isinstance(saved, Delayed)
    assert not catalog.exists()
    assert saved.compute() == tmp_path / "image.zarr"
    np.testing.assert_array_equal(
        read_vector(catalog).iloc[0].gs.to_xarray()["image"].time.values,
        source.time.values,
    )


def test_failed_pixel_write_does_not_publish_catalog(tmp_path):
    import dask.array as da

    @delayed
    def fail():
        raise RuntimeError("pixel read failed")

    source = build_raster(times=1)
    source["red"].data = da.from_delayed(fail(), shape=(1, 2, 2), dtype="uint16")
    catalog = tmp_path / "catalog.parquet"
    with pytest.raises(RuntimeError, match="pixel read failed"):
        source.gs.to_zarr(tmp_path / "image.zarr", catalog=catalog)
    assert not catalog.exists()


def test_companion_catalog_does_not_append_to_existing_records(tmp_path):
    source = build_raster(times=1)
    catalog = tmp_path / "catalog.parquet"
    source.gs.to_zarr(tmp_path / "first.zarr", catalog=catalog, id="first")
    with pytest.raises(FileExistsError):
        source.gs.to_zarr(tmp_path / "second.zarr", catalog=catalog, id="second")
    assert read_vector(catalog).id.tolist() == ["first"]


def test_dataarray_cog_can_publish_its_asset_record(tmp_path):
    source = build_raster(times=1).isel(time=0).red
    saved = source.gs.to_cog(
        tmp_path / "red.tif", catalog=tmp_path / "catalog.parquet", id="red"
    )
    assert saved == tmp_path / "red.tif"
    np.testing.assert_array_equal(
        read_vector(tmp_path / "catalog.parquet")
        .iloc[0]
        .gs.to_xarray()["red"]
        .red.values,
        source.values,
    )


def test_a_local_file_uri_registers_and_reopens_the_same_raster(tmp_path):
    source = build_raster(times=1).isel(time=0)
    saved = source.gs.to_cog(tmp_path / "image.tif")
    row = GeoVector.from_assets({"image": {"href": saved.as_uri()}}).iloc[0]
    assert row.assets["image"]["href"] == str(saved)
    np.testing.assert_array_equal(
        row.gs.to_xarray()["image"].red.values, source.red.values
    )


def test_deferred_failure_does_not_publish_a_catalog(tmp_path):
    import dask.array as da

    @delayed
    def fail():
        raise RuntimeError("pixel read failed")

    source = build_raster(times=1)
    source["red"].data = da.from_delayed(fail(), shape=(1, 2, 2), dtype="uint16")
    catalog = tmp_path / "catalog.parquet"
    saved = source.gs.to_zarr(tmp_path / "image.zarr", catalog=catalog, compute=False)
    with pytest.raises(RuntimeError, match="pixel read failed"):
        saved.compute()
    assert not catalog.exists()


def test_overwrite_replaces_the_catalog_explicitly(tmp_path):
    source = build_raster(times=1)
    catalog = tmp_path / "catalog.parquet"
    source.gs.to_zarr(tmp_path / "image.zarr", catalog=catalog, id="before")
    changed = source.assign(red=source.red + 9)
    changed.gs.to_zarr(
        tmp_path / "image.zarr", catalog=catalog, id="after", overwrite=True
    )
    row = read_vector(catalog).iloc[0]
    assert row.id == "after"
    np.testing.assert_array_equal(
        row.gs.to_xarray()["image"].red.values, changed.red.values
    )

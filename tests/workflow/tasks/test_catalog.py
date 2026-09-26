from pathlib import Path

from dask.callbacks import Callback
import numpy as np
from prefect.cache_policies import NO_CACHE

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.tasks.catalog import save_catalog
from geosave_engine.workflow.tasks.save import write_stack


def write_sample(path, raw, *, day):
    optical = raw["optical"][["red"]]
    label = raster(
        {"class": np.full((4, 4), day, dtype="uint8")}, optical.gs.geobox
    ).assign_coords(time=np.datetime64(f"2025-01-{day:02d}"))
    write_stack({"label": label, "optical": optical}, path)
    return str(path)


def test_save_catalog_registers_completed_samples(tmp_path, raw):
    samples = tmp_path / "prepared" / "samples"
    paths = {
        "north/a": write_sample(samples / "north" / "a.zarr", raw, day=1),
        "south/b": write_sample(samples / "south" / "b.zarr", raw, day=2),
    }
    destination = tmp_path / "prepared" / "manifest.parquet"

    result = save_catalog.fn(paths, destination)

    assert result == str(destination)
    catalog = io.read_vector(destination)
    assert catalog.gdf.sample_id.tolist() == ["north/a", "south/b"]
    assert [Path(path) for path in catalog.gdf.path] == [
        Path(path).resolve() for path in paths.values()
    ]
    assert catalog.gdf.geometry.is_valid.all()
    assert catalog.gdf.start_datetime.notna().all()
    assert catalog.gdf.end_datetime.notna().all()
    assert set(
        ["grid_crs", "grid_transform", "grid_height", "grid_width"]
    ) <= set(catalog.gdf)
    assert set(catalog.gdf.iloc[0].variables) == {
        "label/class",
        "optical/red",
    }


def test_save_catalog_does_not_compute_sample_pixels(tmp_path, raw):
    sample = write_sample(tmp_path / "samples" / "a.zarr", raw, day=1)
    started = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        save_catalog.fn({"a": sample}, tmp_path / "manifest.parquet")

    assert started == []


def test_save_catalog_replaces_rows_atomically(tmp_path, raw):
    samples = tmp_path / "samples"
    first = write_sample(samples / "a.zarr", raw, day=1)
    second = write_sample(samples / "b.zarr", raw, day=2)
    destination = tmp_path / "manifest.parquet"
    save_catalog.fn({"a": first, "b": second}, destination)

    save_catalog.fn({"b": second}, destination)

    assert io.read_vector(destination).gdf.sample_id.tolist() == ["b"]
    assert not (tmp_path / ".manifest.staging.parquet").exists()


def test_save_catalog_is_not_cached_or_persisted():
    assert save_catalog.cache_policy is NO_CACHE
    assert save_catalog.persist_result is False

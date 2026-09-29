from pathlib import Path
from typing import Literal

from dask.callbacks import Callback
import geopandas as gpd
import numpy as np
from odc.geo import CRS
from odc.geo.geobox import GeoBox
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.training_data import SampleFormat
from geosave_engine.workflow.training_data.manifest import write_manifest
from geosave_engine.workflow.training_data.sample import write_sample


def write_dense_sample(
    path, raw, *, day, format: SampleFormat = "zarr"
):
    optical = raw["optical"][["red"]]
    label = raster(
        {"class": np.full((4, 4), day, dtype="uint8")}, optical.gs.geobox
    ).assign_coords(time=np.datetime64(f"2025-01-{day:02d}"))
    write_sample({"label": label, "optical": optical}, path, format=format)
    return str(path)


@pytest.mark.parametrize("format", ["geotiff", "zarr"])
def test_write_manifest_registers_completed_samples(
    tmp_path, raw, format: Literal["geotiff", "zarr"]
):
    samples = tmp_path / "prepared"
    suffix = ".zarr" if format == "zarr" else ""
    paths = {
        "north/a": write_dense_sample(
            samples / "north" / f"a{suffix}", raw, day=1, format=format
        ),
        "south/b": write_dense_sample(
            samples / "south" / f"b{suffix}", raw, day=2, format=format
        ),
    }
    destination = tmp_path / "prepared" / "manifest.parquet"

    result = write_manifest(paths, destination, format=format)

    assert result == str(destination)
    stored = gpd.read_parquet(destination)
    assert list(stored) == [
        "path",
        "format",
        "start_datetime",
        "end_datetime",
        "grid_crs",
        "grid_height",
        "grid_width",
        "geometry",
    ]
    assert stored.path.tolist() == [
        f"north/a{suffix}",
        f"south/b{suffix}",
    ]
    catalog = io.read_vector(destination)
    assert [Path(path) for path in catalog.gdf.path] == [
        Path(path).resolve() for path in paths.values()
    ]
    assert catalog.gdf.geometry.is_valid.all()
    assert catalog.gdf.start_datetime.notna().all()
    assert catalog.gdf.end_datetime.notna().all()
    assert catalog.gdf["format"].tolist() == [format, format]


def test_write_manifest_preserves_ordered_custom_columns_and_nulls(tmp_path, raw):
    samples = tmp_path / "prepared"
    paths = {
        "north/a": write_dense_sample(samples / "north/a.zarr", raw, day=1),
        "south/b": write_dense_sample(samples / "south/b.zarr", raw, day=2),
    }
    metadata = {
        "north/a": {
            "split": "train",
            "quality": 0.9,
            "approved": True,
            "note": None,
        },
        "south/b": {
            "split": "validation",
            "quality": 0.8,
            "approved": False,
            "note": "review",
        },
    }
    destination = samples / "manifest.parquet"

    write_manifest(paths, destination, format="zarr", metadata=metadata)

    stored = gpd.read_parquet(destination)
    assert list(stored) == [
        "path",
        "format",
        "start_datetime",
        "end_datetime",
        "grid_crs",
        "grid_height",
        "grid_width",
        "split",
        "quality",
        "approved",
        "note",
        "geometry",
    ]
    assert stored.split.tolist() == ["train", "validation"]
    assert stored.quality.tolist() == [0.9, 0.8]
    assert stored.approved.tolist() == [True, False]
    assert stored.note.isna().tolist() == [True, False]


def test_write_manifest_does_not_compute_sample_pixels(tmp_path, raw):
    sample = write_dense_sample(tmp_path / "samples" / "a.zarr", raw, day=1)
    started = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        write_manifest(
            {"a": sample}, tmp_path / "manifest.parquet", format="zarr"
        )

    assert started == []


def test_write_manifest_replaces_rows_atomically(tmp_path, raw):
    samples = tmp_path / "samples"
    first = write_dense_sample(samples / "a.zarr", raw, day=1)
    second = write_dense_sample(samples / "b.zarr", raw, day=2)
    destination = tmp_path / "manifest.parquet"
    write_manifest({"a": first, "b": second}, destination, format="zarr")

    write_manifest({"b": second}, destination, format="zarr")

    assert io.read_vector(destination).gdf.path.tolist() == [Path(second).resolve()]
    assert not (tmp_path / ".manifest.staging.parquet").exists()


def test_write_manifest_resolves_relative_sample_paths(tmp_path, raw, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = write_dense_sample(Path("prepared/a.zarr"), raw, day=1)
    destination = Path("prepared/manifest.parquet")

    write_manifest({"a": sample}, destination, format="zarr")

    assert gpd.read_parquet(destination).iloc[0].path == "a.zarr"
    assert io.read_vector(destination).gdf.iloc[0].path == (
        tmp_path / "prepared/a.zarr"
    )


def test_write_manifest_combines_samples_from_different_native_crss(tmp_path):
    paths = {}
    grid_crss = []
    for sample_id, crs in (("utm", "EPSG:32633"), ("global", "EPSG:4326")):
        geobox = GeoBox.from_bbox((0, 0, 4, 4), crs=crs, shape=(4, 4))
        grid_crss.append(geobox.crs)
        label = raster(
            {"class": np.ones((4, 4), dtype="uint8")}, geobox
        ).assign_coords(time=np.datetime64("2025-01-01"))
        path = tmp_path / "samples" / f"{sample_id}.zarr"
        paths[sample_id] = write_sample(
            {"label": label, "optical": label.rename({"class": "red"})},
            path,
            format="zarr",
        )

    write_manifest(paths, tmp_path / "manifest.parquet", format="zarr")

    catalog = io.read_vector(tmp_path / "manifest.parquet")
    assert catalog.crs.to_epsg() == 4326
    assert [CRS(value) for value in catalog.gdf.grid_crs] == grid_crss

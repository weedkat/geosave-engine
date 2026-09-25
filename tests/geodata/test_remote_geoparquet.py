from pathlib import Path
from uuid import uuid4

import fsspec
import geopandas as gpd
import pytest
from fsspec.spec import AbstractFileSystem
from huggingface_hub import get_token
from shapely.geometry import Point, box

from geosave_engine.geodata import GeoVector, read_vector


def memory_catalog() -> str:
    return f"memory://geosave-tests/{uuid4()}/catalog.parquet"


def test_remote_geoparquet_round_trip_filters_and_materializes_path() -> None:
    destination = memory_catalog()
    parent = destination.rsplit("/", 1)[0]
    vector = GeoVector(
        gpd.GeoDataFrame(
            {
                "name": ["near", "far"],
                "path": [f"{parent}/rasters/near.zarr", None],
            },
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs="EPSG:4326",
        )
    )

    written = vector.to_geoparquet(destination, write_covering_bbox=True)
    selected = read_vector(
        destination,
        bbox=(-1, -1, 2, 2),
        columns=["name", "path", "geometry"],
    )

    filesystem, target = fsspec.core.url_to_fs(destination)
    expected_asset = filesystem.unstrip_protocol(
        f"{target.rsplit('/', 1)[0]}/rasters/near.zarr"
    )
    assert written == destination
    assert list(selected.gdf.name) == ["near"]
    assert selected.gdf.iloc[0].path == expected_asset
    assert "token" not in selected.gdf
    assert not hasattr(selected, "source")


def test_external_remote_asset_does_not_require_its_driver() -> None:
    destination = memory_catalog()
    external = "s3://other-bucket/prediction.zarr"

    GeoVector.from_geometry(Point(0, 0), path=external).to_geoparquet(
        destination
    )

    assert read_vector(destination).gdf.iloc[0].path == external


def test_null_asset_pointer_remains_null() -> None:
    destination = memory_catalog()
    vector = GeoVector(
        gpd.GeoDataFrame(
            {"path": [None]}, geometry=[Point(0, 0)], crs="EPSG:4326"
        )
    )

    vector.to_geoparquet(destination)

    assert read_vector(destination).gdf.path.isna().all()


def test_opaque_asset_pointer_fails_before_writing() -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)
    vector = GeoVector.from_geometry(Point(0, 0), path=object())

    with pytest.raises(TypeError, match="asset path must be string or path-like"):
        vector.to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_remote_write_refuses_existing_catalog() -> None:
    destination = memory_catalog()
    vector = GeoVector.from_geometry(Point(0, 0))
    vector.to_geoparquet(destination)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        vector.to_geoparquet(destination)


def test_file_url_returns_a_usable_local_path(tmp_path: Path) -> None:
    destination = tmp_path / "catalog.parquet"

    written = GeoVector.from_geometry(Point(0, 0)).to_geoparquet(
        destination.as_uri()
    )

    assert written == destination
    assert len(read_vector(written)) == 1


def test_invalid_remote_suffix_creates_nothing() -> None:
    destination = memory_catalog().replace(".parquet", ".json")
    filesystem, target = fsspec.core.url_to_fs(destination)

    with pytest.raises(ValueError, match="must end"):
        GeoVector.from_geometry(Point(0, 0)).to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_failed_new_remote_write_removes_its_partial_object(monkeypatch) -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)

    def fail(_self, path: str, **options: object) -> None:
        remote = options["filesystem"]
        assert isinstance(remote, AbstractFileSystem)
        remote.pipe(path, b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)

    with pytest.raises(RuntimeError, match="encode failed"):
        GeoVector.from_geometry(Point(0, 0)).to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_failed_remote_overwrite_does_not_delete_existing_object(
    monkeypatch,
) -> None:
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)
    filesystem.pipe(target, b"existing")

    def fail(_self, path: str, **options: object) -> None:
        remote = options["filesystem"]
        assert remote is filesystem
        assert path == target
        raise RuntimeError("encode failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)

    with pytest.raises(RuntimeError, match="encode failed"):
        GeoVector.from_geometry(Point(0, 0)).to_geoparquet(
            destination, overwrite=True
        )

    assert filesystem.cat(target) == b"existing"


@pytest.mark.integration
def test_huggingface_bucket_geoparquet_round_trip() -> None:
    token = get_token()
    if token is None:
        pytest.skip("HF_TOKEN or `hf auth login` is required")

    prefix = f"hf://buckets/fatmur/test/geosave-tests/{uuid4()}"
    destination = f"{prefix}/catalog.parquet"
    filesystem, prefix_path = fsspec.core.url_to_fs(prefix)
    vector = GeoVector(
        gpd.GeoDataFrame(
            {"name": ["near", "far"]},
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs="EPSG:4326",
        )
    )

    try:
        vector.to_geoparquet(destination, write_covering_bbox=True)
        selected = read_vector(
            destination,
            bbox=(-1, -1, 2, 2),
            columns=["name", "geometry"],
        )

        assert list(selected.gdf.name) == ["near"]
    finally:
        if filesystem.exists(prefix_path):
            filesystem.rm(prefix_path, recursive=True)

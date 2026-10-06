from pathlib import Path

import fsspec
import pytest
from fsspec.spec import AbstractFileSystem

from geosave_engine.geodata import io
from geosave_engine.geodata.io.storage import (
    filesystem_path,
    gdal_path,
    local_target,
)

from tests.geodata.conftest import build_raster


def test_filesystem_path_forwards_storage_options(monkeypatch) -> None:
    memory = fsspec.filesystem("memory")
    received: dict[str, object] = {}

    def resolve(
        location: str, **options: object
    ) -> tuple[AbstractFileSystem, str]:
        received["location"] = location
        received["options"] = options
        return memory, "/catalogs/catalog.parquet"

    monkeypatch.setattr(fsspec.core, "url_to_fs", resolve)

    result = filesystem_path(
        "hf://buckets/fatmur/test/catalog.parquet",
        {"token": "test-token"},
    )

    assert result == (memory, "/catalogs/catalog.parquet")
    assert received == {
        "location": "hf://buckets/fatmur/test/catalog.parquet",
        "options": {"token": "test-token"},
    }


def test_a_local_location_is_its_own_target(tmp_path: Path) -> None:
    with local_target(tmp_path / "scene.tif") as target:
        assert target == tmp_path / "scene.tif"


def test_a_remote_location_uploads_what_was_written(bucket: str) -> None:
    with local_target(f"{bucket}/scene.tif") as target:
        assert target.name == "scene.tif"
        target.write_bytes(b"pixels")

    assert fsspec.open(f"{bucket}/scene.tif").open().read() == b"pixels"
    assert not target.exists()


def test_a_failed_write_uploads_nothing(bucket: str) -> None:
    with pytest.raises(RuntimeError), local_target(f"{bucket}/scene.tif") as target:
        target.write_bytes(b"partial")
        raise RuntimeError("writer failed")

    assert not fsspec.filesystem("memory").exists(f"{bucket}/scene.tif")


def test_an_existing_remote_object_refuses_without_overwrite(bucket: str) -> None:
    with local_target(f"{bucket}/scene.tif") as target:
        target.write_bytes(b"first")

    with pytest.raises(FileExistsError), local_target(f"{bucket}/scene.tif"):
        pytest.fail("the body must not run")
    with local_target(f"{bucket}/scene.tif", overwrite=True) as target:
        target.write_bytes(b"second")
    assert fsspec.open(f"{bucket}/scene.tif").open().read() == b"second"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("hf://buckets/me/samples/forest/a.tif", "s3://me/samples/forest/a.tif"),
        ("s3://bucket/forest/a.tif", "s3://bucket/forest/a.tif"),
        ("https://example.com/a.tif", "https://example.com/a.tif"),
        ("/data/a.tif", "/data/a.tif"),
    ],
)
def test_gdal_reads_an_hf_bucket_through_its_s3_gateway(location, expected) -> None:
    assert gdal_path(location) == expected


def test_only_hf_buckets_have_an_s3_form() -> None:
    with pytest.raises(ValueError, match="buckets"):
        gdal_path("hf://datasets/me/repo/a.tif")


@pytest.mark.parametrize(
    ("name", "write"),
    [
        ("forest.tif", lambda cube, url: io.geotiff.write_cog(cube.isel(time=0), url)),
        ("forest.zarr", lambda cube, url: io.zarr.write(cube, url)),
        ("forest.nc", lambda cube, url: io.netcdf.write(cube, url)),
    ],
)
def test_a_file_url_writes_to_the_local_path_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name, write
) -> None:
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    written = write(build_raster(times=1), f"file://{tmp_path}/{name}")

    assert written == tmp_path / name
    assert (tmp_path / name).exists()
    assert list(elsewhere.iterdir()) == []

from pathlib import Path

import fsspec
from fsspec.spec import AbstractFileSystem

from geosave_engine.geodata.utils.io.storage import (
    filesystem_path,
    resolve_asset_path,
    stored_asset_path,
)


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


def test_remote_asset_paths_round_trip_without_contacting_the_asset() -> None:
    filesystem, catalog_path = filesystem_path(
        "memory://dataset/catalog.parquet"
    )

    assert stored_asset_path(
        "memory://dataset/rasters/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "rasters/prediction.zarr"
    assert resolve_asset_path(
        "rasters/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "memory:///dataset/rasters/prediction.zarr"
    assert stored_asset_path(
        "s3://other-bucket/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "s3://other-bucket/prediction.zarr"


def test_parent_relative_asset_is_allowed_without_asset_lookup() -> None:
    filesystem, catalog_path = filesystem_path(
        "memory://dataset/catalogs/catalog.parquet"
    )

    assert resolve_asset_path(
        "../shared/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == "memory:///dataset/shared/prediction.zarr"


def test_local_asset_paths_are_lexical_and_portable(tmp_path: Path) -> None:
    catalog = tmp_path / "dataset" / "catalog.parquet"
    inside = tmp_path / "dataset" / "rasters" / "prediction.zarr"
    outside = tmp_path / "shared" / "prediction.zarr"
    filesystem, catalog_path = filesystem_path(catalog)

    assert stored_asset_path(
        inside, filesystem=filesystem, catalog_path=catalog_path
    ) == "rasters/prediction.zarr"
    assert stored_asset_path(
        outside, filesystem=filesystem, catalog_path=catalog_path
    ) == str(outside)
    assert resolve_asset_path(
        "../shared/prediction.zarr",
        filesystem=filesystem,
        catalog_path=catalog_path,
    ) == outside

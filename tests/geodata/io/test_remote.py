"""Remote round trips against an S3-compatible service or a Hugging Face bucket.

Set one of these to run them:

- `GEOSAVE_S3_ENDPOINT`, for example `http://127.0.0.1:9000` for the rustfs
  service in `infra/docker-compose.yml`. The bucket comes from
  `GEOSAVE_S3_BUCKET`.
- `GEOSAVE_HF_BUCKET`, as `<namespace>/<bucket>`. Files travel with `HF_TOKEN`
  or the cached login; GDAL reads TIFFs through the bucket's S3 gateway.

Either way GDAL takes its keys from `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`, which for a Hugging Face bucket are the S3 credentials
generated from the token.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit

import dask
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from geosave_engine.geodata import configure_gdal, read_raster, read_vector, stack
from geosave_engine.geodata.io import storage
from geosave_engine.geodata import stac
from geosave_engine.geodata.stac import table

from tests.geodata.conftest import build_raster

pytestmark = pytest.mark.integration

_GDAL_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "AWS_S3_ENDPOINT",
    "AWS_VIRTUAL_HOSTING",
    "AWS_HTTPS",
    "GDAL_DISABLE_READDIR_ON_OPEN",
)


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


def _size(url: str, options: dict) -> int:
    filesystem, path = storage.filesystem_path(url, options)
    return filesystem.size(path)


def _s3_service() -> tuple[str, dict, str]:
    """Return the prefix, storage options and GDAL endpoint of an S3 service."""
    endpoint = os.environ["GEOSAVE_S3_ENDPOINT"]
    bucket = os.environ.get("GEOSAVE_S3_BUCKET", "predictions")
    options = {
        "key": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        "client_kwargs": {"endpoint_url": endpoint},
    }
    filesystem, _ = storage.filesystem_path(f"s3://{bucket}", options)
    if not filesystem.exists(bucket):
        filesystem.mkdir(bucket)
    return f"s3://{bucket}", options, endpoint


def _hf_bucket() -> tuple[str, dict, str]:
    """Return the prefix, storage options and GDAL endpoint of an HF bucket."""
    return f"hf://buckets/{os.environ['GEOSAVE_HF_BUCKET']}", {}, "https://s3.hf.co"


@pytest.fixture(params=["s3", "hf"])
def remote(request):
    """Yield a unique remote prefix and its storage options, emptied afterwards."""
    variable = {"s3": "GEOSAVE_S3_ENDPOINT", "hf": "GEOSAVE_HF_BUCKET"}[request.param]
    if variable not in os.environ:
        pytest.skip(f"set {variable} to run against this service")
    root, options, endpoint = _s3_service() if request.param == "s3" else _hf_bucket()
    prefix = f"{root}/geosave-tests/{uuid.uuid4()}"

    # GDAL reads TIFFs itself, so it is told about the same service.
    before = {name: os.environ.get(name) for name in _GDAL_KEYS}
    address = urlsplit(endpoint)
    configure_gdal(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        aws_default_region="us-east-1",
        aws_s3_endpoint=address.netloc,
        aws_virtual_hosting=False,
        aws_https=address.scheme == "https",
        gdal_disable_readdir_on_open="EMPTY_DIR",
    )
    try:
        yield prefix, options
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        filesystem, path = storage.filesystem_path(prefix, options)
        if filesystem.exists(path):
            filesystem.rm(path, recursive=True)


def test_cogs_round_trip_lazily(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)

    paths = source.gs.to_cog(f"{prefix}/forest", storage_options=options)

    assert all(str(path).startswith(prefix) for path in paths)
    with dask.config.set(scheduler=_refuse):
        restored = read_raster(list(paths), chunks={})
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


def test_a_zarr_store_round_trips_lazily(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)

    store = source.gs.to_zarr(f"{prefix}/forest.zarr", storage_options=options)

    with dask.config.set(scheduler=_refuse):
        restored = read_raster(store, chunks={}, storage_options=options)
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


def test_a_netcdf_file_uploads(remote) -> None:
    prefix, options = remote

    written = build_raster(times=2).gs.to_netcdf(
        f"{prefix}/forest.nc", storage_options=options
    )

    assert written == f"{prefix}/forest.nc"
    assert _size(written, options) > 0


@pytest.mark.parametrize("suffix", [".parquet", ".geojson", ".gpkg"])
def test_a_plain_vector_uploads(remote, suffix) -> None:
    prefix, options = remote
    plots = gpd.GeoDataFrame(
        {"name": ["near", "far"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    write = {
        ".parquet": plots.gs.to_geoparquet,
        ".geojson": plots.gs.to_geojson,
        ".gpkg": plots.gs.to_geopackage,
    }[suffix]

    written = write(f"{prefix}/plots{suffix}", storage_options=options)

    assert written == f"{prefix}/plots{suffix}"
    assert _size(written, options) > 0


def test_geoparquet_reads_back_with_a_bbox(remote) -> None:
    prefix, options = remote
    plots = gpd.GeoDataFrame(
        {"name": ["near", "far"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    written = plots.gs.to_geoparquet(
        f"{prefix}/plots.parquet", write_covering_bbox=True, storage_options=options
    )

    selected = read_vector(written, bbox=(-1, -1, 2, 2), storage_options=options)

    assert selected["name"].tolist() == ["near"]


def test_an_item_table_of_cogs_loads_back(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)
    paths = source.gs.to_cog(f"{prefix}/forest", storage_options=options)
    # GDAL reads a remote TIFF with the keys in its environment, not fsspec options.
    items = stac.create_items(paths)

    catalog = table.write(items, f"{prefix}/catalog.parquet", storage_options=options)

    rows = table.read(catalog, storage_options=options)
    assert rows.iloc[0]["assets"]["image"]["href"] == str(paths[0])
    with dask.config.set(scheduler=_refuse):
        restored = table.load(table.to_items(rows))
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


def test_an_item_table_of_a_stack_store_loads_each_group(remote) -> None:
    prefix, options = remote
    optical = build_raster(times=2)
    sample = stack({"optical": optical, "label": build_raster(times=2)})
    stores = sample.gs.to_zarr(f"{prefix}/s0", storage_options=options)
    items = stac.create_stack_items(stores, name="s0", storage_options=options)
    catalog = table.write(items, f"{prefix}/catalog.parquet", storage_options=options)

    rows = table.read(catalog, storage_options=options)
    restored = table.load(
        table.to_items(rows[rows["collection"] == "optical"]), storage_options=options
    )

    np.testing.assert_array_equal(restored.red.values, optical.red.values)
    restored.close()

from pathlib import Path

import dask.array as da
from dask import delayed
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
from rioxarray._io import RasterioArrayWrapper
from torch import nn
import xarray as xr

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.inference import infer
from geosave_engine.workflow.io import open_rasters, write_stack
from geosave_engine.workflow.preprocessing import preprocess
from geosave_engine.workflow.spec import ModelSpec


@pytest.fixture
def scene() -> xr.Dataset:
    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32633", resolution=10)
    return raster(
        {
            "red": np.array([[1, 2], [3, 4]], dtype="uint16"),
            "nir": np.array([[5, 6], [7, 8]], dtype="uint16"),
        },
        box,
    )


def test_open_rasters_preserves_caller_owned_native_objects(scene):
    raw = stack({"optical": scene})
    closed = False

    def mark_closed():
        nonlocal closed
        closed = True

    raw.set_close(mark_closed)
    with open_rasters(raw) as opened:
        assert opened is raw
    assert not closed

    scene.set_close(mark_closed)
    with open_rasters({"optical": scene}) as opened:
        assert opened.gs.groups == ("optical",)
        assert opened.gs.rasters["optical"].red.data is scene.red.data
    assert not closed


def test_open_rasters_reads_named_geotiff_and_closes_owned_file(
    monkeypatch, scene, tmp_path
):
    path = tmp_path / "optical.tif"
    io.geotiff.write_gtiff(scene, path)

    original_open = io.gdal.rioxarray.open_rasterio
    opened = []
    closed = []

    def track_backend_close(*args, **kwargs):
        raster = original_open(*args, **kwargs)
        backend_close = raster._close
        assert backend_close is not None

        def close():
            closed.append(raster)
            backend_close()

        raster.set_close(close)
        opened.append(raster)
        return raster

    monkeypatch.setattr(io.gdal.rioxarray, "open_rasterio", track_backend_close)

    try:
        with open_rasters({"optical": path}) as rasters:
            optical = rasters.gs.rasters["optical"]
            assert list(optical.data_vars) == ["red", "nir"]
            np.testing.assert_array_equal(optical.red, scene.red)
        assert closed == opened
    finally:
        for raster in opened:
            raster.close()


def test_open_rasters_keeps_geotiff_pixels_lazy_until_after_tile_context(
    monkeypatch, tmp_path
):
    box = GeoBox.from_bbox((0, 0, 160, 160), crs="EPSG:32633", resolution=10)
    scene = raster({"signal": np.ones((16, 16), dtype="float32")}, box)
    path = tmp_path / "image.tif"
    io.geotiff.write_gtiff(scene, path)
    spec = ModelSpec(
        sources={"image": {"variables": ("signal",)}},
        inference={
            "inputs": {"image": {"raster": "image"}},
            "tiling": {
                "raster": "image",
                "tile_shape": (4, 4),
                "overlap": 2,
                "window": "hann",
            },
        },
    )
    events = []
    original_getitem = RasterioArrayWrapper._getitem

    def track_read(self, key):
        events.append(("read", str(key)))
        return original_getitem(self, key)

    class Identity(nn.Module):
        def forward(self, image):
            return image

    def context(tile):
        events.append(("context", str(tile.gs.geobox.shape)))
        return {}

    monkeypatch.setattr(RasterioArrayWrapper, "_getitem", track_read)
    with open_rasters({"image": path}) as opened:
        image = opened.gs.rasters["image"]
        assert isinstance(image.signal.data, da.Array)
        prepared = preprocess(opened, spec=spec)
        assert events == []
        outputs = infer(
            prepared,
            model=Identity(),
            settings=spec.inference,
            model_context=context,
            batch_size=4,
        )

    assert events[0][0] == "context"
    assert any(kind == "read" for kind, _ in events)
    np.testing.assert_allclose(outputs[0], 1)


def test_open_rasters_closes_owned_files_after_a_partial_open_failure(
    monkeypatch, scene
):
    closed = False

    def mark_closed():
        nonlocal closed
        closed = True

    scene.set_close(mark_closed)

    def read_raster(path, **options):
        assert options == {"chunks": "auto"}
        if path == Path("first.tif"):
            return scene
        raise ValueError("unsupported raster suffix")

    monkeypatch.setattr(io, "read_raster", read_raster)
    with pytest.raises(ValueError, match="unsupported"):
        with open_rasters(
            {"first": Path("first.tif"), "broken": Path("broken.unknown")}
        ):
            pytest.fail("a partially opened mapping must not be yielded")
    assert closed


def test_open_rasters_reports_an_unknown_named_raster_suffix(tmp_path):
    with pytest.raises(ValueError, match="no supported raster format"):
        with open_rasters({"broken": tmp_path / "broken.unknown"}):
            pytest.fail("an unsupported path must not be yielded")


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_open_rasters_reads_named_single_raster_stores(scene, tmp_path, suffix):
    path = tmp_path / f"optical{suffix}"
    if suffix == ".zarr":
        io.zarr.write(scene, path)
    else:
        io.netcdf.write(scene, path)

    with open_rasters({"optical": path}) as opened:
        optical = opened.gs.rasters["optical"]
        assert list(optical.data_vars) == ["red", "nir"]
        np.testing.assert_array_equal(optical.nir, scene.nir)


def test_open_rasters_closes_original_named_zarr(monkeypatch, scene, tmp_path):
    path = tmp_path / "optical.zarr"
    io.zarr.write(scene, path)
    original_open = io.zarr.xr.open_dataset
    opened = []
    closed = []

    def track_backend_close(*args, **kwargs):
        raster = original_open(*args, **kwargs)
        backend_close = raster._close
        assert backend_close is not None

        def close():
            closed.append(raster)
            backend_close()

        raster.set_close(close)
        opened.append(raster)
        return raster

    monkeypatch.setattr(io.zarr.xr, "open_dataset", track_backend_close)
    try:
        with open_rasters({"optical": path}):
            pass
        assert closed == opened
    finally:
        for raster in opened:
            raster.close()


def test_open_rasters_closes_original_saved_stack(monkeypatch, scene, tmp_path):
    path = tmp_path / "raw.zarr"
    write_stack(stack({"optical": scene}), path)
    original_open = io.zarr.xr.open_datatree
    opened = []
    closed = []

    def track_backend_close(*args, **kwargs):
        tree = original_open(*args, **kwargs)
        owner = next(node for node in tree.subtree if node._close is not None)
        backend_close = owner._close

        def close():
            closed.append(owner)
            backend_close()

        owner.set_close(close)
        opened.append(tree)
        return tree

    monkeypatch.setattr(io.zarr.xr, "open_datatree", track_backend_close)
    try:
        with open_rasters(path):
            pass
        assert len(closed) == 1
    finally:
        for tree in opened:
            tree.close()


def test_write_stack_completes_before_return_and_reopens_as_standalone_path(
    scene, tmp_path
):
    raw = stack({"optical": scene[["red"]], "terrain": scene[["nir"]]})
    path = tmp_path / "raw.zarr"

    saved = write_stack(raw, path)

    assert saved == str(path)
    with open_rasters(saved) as restored:
        assert set(restored.gs.groups) == {"optical", "terrain"}
        np.testing.assert_array_equal(restored.gs.rasters["optical"].red, scene.red)
        np.testing.assert_array_equal(restored.gs.rasters["terrain"].nir, scene.nir)


def test_write_stack_rejects_remote_destinations_and_existing_stores(scene, tmp_path):
    raw = stack({"optical": scene})
    with pytest.raises(ValueError, match="local"):
        write_stack(raw, "s3://bucket/raw.zarr")

    path = tmp_path / "raw.zarr"
    write_stack(raw, path)
    with pytest.raises(FileExistsError):
        write_stack(stack({"replacement": scene * 10}), path)
    with open_rasters(path) as preserved:
        assert preserved.gs.groups == ("optical",)
        np.testing.assert_array_equal(preserved.gs.rasters["optical"].red, scene.red)
    assert list(tmp_path.glob(f".{path.name}-*")) == []


def test_write_stack_removes_partial_store_and_allows_retry(scene, tmp_path):
    def fail_chunk():
        raise RuntimeError("delayed chunk failed")

    failed = scene.copy()
    failed["red"] = (
        scene.red.dims,
        da.from_delayed(
            delayed(fail_chunk)(), shape=scene.red.shape, dtype=scene.red.dtype
        ),
    )
    path = tmp_path / "raw.zarr"

    with pytest.raises(RuntimeError, match="delayed chunk failed"):
        write_stack(stack({"optical": failed}), path)

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}-*")) == []

    saved = write_stack(stack({"optical": scene}), path)
    assert saved == str(path)
    with open_rasters(path) as restored:
        np.testing.assert_array_equal(restored.gs.rasters["optical"].red, scene.red)

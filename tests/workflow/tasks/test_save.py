import dask.array as da
from dask.delayed import delayed
import numpy as np
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
import geosave_engine.workflow.tasks.save as save_module
from geosave_engine.workflow.tasks.save import write_stack


def _sample_rasters(raw):
    optical = raw["optical"][["red", "nir"]].astype("uint16")
    label = raster(
        {"class": np.ones(optical.gs.geobox.shape, dtype="uint8")},
        optical.gs.geobox,
    ).assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    return {"label": label, "optical": optical}


def test_write_stack_completes_before_return_and_reopens(raw, tmp_path):
    output = tmp_path / "raw.zarr"

    saved = write_stack(raw, output)

    assert saved == str(output)
    with io.read_stack(output, chunks="auto") as restored:
        assert restored.gs.groups == ("optical",)
        np.testing.assert_array_equal(restored["optical"].red, raw["optical"].red)


def test_write_stack_preserves_existing_destination(raw, tmp_path):
    output = tmp_path / "raw.zarr"
    write_stack(raw, output)

    with pytest.raises(FileExistsError, match="already exists"):
        write_stack({"replacement": raw["optical"] * 10}, output)

    with io.read_stack(output, chunks="auto") as restored:
        assert restored.gs.groups == ("optical",)
        np.testing.assert_array_equal(restored["optical"].red, raw["optical"].red)


@pytest.mark.parametrize("output", ["raw.nc", "s3://bucket/raw.zarr"])
def test_write_stack_requires_a_local_zarr_destination(output):
    with pytest.raises(ValueError, match="local .zarr"):
        write_stack({}, output)


def test_write_stack_rechecks_destination_before_publish(raw, tmp_path, monkeypatch):
    output = tmp_path / "raw.zarr"
    original = io.zarr.write

    def create_competing_output(*args, **kwargs):
        result = original(*args, **kwargs)
        output.mkdir()
        return result

    monkeypatch.setattr(io.zarr, "write", create_competing_output)

    with pytest.raises(FileExistsError, match="already exists"):
        write_stack(raw, output)


def test_failed_write_cleans_staging_and_allows_retry(raw, tmp_path):
    def fail_chunk():
        raise RuntimeError("delayed chunk failed")

    failed = raw["optical"].copy()
    failed["red"] = (
        failed.red.dims,
        da.from_delayed(
            delayed(fail_chunk)(), shape=failed.red.shape, dtype=failed.red.dtype
        ),
    )
    output = tmp_path / "raw.zarr"

    with pytest.raises(RuntimeError, match="delayed chunk failed"):
        write_stack({"optical": failed}, output)

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}-*")) == []
    assert write_stack(raw, output) == str(output)


def test_write_sample_publishes_flat_geotiff_assets(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    output = tmp_path / "sample"

    result = save_module.write_sample(rasters, output)

    assert result == str(output)
    assert {path.name for path in output.iterdir()} == {"label.tif", "optical.tif"}
    with save_module.open_sample(output, format="geotiff") as restored:
        assert restored.gs.groups == ("label", "optical")
        assert restored.gs.geobox == rasters["label"].gs.geobox
        assert tuple(restored["label"].data_vars) == ("class",)
        assert tuple(restored["optical"].data_vars) == ("red", "nir")
        assert restored["label"]["class"].dtype == np.dtype("uint8")
        assert restored["optical"].red.dtype == np.dtype("uint16")


def test_open_sample_rebuilds_the_logical_stack(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    output = tmp_path / "sample"
    output.mkdir()
    for name, value in rasters.items():
        io.geotiff.write_cog(value, output / f"{name}.tif")

    with save_module.open_sample(output, format="geotiff") as restored:
        np.testing.assert_array_equal(
            restored["label"]["class"], rasters["label"]["class"]
        )
        np.testing.assert_array_equal(restored["optical"].red, rasters["optical"].red)


def test_write_sample_does_not_depend_on_raster_order(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)

    result = save_module.write_sample(
        {"optical": rasters["optical"], "label": rasters["label"]},
        tmp_path / "sample",
    )

    assert result == str(tmp_path / "sample")


def test_write_sample_keeps_a_singleton_time_as_a_scalar(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    instant = np.datetime64("2025-01-15T12:00:00")
    rasters["optical"] = rasters["optical"].expand_dims(time=[instant])

    save_module.write_sample(rasters, tmp_path / "sample")

    with save_module.open_sample(tmp_path / "sample", format="geotiff") as restored:
        assert restored["optical"].time.dims == ()
        assert restored["optical"].time.values == instant


def test_write_sample_refuses_a_multitemporal_geotiff(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    rasters["optical"] = rasters["optical"].expand_dims(
        time=[np.datetime64("2025-01-15"), np.datetime64("2025-01-16")]
    )

    with pytest.raises(ValueError, match="use format='zarr'"):
        save_module.write_sample(rasters, tmp_path / "sample")

    assert not (tmp_path / "sample").exists()


@pytest.mark.parametrize("option", ["layout", "split_bands", "overwrite", "compute"])
def test_write_sample_reserves_publication_options(
    raw, tmp_path, monkeypatch, option
) -> None:
    def unexpected_write(*args, **kwargs):
        pytest.fail(f"writer called with reserved option: {args}, {kwargs}")

    monkeypatch.setattr(io.geotiff, "write_cog", unexpected_write)

    with pytest.raises(ValueError, match=option):
        save_module.write_sample(
            _sample_rasters(raw),
            tmp_path / "sample",
            write_options={option: True},
        )


def test_write_sample_preserves_an_existing_destination_file(raw, tmp_path) -> None:
    output = tmp_path / "sample"
    output.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        save_module.write_sample(_sample_rasters(raw), output)

    assert output.is_file()


def test_failed_geotiff_sample_cleans_staging(raw, tmp_path) -> None:
    def fail_chunk():
        raise RuntimeError("delayed chunk failed")

    rasters = _sample_rasters(raw)
    failed = rasters["optical"].copy()
    failed["red"] = (
        failed.red.dims,
        da.from_delayed(
            delayed(fail_chunk)(), shape=failed.red.shape, dtype=failed.red.dtype
        ).rechunk((2, 2)),
    )
    output = tmp_path / "sample"

    with pytest.raises(RuntimeError, match="delayed chunk failed"):
        save_module.write_sample(
            {"label": rasters["label"], "optical": failed}, output
        )

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}-*")) == []


def test_write_sample_zarr_round_trip(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    output = tmp_path / "sample.zarr"

    assert save_module.write_sample(rasters, output, format="zarr") == str(output)

    with save_module.open_sample(output, format="zarr") as restored:
        assert set(restored.gs.groups) == {"label", "optical"}
        np.testing.assert_array_equal(
            restored["label"]["class"], rasters["label"]["class"]
        )


def test_write_sample_forwards_geotiff_encoding_options(
    raw, tmp_path, monkeypatch
) -> None:
    calls = []
    original = io.geotiff.write_cog

    def record_options(ds, path, **options):
        calls.append((path.name, options))
        return original(ds, path, **options)

    monkeypatch.setattr(io.geotiff, "write_cog", record_options)

    save_module.write_sample(
        _sample_rasters(raw),
        tmp_path / "sample",
        write_options={"compress": "ZSTD", "blocksize": 256},
    )

    assert calls == [
        ("label.tif", {"compress": "ZSTD", "blocksize": 256}),
        ("optical.tif", {"compress": "ZSTD", "blocksize": 256}),
    ]

import dask.array as da
from dask.delayed import delayed
import numpy as np
import pytest

from geosave_engine.geodata import read_stack
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata import io
import geosave_engine.workflow.tasks.sample as save_module


def _sample_rasters(raw):
    optical = raw["optical"][["red", "nir"]].astype("uint16")
    label = raster(
        {
            "class": (
                ("y", "x"),
                np.ones(optical.gs.geobox.shape, dtype="uint8"),
            )
        },
        optical.gs.geobox,
    ).assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    return {"label": label, "optical": optical}


def test_write_sample_publishes_flat_geotiff_assets(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    output = tmp_path / "sample"

    result = save_module.write_sample(rasters, output)

    assert result == str(output)
    assert {path.name for path in output.iterdir()} == {"label.tif", "optical.tif"}
    with read_stack(output) as restored:
        assert restored.gs.groups == ("label", "optical")
        assert restored.gs.geobox == rasters["label"].gs.geobox
        assert tuple(restored["label"].data_vars) == ("class",)
        assert tuple(restored["optical"].data_vars) == ("red", "nir")
        assert restored["label"]["class"].dtype == np.dtype("uint8")
        assert restored["optical"].red.dtype == np.dtype("uint16")


def test_write_sample_refuses_rasters_on_different_grids_before_writing(
    raw, tmp_path, monkeypatch
) -> None:
    rasters = _sample_rasters(raw)
    grid = rasters["label"].gs.geobox
    moved = raster(
        {
            "class": (
                ("y", "x"),
                np.ones(grid.shape, dtype="uint8"),
            )
        },
        grid.translate_pix(1, 1),
    )

    def unexpected_write(*args, **kwargs):
        pytest.fail(f"writer called for misaligned rasters: {args}, {kwargs}")

    monkeypatch.setattr(io.geotiff, "write_cog", unexpected_write)

    with pytest.raises(ValueError, match="do not share a grid"):
        save_module.write_sample(
            {"label": moved, "optical": rasters["optical"]}, tmp_path / "sample"
        )

    assert list(tmp_path.iterdir()) == []


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

    with read_stack(tmp_path / "sample") as restored:
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
        save_module.write_sample({"label": rasters["label"], "optical": failed}, output)

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}-*")) == []


def test_write_sample_zarr_round_trip(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    output = tmp_path / "sample"

    assert save_module.write_sample(rasters, output, format="zarr") == str(output)

    assert {path.name for path in output.iterdir()} == {"label.zarr", "optical.zarr"}
    with read_stack(output) as restored:
        assert restored.gs.groups == ("label", "optical")
        np.testing.assert_array_equal(
            restored["label"]["class"], rasters["label"]["class"]
        )


def test_a_zarr_sample_keeps_a_time_series(raw, tmp_path) -> None:
    rasters = _sample_rasters(raw)
    days = [np.datetime64("2025-01-15"), np.datetime64("2025-01-16")]
    rasters["optical"] = rasters["optical"].expand_dims(time=days)

    save_module.write_sample(rasters, tmp_path / "sample", format="zarr")

    with read_stack(tmp_path / "sample") as restored:
        assert restored["optical"].sizes["time"] == 2


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


def test_write_sample_forwards_zarr_options(raw, tmp_path, monkeypatch) -> None:
    calls = []
    original = io.zarr.write

    def record_options(ds, path, **options):
        calls.append((path.name, options))
        return original(ds, path, **options)

    monkeypatch.setattr(io.zarr, "write", record_options)

    save_module.write_sample(
        _sample_rasters(raw),
        tmp_path / "sample",
        format="zarr",
        write_options={"consolidated": True},
    )

    assert calls == [
        ("label.zarr", {"consolidated": True}),
        ("optical.zarr", {"consolidated": True}),
    ]


@pytest.mark.parametrize("name", ["scene.zarr", "product.SAFE"])
def test_write_sample_refuses_a_destination_named_like_a_store(
    raw, tmp_path, name: str
) -> None:
    with pytest.raises(ValueError, match="names a store"):
        save_module.write_sample(_sample_rasters(raw), tmp_path / name)

    assert not (tmp_path / name).exists()

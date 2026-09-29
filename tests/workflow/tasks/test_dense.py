from importlib import import_module
from types import SimpleNamespace
import numpy as np
from odc.geo.geobox import GeoBox
from prefect.cache_policies import NO_CACHE
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.model_spec import ModelSpec, RasterRequirement, StacRecipe
from geosave_engine.workflow import tasks as workflow_tasks
from geosave_engine.workflow.tasks.save import open_sample, write_sample

try:
    dense_module = import_module("geosave_engine.workflow.tasks.dense")
except ModuleNotFoundError:
    dense_module = SimpleNamespace()
    _dense_imported = False
else:
    _dense_imported = True


def test_dense_task_module_exists() -> None:
    assert _dense_imported


def _write_label(path, anchor, *, dated=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = raster(
        {"class": np.full((4, 4), 1, dtype="uint8")}, anchor.geobox
    )
    if dated:
        label = label.assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    return label, io.geotiff.write_cog(label, path)


def _requirement(url, *, dims=None):
    return RasterRequirement(
        variables=("red", "nir"),
        stac=StacRecipe.model_validate(
            {
                "collection": "optical",
                "endpoints": (url,),
                "load": {"bands": ["red", "nir"]},
            }
        ),
        dims=dims,
        require_crs=True,
    )


def _model(requirements):
    return ModelSpec(schema_version=2, rasters=requirements)


def test_prepare_dense_sample_is_not_cached_or_persisted() -> None:
    assert dense_module.prepare_dense_sample is workflow_tasks.prepare_dense_sample
    assert dense_module.prepare_dense_sample.cache_policy is NO_CACHE
    assert dense_module.prepare_dense_sample.persist_result is False


def test_prepare_dense_sample_writes_label_and_matching_imagery(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"

    result = dense_module.prepare_dense_sample.fn(
        label_path,
        _model({"optical": _requirement(url)}),
        output,
    )

    assert result == str(output)
    assert requests == [["optical"]]
    assert sorted(path.name for path in output.iterdir()) == [
        "label.tif",
        "optical.tif",
    ]
    with open_sample(output, format="geotiff") as sample:
        assert set(sample.gs.groups) == {"label", "optical"}
        assert sample.gs.geobox == anchor.geobox
        np.testing.assert_array_equal(sample["label"]["class"], label["class"])
        assert tuple(sample["optical"].data_vars) == ("red", "nir")
        assert sample["label"]["class"].chunks is not None
        assert sample["optical"].red.chunks is not None


def test_prepare_dense_sample_supports_zarr(tmp_path, stac_server) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"

    result = dense_module.prepare_dense_sample.fn(
        label_path,
        _model({"optical": _requirement(url)}),
        output,
        format="zarr",
    )

    assert result == str(output)
    assert requests == [["optical"]]
    with open_sample(output, format="zarr") as sample:
        assert set(sample.gs.groups) == {"label", "optical"}
        np.testing.assert_array_equal(sample["label"]["class"], label["class"])


def test_prepare_dense_sample_reuses_a_valid_sample_without_stac(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    model = _model({"optical": _requirement(url)})
    output = tmp_path / "sample"
    dense_module.prepare_dense_sample.fn(label_path, model, output)

    assert dense_module.prepare_dense_sample.fn(
        label_path, model, output
    ) == str(output)
    assert requests == [["optical"]]


def test_prepare_dense_sample_reuses_geotiff_with_required_time_dimension(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    model = _model({
        "optical": _requirement(url, dims=("time", "y", "x"))
    })
    output = tmp_path / "sample"

    dense_module.prepare_dense_sample.fn(label_path, model, output)

    assert dense_module.prepare_dense_sample.fn(
        label_path, model, output
    ) == str(output)
    assert requests == [["optical"]]


def test_prepare_dense_sample_rejects_an_incomplete_existing_sample(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    write_sample({"optical": label.rename({"class": "red"})}, output)

    with pytest.raises(ValueError, match="does not match model rasters"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _model({"optical": _requirement(url)}),
            output,
        )

    assert requests == []
    with open_sample(output, format="geotiff") as sample:
        assert sample.gs.groups == ("optical",)


def test_validate_dense_sample_rejects_stale_raster_variables(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, _ = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    optical = raster({"red": np.ones((4, 4), dtype="uint16")}, anchor.geobox)
    write_sample({"label": label, "optical": optical}, output)

    with pytest.raises(ValueError, match="nir"):
        dense_module._validate_dense_sample(
            output, {"optical": _requirement(url)}
        )

    assert requests == []


def test_prepare_dense_sample_rejects_unexpected_geotiff_asset(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    optical = raster(
        {
            "red": np.ones((4, 4), dtype="uint16"),
            "nir": np.ones((4, 4), dtype="uint16"),
        },
        anchor.geobox,
    )
    write_sample({"label": label, "optical": optical}, output)
    io.geotiff.write_cog(optical, output / "unexpected.tif")

    with pytest.raises(ValueError, match="does not match model rasters"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _model({"optical": _requirement(url)}),
            output,
        )

    assert requests == []
    assert (output / "unexpected.tif").exists()


def test_prepare_dense_sample_rejects_misaligned_geotiff_assets(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    output.mkdir()
    io.geotiff.write_cog(label, output / "label.tif")
    shifted = GeoBox.from_bbox(
        (10, 0, 50, 40), crs=anchor.geobox.crs, shape=(4, 4)
    )
    optical = raster(
        {
            "red": np.ones((4, 4), dtype="uint16"),
            "nir": np.ones((4, 4), dtype="uint16"),
        },
        shifted,
    )
    io.geotiff.write_cog(optical, output / "optical.tif")

    with pytest.raises(ValueError):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _model({"optical": _requirement(url)}),
            output,
        )

    assert requests == []
    assert sorted(path.name for path in output.iterdir()) == [
        "label.tif",
        "optical.tif",
    ]


def test_prepare_dense_sample_requires_label_time(tmp_path, stac_server) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(
        tmp_path / "timeless.tif", anchor, dated=False
    )

    with pytest.raises(ValueError, match="Label raster has no time"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _model({"optical": _requirement(url)}),
            tmp_path / "sample",
        )

    assert requests == []

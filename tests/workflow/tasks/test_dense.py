from importlib import import_module
from types import SimpleNamespace
import numpy as np
from prefect.cache_policies import NO_CACHE
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import RasterRequirement
from geosave_engine.workflow.tasks.save import write_stack

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


def _requirement(url):
    return RasterRequirement(
        variables=("red", "nir"),
        collection="optical",
        endpoints=(url,),
        require_crs=True,
    )


def test_prepare_dense_sample_is_not_cached_or_persisted() -> None:
    assert dense_module._prepare_dense_sample.cache_policy is NO_CACHE
    assert dense_module._prepare_dense_sample.persist_result is False


def test_prepare_dense_sample_writes_label_and_matching_imagery(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"

    result = dense_module._prepare_dense_sample.fn(
        label_path,
        {"optical": SourceConfig()},
        {"optical": _requirement(url)},
        output,
    )

    assert result == str(output)
    assert requests == [["optical"]]
    with io.read_stack(output, chunks="auto") as sample:
        assert set(sample.gs.groups) == {"label", "optical"}
        assert sample.gs.geobox == anchor.geobox
        np.testing.assert_array_equal(sample["label"]["class"], label["class"])
        assert tuple(sample["optical"].data_vars) == ("red", "nir")
        assert sample["label"]["class"].chunks is not None
        assert sample["optical"].red.chunks is not None


def test_prepare_dense_sample_reuses_a_valid_sample_without_stac(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    sources = {"optical": SourceConfig()}
    requirements = {"optical": _requirement(url)}
    output = tmp_path / "sample.zarr"
    dense_module._prepare_dense_sample.fn(
        label_path, sources, requirements, output
    )

    assert dense_module._prepare_dense_sample.fn(
        label_path, sources, requirements, output
    ) == str(output)
    assert requests == [["optical"]]


def test_prepare_dense_sample_rejects_an_incomplete_existing_sample(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"
    write_stack({"optical": label.rename({"class": "red"})}, output)

    with pytest.raises(ValueError, match="does not match model sources"):
        dense_module._prepare_dense_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": _requirement(url)},
            output,
        )

    assert requests == []
    with io.read_stack(output) as sample:
        assert sample.gs.groups == ("optical",)


def test_validate_dense_sample_rejects_stale_source_variables(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, _ = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"
    optical = raster({"red": np.ones((4, 4), dtype="uint16")}, anchor.geobox)
    write_stack({"label": label, "optical": optical}, output)

    with pytest.raises(ValueError, match="nir"):
        dense_module._validate_dense_sample(
            output, {"optical": _requirement(url)}
        )

    assert requests == []


def test_prepare_dense_sample_requires_label_time(tmp_path, stac_server) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(
        tmp_path / "timeless.tif", anchor, dated=False
    )

    with pytest.raises(ValueError, match="Label raster has no time"):
        dense_module._prepare_dense_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": _requirement(url)},
            tmp_path / "sample.zarr",
        )

    assert requests == []

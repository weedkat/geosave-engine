from pathlib import Path
from uuid import uuid4

import numpy as np
from prefect import Flow
from prefect.cache_policies import NO_CACHE
from prefect.concurrency.asyncio import ConcurrencySlotAcquisitionError
import pytest

import geosave_engine.workflow as workflow
from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow import flows, tasks
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement
from geosave_engine.workflow.tasks.save import write_stack


def _write_label(path, anchor, *, day=15, dated=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = raster(
        {"class": np.full((4, 4), day, dtype="uint8")}, anchor.geobox
    )
    if dated:
        label = label.assign_coords(
            time=np.datetime64(f"2025-01-{day:02d}T12:00:00")
        )
    return label, io.geotiff.write_cog(label, path)


def _save_spec(tmp_path, sources):
    return ModelSpec(schema_version=2, sources=sources).save(
        tmp_path / "model_spec.yaml"
    )


def _requirement(url):
    return RasterRequirement(
        variables=("red", "nir"),
        collection="optical",
        endpoints=(url,),
        require_crs=True,
    )


def _acquisition_spec(tmp_path, url):
    return _save_spec(tmp_path, {"optical": _requirement(url)})


def test_dense_exposes_only_prepare_and_validate_sample():
    assert hasattr(workflow, "dense")
    assert isinstance(workflow.dense.prepare, Flow)
    assert callable(workflow.dense.validate_sample)
    assert not hasattr(flows, "prepare_training")
    assert not hasattr(tasks, "write_sample")
    assert not hasattr(tasks, "prepare_sample")


@pytest.mark.slow
def test_prepare_writes_many_samples_and_resumes(
    tmp_path, stac_server, prefect_server
):
    url, requests, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "train" / "b.tif", anchor, day=2)
    _write_label(labels / "train" / "a.tif", anchor, day=1)
    output = tmp_path / "prepared"
    arguments = {
        "labels": str(labels),
        "sources": {"optical": {"query": {}, "load": {}}},
        "output": str(output),
        "spec": str(_acquisition_spec(tmp_path, url)),
    }

    result = workflow.dense.prepare(**arguments)

    assert result == str(output / "manifest.parquet")
    for sample in ("a", "b"):
        with io.read_stack(output / "samples/train" / f"{sample}.zarr") as stack:
            assert set(stack.gs.groups) == {"label", "optical"}
            assert stack.gs.geobox == anchor.geobox
    manifest = io.read_vector(result)
    assert manifest.gdf.sample_id.tolist() == ["train/a.tif", "train/b.tif"]
    assert [Path(path) for path in manifest.gdf.path] == [
        (output / "samples/train/a.zarr").resolve(),
        (output / "samples/train/b.zarr").resolve(),
    ]
    assert requests == [["optical"], ["optical"]]

    assert workflow.dense.prepare(**arguments) == result
    assert requests == [["optical"], ["optical"]]


def test_prepare_requires_labels(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    spec = _save_spec(
        tmp_path, {"optical": RasterRequirement(variables=("red",))}
    )

    with pytest.raises(ValueError, match="No labels found"):
        workflow.dense.prepare.fn(
            labels=str(labels),
            sources={"optical": {}},
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_prepare_requires_unique_sample_paths(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.tif").touch()
    (labels / "a.tiff").touch()
    spec = _save_spec(
        tmp_path, {"optical": RasterRequirement(variables=("red",))}
    )

    with pytest.raises(ValueError, match="unique sample paths"):
        workflow.dense.prepare.fn(
            labels=str(labels),
            pattern="**/*.tif*",
            sources={"optical": {}},
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


@pytest.mark.parametrize("sources", [{}, {"optical": {}, "extra": {}}])
def test_prepare_requires_exact_source_bindings(tmp_path, sources):
    spec = _save_spec(
        tmp_path, {"optical": RasterRequirement(variables=("red",))}
    )

    with pytest.raises(ValueError, match="must match model sources"):
        workflow.dense.prepare.fn(
            labels=str(tmp_path / "labels"),
            sources=sources,
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_prepare_reserves_the_label_group(tmp_path):
    spec = _save_spec(
        tmp_path, {"label": RasterRequirement(variables=("class",))}
    )

    with pytest.raises(ValueError, match="label.*reserved"):
        workflow.dense.prepare.fn(
            labels=str(tmp_path / "labels"),
            sources={"label": {}},
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


@pytest.mark.slow
def test_prepare_keeps_samples_without_replacing_manifest(
    tmp_path, stac_server, prefect_server
):
    url, _, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "a.tif", anchor, day=1)
    _write_label(labels / "b.tif", anchor, day=2, dated=False)
    output = tmp_path / "prepared"
    output.mkdir()
    manifest = output / "manifest.parquet"
    GeoVector.from_anchor(anchor, sample_id="old").to_geoparquet(manifest)

    with pytest.raises(ValueError, match="Label raster has no time"):
        workflow.dense.prepare(
            labels=str(labels),
            sources={"optical": {}},
            output=str(output),
            spec=str(_acquisition_spec(tmp_path, url)),
        )

    assert (output / "samples/a.zarr").is_dir()
    assert io.read_vector(manifest).gdf.sample_id.tolist() == ["old"]


def test_prepare_sample_writes_label_and_matching_imagery(tmp_path, stac_server):
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"

    result = workflow.dense._prepare_sample.fn(
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


@pytest.mark.slow
def test_prepare_sample_requires_configured_source_limit(
    tmp_path, stac_server, prefect_server
):
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"

    with pytest.raises(ConcurrencySlotAcquisitionError):
        workflow.dense._prepare_sample.fn(
            label_path,
            {"optical": SourceConfig(concurrency=f"missing-{uuid4()}")},
            {"optical": _requirement(url)},
            output,
        )

    assert requests == []
    assert not output.exists()


def test_prepare_sample_reuses_a_valid_sample_without_stac(tmp_path, stac_server):
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    sources = {"optical": SourceConfig()}
    requirements = {"optical": _requirement(url)}
    output = tmp_path / "sample.zarr"
    workflow.dense._prepare_sample.fn(label_path, sources, requirements, output)

    assert workflow.dense._prepare_sample.fn(
        label_path, sources, requirements, output
    ) == str(output)
    assert requests == [["optical"]]


def test_prepare_sample_rejects_an_incomplete_existing_sample(
    tmp_path, stac_server
):
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"
    write_stack({"optical": label.rename({"class": "red"})}, output)

    with pytest.raises(ValueError, match="does not match model sources"):
        workflow.dense._prepare_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": _requirement(url)},
            output,
        )

    assert requests == []
    with io.read_stack(output) as sample:
        assert sample.gs.groups == ("optical",)


def test_validate_sample_rejects_an_incomplete_sample(tmp_path, raw):
    output = tmp_path / "sample.zarr"
    write_stack({"optical": raw["optical"]}, output)

    with pytest.raises(ValueError, match="does not match model sources"):
        workflow.dense.validate_sample(
            output, {"optical": RasterRequirement(variables=("red",))}
        )


def test_prepare_sample_rejects_stale_source_variables(tmp_path, stac_server):
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample.zarr"
    optical = raster({"red": np.ones((4, 4), dtype="uint16")}, anchor.geobox)
    write_stack({"label": label, "optical": optical}, output)

    with pytest.raises(ValueError, match="nir"):
        workflow.dense._prepare_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": _requirement(url)},
            output,
        )

    assert requests == []


def test_prepare_sample_requires_label_time(tmp_path, stac_server):
    url, requests, anchor = stac_server
    _, label_path = _write_label(
        tmp_path / "timeless.tif", anchor, dated=False
    )

    with pytest.raises(ValueError, match="Label raster has no time"):
        workflow.dense._prepare_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": _requirement(url)},
            tmp_path / "sample.zarr",
        )

    assert requests == []


def test_prepare_sample_is_not_cached_or_persisted():
    assert workflow.dense._prepare_sample.cache_policy is NO_CACHE
    assert workflow.dense._prepare_sample.persist_result is False

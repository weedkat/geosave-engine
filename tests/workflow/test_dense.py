from pathlib import Path
import numpy as np
from prefect import Flow
import pytest

import geosave_engine.workflow as workflow
from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow import flows, tasks
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement


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


def test_dense_exposes_only_prepare():
    assert hasattr(workflow, "dense")
    assert isinstance(workflow.dense.prepare, Flow)
    assert not hasattr(workflow.dense, "validate_sample")
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

    assert isinstance(result, str)
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

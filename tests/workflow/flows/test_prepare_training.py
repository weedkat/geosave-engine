from pathlib import Path

import numpy as np
from prefect import Flow
import pytest

from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.flows import prepare_training
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement


def write_label(path, anchor, *, day, dated=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = raster(
        {"class": np.full((4, 4), day, dtype="uint8")}, anchor.geobox
    )
    if dated:
        label = label.assign_coords(time=np.datetime64(f"2025-01-{day:02d}"))
    return io.geotiff.write_cog(label, path)


def save_spec(tmp_path, sources):
    return ModelSpec(schema_version=2, sources=sources).save(
        tmp_path / "model_spec.yaml"
    )


def acquisition_spec(tmp_path, url):
    return save_spec(
        tmp_path,
        {
            "optical": RasterRequirement(
                variables=("red", "nir"),
                collection="optical",
                endpoints=(url,),
                require_crs=True,
            )
        },
    )


@pytest.mark.slow
def test_prepare_training_writes_many_samples_and_resumes(
    tmp_path, stac_server, prefect_server
):
    url, requests, anchor = stac_server
    labels = tmp_path / "labels"
    write_label(labels / "train" / "b.tif", anchor, day=2)
    write_label(labels / "train" / "a.tif", anchor, day=1)
    output = tmp_path / "prepared"
    arguments = {
        "labels": str(labels),
        "sources": {"optical": {"query": {}, "load": {}}},
        "output": str(output),
        "spec": str(acquisition_spec(tmp_path, url)),
    }

    result = prepare_training(**arguments)

    assert isinstance(prepare_training, Flow)
    assert result == str(output / "manifest.parquet")
    for sample in ("a", "b"):
        with io.read_stack(output / "samples/train" / f"{sample}.zarr") as stack:
            assert set(stack.gs.groups) == {"label", "optical"}
            assert stack.gs.geobox == anchor.geobox
    catalog = io.read_vector(result)
    assert catalog.gdf.sample_id.tolist() == ["train/a.tif", "train/b.tif"]
    assert [Path(path) for path in catalog.gdf.path] == [
        (output / "samples/train/a.zarr").resolve(),
        (output / "samples/train/b.zarr").resolve(),
    ]
    assert requests == [["optical"], ["optical"]]

    assert prepare_training(**arguments) == result
    assert requests == [["optical"], ["optical"]]


def test_prepare_training_requires_labels(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    spec = save_spec(
        tmp_path, {"optical": RasterRequirement(variables=("red",))}
    )

    with pytest.raises(ValueError, match="No labels found"):
        prepare_training.fn(
            labels=str(labels),
            sources={"optical": {}},
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_prepare_training_requires_unique_sample_paths(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.tif").touch()
    (labels / "a.tiff").touch()
    spec = save_spec(
        tmp_path, {"optical": RasterRequirement(variables=("red",))}
    )

    with pytest.raises(ValueError, match="unique sample paths"):
        prepare_training.fn(
            labels=str(labels),
            pattern="**/*.tif*",
            sources={"optical": {}},
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


@pytest.mark.parametrize("sources", [{}, {"optical": {}, "extra": {}}])
def test_prepare_training_requires_exact_source_bindings(tmp_path, sources):
    spec = save_spec(
        tmp_path, {"optical": RasterRequirement(variables=("red",))}
    )

    with pytest.raises(ValueError, match="must match model sources"):
        prepare_training.fn(
            labels=str(tmp_path / "labels"),
            sources=sources,
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_prepare_training_reserves_the_label_group(tmp_path):
    spec = save_spec(
        tmp_path, {"label": RasterRequirement(variables=("class",))}
    )

    with pytest.raises(ValueError, match="label.*reserved"):
        prepare_training.fn(
            labels=str(tmp_path / "labels"),
            sources={"label": {}},
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


@pytest.mark.slow
def test_prepare_training_keeps_samples_without_replacing_manifest(
    tmp_path, stac_server, prefect_server
):
    url, _, anchor = stac_server
    labels = tmp_path / "labels"
    write_label(labels / "a.tif", anchor, day=1)
    write_label(labels / "b.tif", anchor, day=2, dated=False)
    output = tmp_path / "prepared"
    output.mkdir()
    manifest = output / "manifest.parquet"
    GeoVector.from_anchor(anchor, sample_id="old").to_geoparquet(manifest)

    with pytest.raises(ValueError, match="Label raster has no time"):
        prepare_training(
            labels=str(labels),
            sources={"optical": {}},
            output=str(output),
            spec=str(acquisition_spec(tmp_path, url)),
        )

    assert (output / "samples/a.zarr").is_dir()
    assert io.read_vector(manifest).gdf.sample_id.tolist() == ["old"]

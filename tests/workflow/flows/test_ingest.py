from pathlib import Path

import numpy as np
from prefect import Flow
import pytest

from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.flows import ingest
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement
from geosave_engine.workflow.tasks import ingest_sample, save_catalog


def save_spec(tmp_path, sources=None):
    return ModelSpec(
        schema_version=2,
        sources=sources
        if sources is not None
        else {"optical": RasterRequirement(variables=("red", "nir"))},
    ).save(tmp_path / "model_spec.yaml")


def run_preflight(tmp_path, labels, *, pattern="**/*.tif"):
    return ingest.fn(
        labels=str(labels),
        pattern=pattern,
        sources={"optical": {}},
        output=str(tmp_path / "prepared"),
        spec=str(save_spec(tmp_path)),
    )


def write_label(path, anchor, *, day, dated=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = raster(
        {"class": np.full((4, 4), day, dtype="uint8")}, anchor.geobox
    )
    if dated:
        label = label.assign_coords(time=np.datetime64(f"2025-01-{day:02d}"))
    return io.geotiff.write_cog(label, path)


def acquisition_spec(tmp_path, url):
    return save_spec(
        tmp_path,
        sources={
            "optical": RasterRequirement(
                variables=("red", "nir"),
                collection="optical",
                endpoints=(url,),
                require_crs=True,
            )
        },
    )


@pytest.fixture
def reject_task_submission(monkeypatch):
    def unexpected_submission(*args, **kwargs):
        raise AssertionError("task submitted before flow validation completed")

    monkeypatch.setattr(ingest_sample, "submit", unexpected_submission)
    monkeypatch.setattr(save_catalog, "submit", unexpected_submission)


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_ingest_rejects_invalid_label_root(
    tmp_path, reject_task_submission, kind
):
    root = tmp_path / "labels"
    if kind == "file":
        root.write_text("not a directory")

    with pytest.raises(ValueError, match="label root.*directory"):
        run_preflight(tmp_path, root)


def test_ingest_rejects_absolute_glob(tmp_path, reject_task_submission):
    root = tmp_path / "labels"
    root.mkdir()

    with pytest.raises(ValueError, match="pattern.*relative"):
        run_preflight(tmp_path, root, pattern="/tmp/*.tif")


def test_ingest_rejects_parent_traversing_glob(
    tmp_path, reject_task_submission
):
    root = tmp_path / "labels"
    root.mkdir()

    with pytest.raises(ValueError, match=r"pattern.*\.\."):
        run_preflight(tmp_path, root, pattern="../*.tif")


def test_ingest_rejects_glob_without_matches(tmp_path, reject_task_submission):
    root = tmp_path / "labels"
    root.mkdir()

    with pytest.raises(ValueError, match="No labels match"):
        run_preflight(tmp_path, root)


def test_ingest_rejects_directory_matched_as_label(
    tmp_path, reject_task_submission
):
    root = tmp_path / "labels"
    (root / "nested").mkdir(parents=True)

    with pytest.raises(ValueError, match="not a file"):
        run_preflight(tmp_path, root, pattern="**/*")


def test_ingest_rejects_output_collision(tmp_path, reject_task_submission):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "a.tif").touch()
    (root / "a.tiff").touch()

    with pytest.raises(ValueError, match=r"same sample output.*a\.zarr"):
        run_preflight(tmp_path, root, pattern="**/*.tif*")


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ({}, "Source bindings are missing"),
        ({"optical": {}, "extra": {}}, "Unknown source bindings"),
    ],
)
def test_ingest_rejects_source_names_before_task_submission(
    tmp_path, reject_task_submission, sources, message
):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "a.tif").touch()

    with pytest.raises(ValueError, match=message):
        ingest.fn(
            labels=str(root),
            sources=sources,
            output=str(tmp_path / "prepared"),
            spec=str(save_spec(tmp_path)),
        )


def test_ingest_requires_a_model_source(tmp_path, reject_task_submission):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "a.tif").touch()

    with pytest.raises(ValueError, match="At least one model source"):
        ingest.fn(
            labels=str(root),
            sources={},
            output=str(tmp_path / "prepared"),
            spec=str(save_spec(tmp_path, sources={})),
        )


def test_ingest_reserves_the_label_group(tmp_path, reject_task_submission):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "a.tif").touch()
    spec = save_spec(
        tmp_path,
        sources={"label": RasterRequirement(variables=("class",))},
    )

    with pytest.raises(ValueError, match="label.*reserved"):
        ingest.fn(
            labels=str(root),
            sources={"label": {}},
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


@pytest.mark.parametrize("kind", ["uri", "file"])
def test_ingest_requires_a_local_output_directory(
    tmp_path, reject_task_submission, kind
):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "a.tif").touch()
    output = "s3://bucket/prepared" if kind == "uri" else tmp_path / "prepared"
    if kind == "file":
        output.write_text("not a directory")

    with pytest.raises(ValueError, match="output.*local directory"):
        ingest.fn(
            labels=str(root),
            sources={"optical": {}},
            output=str(output),
            spec=str(save_spec(tmp_path)),
        )


def test_ingest_validates_nested_sources_before_loading_spec(
    tmp_path, reject_task_submission
):
    with pytest.raises(ValueError, match="unexpected"):
        ingest.fn(
            labels=str(tmp_path / "missing-labels"),
            sources={"optical": {"query": {"unexpected": True}}},
            output=str(tmp_path / "prepared"),
            spec=str(tmp_path / "missing.yaml"),
        )


@pytest.mark.slow
def test_ingest_prepares_many_labels_and_resumes(
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

    result = ingest(**arguments)

    assert isinstance(ingest, Flow)
    assert result == str(output / "manifest.parquet")
    for sample in ("a", "b"):
        with io.read_stack(output / "samples" / "train" / f"{sample}.zarr") as tree:
            assert set(tree.gs.groups) == {"label", "optical"}
            assert tree.gs.geobox == anchor.geobox
    catalog = io.read_vector(result)
    assert catalog.gdf.sample_id.tolist() == ["train/a.tif", "train/b.tif"]
    assert [Path(path) for path in catalog.gdf.path] == [
        (output / "samples" / "train" / "a.zarr").resolve(),
        (output / "samples" / "train" / "b.zarr").resolve(),
    ]
    assert requests == [["optical"], ["optical"]]

    assert ingest(**arguments) == result
    assert requests == [["optical"], ["optical"]]


@pytest.mark.slow
def test_ingest_keeps_completed_samples_without_replacing_manifest(
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
        ingest(
            labels=str(labels),
            sources={"optical": {}},
            output=str(output),
            spec=str(acquisition_spec(tmp_path, url)),
        )

    assert (output / "samples" / "a.zarr").is_dir()
    assert io.read_vector(manifest).gdf.sample_id.tolist() == ["old"]

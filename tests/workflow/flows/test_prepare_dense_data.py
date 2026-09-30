from importlib import import_module
from pathlib import Path
from threading import Barrier, Event, Lock
from time import sleep

import numpy as np
import pandas as pd
import geopandas as gpd
from prefect import Flow, Task
from pydantic import ValidationError
import pytest

from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.model_spec import ModelSpec, RasterRequirement, StacRecipe
from geosave_engine.workflow import flows, tasks
from geosave_engine.workflow.flows import prepare_dense_data
from geosave_engine.workflow.tasks import prepare_dense_sample
from geosave_engine.workflow.tasks.manifest import find_labels, sample_path

flow_module = import_module("geosave_engine.workflow.flows.prepare_dense_data")


def _write_label(path, anchor, *, day=15, dated=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = raster({"class": np.full((4, 4), day, dtype="uint8")}, anchor.geobox)
    if dated:
        label = label.assign_coords(time=np.datetime64(f"2025-01-{day:02d}T12:00:00"))
    return label, io.geotiff.write_cog(label, path)


def _save_spec(tmp_path, rasters):
    return ModelSpec(schema_version=2, rasters=rasters).save(
        tmp_path / "model_spec.yaml"
    )


def _requirement(url=None):
    return RasterRequirement(
        variables=("red", "nir"),
        stac=(
            StacRecipe.model_validate(
                {
                    "collection": "optical",
                    "endpoints": (url,),
                    "load": {"bands": ["red", "nir"]},
                }
            )
            if url is not None
            else None
        ),
        require_crs=url is not None,
    )


def _touch_labels(root: Path, count: int) -> None:
    root.mkdir()
    for index in range(count):
        (root / f"{index}.tif").touch()


def test_layers_export_only_supported_operations() -> None:
    assert flows.__all__ == ["ingest", "prepare_dense_data"]
    assert tasks.__all__ == ["prepare_dense_sample"]
    assert isinstance(flows.ingest, Flow)
    assert isinstance(flows.prepare_dense_data, Flow)
    assert isinstance(tasks.prepare_dense_sample, Task)
    assert prepare_dense_data is flows.prepare_dense_data
    assert prepare_dense_sample is tasks.prepare_dense_sample
    for helper in (
        "SampleFormat",
        "find_labels",
        "open_sample",
        "read_sample_metadata",
        "sample_path",
        "write_manifest",
        "write_sample",
    ):
        assert not hasattr(tasks, helper)


def test_prepare_dense_data_requires_labels(tmp_path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    spec = _save_spec(tmp_path, {"optical": _requirement("https://stac.test")})

    with pytest.raises(ValueError, match="No labels found"):
        flow_module.prepare_dense_data.fn(
            labels=str(labels),
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_find_labels_preserves_tree_and_removes_only_the_final_suffix(
    tmp_path,
) -> None:
    labels = tmp_path / "labels"
    _touch_labels(labels, 0)
    path = labels / "train" / "region" / "tile.v1.tif"
    path.parent.mkdir(parents=True)
    path.touch()

    assert find_labels(labels, "**/*.tif") == {
        "train/region/tile.v1": path
    }


@pytest.mark.parametrize(
    ("format", "relative"),
    [
        ("geotiff", "train/region/tile.v1"),
        ("zarr", "train/region/tile.v1.zarr"),
    ],
)
def test_sample_path_preserves_the_suffix_free_identity(
    tmp_path, format, relative
) -> None:
    assert sample_path(
        tmp_path / "prepared", "train/region/tile.v1", format
    ) == tmp_path / "prepared" / relative


def test_prepare_dense_data_requires_unique_sample_paths(tmp_path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.tif").touch()
    (labels / "a.tiff").touch()
    spec = _save_spec(tmp_path, {"optical": _requirement("https://stac.test")})

    with pytest.raises(ValueError, match="unique sample paths"):
        flow_module.prepare_dense_data.fn(
            labels=str(labels),
            output=str(tmp_path / "prepared"),
            spec=str(spec),
            pattern="**/*.tif*",
        )


def test_prepare_dense_data_requires_all_stac_recipes_before_submission(
    tmp_path, monkeypatch
):
    spec = _save_spec(
        tmp_path,
        {
            "optical": _requirement(),
            "elevation": RasterRequirement(variables=("dem",)),
        },
    )

    def unexpected_submission(*args, **kwargs):
        pytest.fail(f"task submitted before validation: {args}, {kwargs}")

    monkeypatch.setattr(
        flow_module.prepare_dense_sample, "submit", unexpected_submission
    )

    with pytest.raises(ValueError, match="optical.*elevation"):
        flow_module.prepare_dense_data.fn(
            labels=str(tmp_path / "labels"),
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_prepare_dense_data_reserves_the_label_group(tmp_path) -> None:
    spec = _save_spec(tmp_path, {"label": RasterRequirement(variables=("class",))})

    with pytest.raises(ValueError, match="label.*reserved"):
        flow_module.prepare_dense_data.fn(
            labels=str(tmp_path / "labels"),
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_prepare_dense_data_rejects_zero_concurrency_before_submission(
    tmp_path, monkeypatch
) -> None:
    labels = tmp_path / "labels"
    _touch_labels(labels, 1)
    spec = _save_spec(tmp_path, {"optical": _requirement("https://stac.test")})

    def unexpected_submission(*args, **kwargs):
        pytest.fail(f"task submitted before validation: {args}, {kwargs}")

    monkeypatch.setattr(
        flow_module.prepare_dense_sample, "submit", unexpected_submission
    )

    with pytest.raises(ValidationError, match="greater than 0"):
        flow_module.prepare_dense_data.fn(
            labels=str(labels),
            output=str(tmp_path / "prepared"),
            spec=str(spec),
            max_concurrency=0,
        )


def test_invalid_metadata_prevents_submission_and_preserves_manifest(
    tmp_path, monkeypatch
) -> None:
    labels = tmp_path / "labels"
    _touch_labels(labels, 1)
    metadata = tmp_path / "samples.csv"
    pd.DataFrame({"label_path": ["labels/missing.tif"]}).to_csv(
        metadata, index=False
    )
    output = tmp_path / "prepared"
    output.mkdir()
    manifest = output / "manifest.parquet"
    manifest.write_bytes(b"existing manifest")
    spec = _save_spec(tmp_path, {"optical": _requirement("https://stac.test")})

    def unexpected_submission(*args, **kwargs):
        pytest.fail(f"task submitted before metadata validation: {args}, {kwargs}")

    monkeypatch.setattr(
        flow_module.prepare_dense_sample, "submit", unexpected_submission
    )

    with pytest.raises(ValueError, match="missing.*0.tif"):
        flow_module.prepare_dense_data.fn(
            labels=str(labels),
            output=str(output),
            spec=str(spec),
            metadata=str(metadata),
        )

    assert manifest.read_bytes() == b"existing manifest"


@pytest.mark.slow
@pytest.mark.parametrize(("limit", "expected_peak"), [(1, 1), (2, 2)])
def test_prepare_dense_data_bounds_active_samples(
    tmp_path, prefect_server, monkeypatch, limit, expected_peak
) -> None:
    labels = tmp_path / "labels"
    _touch_labels(labels, 3)
    spec = _save_spec(
        tmp_path,
        {
            "optical": _requirement("https://stac.test"),
            "elevation": RasterRequirement(
                variables=("dem",),
                stac=StacRecipe.model_validate(
                    {"collection": "dem", "endpoints": ("https://stac.test",)}
                ),
            ),
        },
    )
    lock = Lock()
    barrier = Barrier(2) if limit == 2 else None
    active = 0
    peak = 0

    def prepare(label, model, output, *, format, write_options):
        nonlocal active, peak
        assert set(model.rasters) == {"optical", "elevation"}
        assert format == "geotiff"
        assert write_options is None
        with lock:
            active += 1
            peak = max(peak, active)
        if barrier is not None and Path(label).name in {"0.tif", "1.tif"}:
            barrier.wait(timeout=5)
        sleep(0.05)
        Path(output).mkdir(parents=True)
        with lock:
            active -= 1
        return str(output)

    def manifest(samples, destination, *, format, metadata):
        assert format == "geotiff"
        assert metadata == {key: {} for key in samples}
        return str(destination)

    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", prepare)
    monkeypatch.setattr(flow_module, "write_manifest", manifest)

    result = flow_module.prepare_dense_data(
        labels=str(labels),
        output=str(tmp_path / "prepared"),
        spec=str(spec),
        max_concurrency=limit,
    )

    assert result == str(tmp_path / "prepared/manifest.parquet")
    assert peak == expected_peak


@pytest.mark.slow
def test_prepare_dense_data_stops_submitting_after_failure(
    tmp_path, prefect_server, monkeypatch
) -> None:
    labels = tmp_path / "labels"
    _touch_labels(labels, 4)
    output = tmp_path / "prepared"
    output.mkdir()
    manifest = output / "manifest.parquet"
    GeoVector.from_anchor(workflow_anchor(), sample_id="old").to_geoparquet(manifest)
    original = manifest.read_bytes()
    spec = _save_spec(tmp_path, {"optical": _requirement("https://stac.test")})
    second_started = Event()
    started: list[str] = []

    def prepare(label, model, destination, *, format, write_options):
        assert format == "geotiff"
        assert write_options is None
        name = Path(label).name
        started.append(name)
        if name == "0.tif":
            second_started.wait(timeout=5)
            raise RuntimeError("ingestion failed")
        second_started.set()
        Path(destination).mkdir(parents=True)
        sleep(0.05)
        return str(destination)

    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", prepare)

    with pytest.raises(RuntimeError, match="ingestion failed"):
        flow_module.prepare_dense_data(
            labels=str(labels),
            output=str(output),
            spec=str(spec),
            max_concurrency=2,
        )

    assert set(started) == {"0.tif", "1.tif"}
    assert (output / "1").is_dir()
    assert manifest.read_bytes() == original


def workflow_anchor():
    from geosave_engine.geodata.core import GeoAnchor

    return GeoAnchor.from_coordinates(
        45,
        12,
        4,
        10,
        crs="EPSG:32633",
        timespan="2025-01",
    )


@pytest.mark.slow
def test_prepare_dense_data_uses_model_recipes_and_resumes(
    tmp_path, stac_server, prefect_server
) -> None:
    url, requests, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "train/b.tif", anchor, day=2)
    _write_label(labels / "train/a.tif", anchor, day=1)
    metadata = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "label_path": ["labels/train/a.tif", "labels/train/b.tif"],
            "split": ["train", "validation"],
        }
    ).to_csv(metadata, index=False)
    output = tmp_path / "prepared"
    arguments = {
        "labels": str(labels),
        "output": str(output),
        "spec": str(_save_spec(tmp_path, {"optical": _requirement(url)})),
        "metadata": str(metadata),
    }

    result = flow_module.prepare_dense_data(**arguments)

    assert result == str(output / "manifest.parquet")
    assert requests == [["optical"], ["optical"]]
    assert flow_module.prepare_dense_data(**arguments) == result
    assert requests == [["optical"], ["optical"]]
    manifest = gpd.read_parquet(result)
    assert manifest.path.tolist() == ["train/a", "train/b"]
    assert manifest["format"].tolist() == ["geotiff", "geotiff"]
    assert manifest.split.tolist() == ["train", "validation"]
    for sample in (output / "train/a", output / "train/b"):
        assert sorted(path.name for path in sample.iterdir()) == [
            "label.tif",
            "optical.tif",
        ]


@pytest.mark.slow
def test_prepare_dense_data_supports_explicit_zarr(
    tmp_path, stac_server, prefect_server
) -> None:
    url, requests, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "val/region/tile.v1.tif", anchor)
    output = tmp_path / "prepared"

    result = flow_module.prepare_dense_data(
        labels=str(labels),
        output=str(output),
        spec=str(_save_spec(tmp_path, {"optical": _requirement(url)})),
        format="zarr",
    )

    assert result == str(output / "manifest.parquet")
    assert requests == [["optical"]]
    assert (output / "val/region/tile.v1.zarr").is_dir()
    manifest = gpd.read_parquet(result)
    assert manifest.path.tolist() == ["val/region/tile.v1.zarr"]
    assert manifest["format"].tolist() == ["zarr"]


@pytest.mark.slow
def test_prepare_dense_data_keeps_samples_without_replacing_manifest(
    tmp_path, stac_server, prefect_server
) -> None:
    url, _, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "a.tif", anchor, day=1)
    _write_label(labels / "b.tif", anchor, day=2, dated=False)
    output = tmp_path / "prepared"
    output.mkdir()
    manifest = output / "manifest.parquet"
    GeoVector.from_anchor(anchor, sample_id="old").to_geoparquet(manifest)

    with pytest.raises(ValueError, match="Label raster has no time"):
        flow_module.prepare_dense_data(
            labels=str(labels),
            output=str(output),
            spec=str(_save_spec(tmp_path, {"optical": _requirement(url)})),
        )

    assert (output / "a").is_dir()
    assert io.read_vector(manifest).gdf.sample_id.tolist() == ["old"]

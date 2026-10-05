from importlib import import_module
from pathlib import Path
from threading import Barrier, Lock
from time import sleep

import numpy as np
import pandas as pd
import geopandas as gpd
from prefect import Flow, Task
from prefect.exceptions import ParameterTypeError
import pytest

from shapely.geometry import Point

from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata import io
from geosave_engine.model.spec import ModelSpec, RasterRequirement, StacRecipe
from geosave_engine.workflow import flows, tasks
from geosave_engine.workflow.flows import prepare_dense_data
from geosave_engine.workflow.tasks import prepare_dense_sample

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
        "sample_assets",
        "write_manifest",
        "write_sample",
    ):
        assert not hasattr(tasks, helper)


def test_active_files_do_not_reference_removed_workflow_packages() -> None:
    root = Path(__file__).parents[3]
    paths = [
        root / "README.md",
        *(root / "src/geosave_engine").rglob("*.py"),
        *(root / "tests").rglob("*.py"),
        *(root / "docs/guides").rglob("*.md"),
    ]
    removed = (
        ".".join(("geosave_engine", "workflow", "ingestion")),
        ".".join(("geosave_engine", "workflow", "training_data")),
    )
    references = {
        str(path.relative_to(root)): package
        for path in paths
        for package in removed
        if package in path.read_text()
    }

    assert references == {}


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


def test_invalid_table_ids_leave_existing_outputs_untouched(tmp_path, monkeypatch):
    labels = GeoVector.from_geometry(
        Point(0, 0),
        properties={
            "id": ".",
            "assets": {"label": {"href": "label.tif"}},
        },
    ).gs.to_geoparquet(tmp_path / "labels.parquet")
    output = tmp_path / "prepared"
    output.mkdir()
    manifest = output / "manifest.parquet"
    GeoVector.from_geometry(
        Point(0, 0),
        properties={
            "id": "scene",
            "assets": {"image": {"href": "scene/image.tif"}},
        },
    ).gs.to_geoparquet(manifest)
    original_manifest = manifest.read_bytes()
    scene = output / "scene"
    scene.mkdir()
    asset = scene / "image.tif"
    asset.write_bytes(b"existing scene")
    spec = _save_spec(tmp_path, {"optical": _requirement()})

    def submit(*args, **kwargs):
        pytest.fail("invalid IDs must fail before submitting tasks")

    monkeypatch.setattr(flow_module.prepare_dense_sample, "submit", submit)
    with pytest.raises(ValueError, match="canonical"):
        flow_module.prepare_dense_data.fn(
            labels=str(labels), output=str(output), spec=str(spec)
        )
    assert manifest.read_bytes() == original_manifest
    assert asset.read_bytes() == b"existing scene"


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


def test_prepare_dense_data_reserves_the_label_group(tmp_path) -> None:
    spec = _save_spec(tmp_path, {"label": RasterRequirement(variables=("class",))})

    with pytest.raises(ValueError, match="label.*reserved"):
        flow_module.prepare_dense_data.fn(
            labels=str(tmp_path / "labels"),
            output=str(tmp_path / "prepared"),
            spec=str(spec),
        )


def test_prepare_dense_data_parameters_reject_zero_concurrency() -> None:
    with pytest.raises(ParameterTypeError, match="greater than 0"):
        flow_module.prepare_dense_data.validate_parameters(
            {
                "labels": "labels",
                "output": "prepared",
                "spec": "model_spec.yaml",
                "max_concurrency": 0,
            }
        )


def _labels(root: Path, names: tuple[str, ...] = ("a", "b", "c")) -> Path:
    """Write one dated label raster per name and return their directory."""
    from odc.geo.geobox import GeoBox

    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    for day, name in enumerate(names, start=1):
        label = raster(
            {"class": np.full((4, 4), day, dtype="uint8")}, grid
        ).assign_coords(time=np.datetime64(f"2025-01-{day:02d}T12:00:00"))
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        io.geotiff.write_cog(label, root / f"{name}.tif")
    return root


def _prepare(calls: list[str], *, failing: str | None = None, layers=("optical",)):
    """Stand in for the sample task: write a real sample without any catalog."""
    from geosave_engine.geodata import read_stack
    from geosave_engine.workflow.tasks.sample import write_sample

    def prepare(
        label_path, spec, output, *, sample_id, properties, format, write_options
    ):
        calls.append(sample_id)
        if sample_id == failing:
            raise RuntimeError(f"no imagery for {sample_id}")
        with io.read_raster(label_path) as label:
            imagery = raster(
                {"red": np.ones((4, 4), "uint16"), "nir": np.ones((4, 4), "uint16")},
                label.gs.geobox,
            ).assign_coords(time=label.time)
            write_sample(
                {"label": label, **dict.fromkeys(layers, imagery)},
                output,
                format=format,
            )
        return GeoVector.from_xarray(
            read_stack(output), id=sample_id, properties=properties
        )

    return prepare


def _run(tmp_path, labels, **options):
    spec = _save_spec(tmp_path, {"optical": _requirement("https://stac.test")})
    return flow_module.prepare_dense_data(
        labels=str(labels), output=str(tmp_path / "prepared"), spec=str(spec), **options
    )


def test_a_manifest_that_is_not_a_stac_table_stops_the_run(tmp_path) -> None:
    labels = _labels(tmp_path / "labels")
    manifest = tmp_path / "prepared" / "manifest.parquet"
    manifest.parent.mkdir()
    GeoVector.from_geometry(
        Point(0, 0), properties={"sample_id": "old"}
    ).gs.to_geoparquet(manifest)
    spec = _save_spec(tmp_path, {"optical": _requirement("https://stac.test")})

    with pytest.raises(ValueError, match="is not a sample manifest"):
        flow_module.prepare_dense_data.fn(
            labels=str(labels), output=str(tmp_path / "prepared"), spec=str(spec)
        )

    assert sorted(path.name for path in manifest.parent.iterdir()) == [
        "manifest.parquet"
    ]


@pytest.mark.slow
def test_a_label_directory_and_its_table_give_the_same_manifest(
    tmp_path, prefect_server, monkeypatch
) -> None:
    from geosave_engine.workflow.tasks.labels import read_labels

    labels = _labels(tmp_path / "labels", ("north/a", "south/b"))
    table = read_labels(labels).assign(quality=[0.97, None], surveyor=["ana", "ben"])
    table_path = table.gs.to_geoparquet(tmp_path / "labels.parquet")
    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", _prepare([]))

    by_directory = gpd.read_parquet(_run(tmp_path / "one", labels))
    by_table = gpd.read_parquet(_run(tmp_path / "two", table_path))

    assert by_directory.id.tolist() == by_table.id.tolist() == ["north/a", "south/b"]
    assert [sorted(assets) for assets in by_table.assets] == [["label", "optical"]] * 2
    assert by_table.assets[0]["optical"]["href"] == "north/a/optical.tif"
    assert by_table.surveyor.tolist() == ["ana", "ben"]
    assert by_table.quality.tolist()[0] == 0.97 and pd.isna(
        by_table.quality.tolist()[1]
    )
    assert "quality" not in by_directory


@pytest.mark.slow
def test_a_failed_run_keeps_what_finished_and_a_rerun_does_the_rest(
    tmp_path, prefect_server, monkeypatch
) -> None:
    labels = _labels(tmp_path / "labels")
    calls: list[str] = []
    monkeypatch.setattr(
        flow_module.prepare_dense_sample, "fn", _prepare(calls, failing="b")
    )

    with pytest.raises(RuntimeError, match="no imagery for b"):
        _run(tmp_path, labels)

    manifest = tmp_path / "prepared" / "manifest.parquet"
    assert gpd.read_parquet(manifest).id.tolist() == ["a"]
    assert calls == ["a", "b"]

    calls.clear()
    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", _prepare(calls))
    result = _run(tmp_path, labels)

    assert calls == ["b", "c"]
    assert gpd.read_parquet(result).id.tolist() == ["a", "b", "c"]


@pytest.mark.slow
def test_a_recorded_sample_is_reused_and_dropped_labels_leave_the_manifest(
    tmp_path, prefect_server, monkeypatch
) -> None:
    labels = _labels(tmp_path / "labels")
    calls: list[str] = []
    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", _prepare(calls))
    _run(tmp_path, labels)
    calls.clear()
    (labels / "c.tif").unlink()

    result = _run(tmp_path, labels)

    assert calls == []
    assert gpd.read_parquet(result).id.tolist() == ["a", "b"]


@pytest.mark.slow
def test_a_manifest_from_another_spec_stops_the_run(
    tmp_path, prefect_server, monkeypatch
) -> None:
    labels = _labels(tmp_path / "labels")
    calls: list[str] = []
    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", _prepare(calls))
    _run(tmp_path, labels)
    calls.clear()
    wider = _save_spec(
        tmp_path,
        {
            "optical": _requirement("https://stac.test"),
            "radar": _requirement("https://stac.test"),
        },
    )

    with pytest.raises(ValueError, match="does not match model rasters"):
        flow_module.prepare_dense_data(
            labels=str(labels), output=str(tmp_path / "prepared"), spec=str(wider)
        )

    assert calls == []


@pytest.mark.slow
@pytest.mark.parametrize(("limit", "expected_peak"), [(1, 1), (2, 2)])
def test_prepare_dense_data_bounds_active_samples(
    tmp_path, prefect_server, monkeypatch, limit: int, expected_peak: int
) -> None:
    labels = _labels(tmp_path / "labels")
    write = _prepare([])
    lock = Lock()
    barrier = Barrier(2) if limit == 2 else None
    active = 0
    peak = 0

    def prepare(label_path, spec, output, **options):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if barrier is not None and options["sample_id"] in {"a", "b"}:
            barrier.wait(timeout=5)
        sleep(0.05)
        try:
            return write(label_path, spec, output, **options)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", prepare)

    result = _run(tmp_path, labels, max_concurrency=limit)

    assert result == str(tmp_path / "prepared/manifest.parquet")
    assert peak == expected_peak
    assert gpd.read_parquet(result).id.tolist() == ["a", "b", "c"]


@pytest.mark.slow
def test_prepare_dense_data_uses_model_recipes_and_resumes(
    tmp_path, stac_server, prefect_server
) -> None:
    url, requests, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "train/b.tif", anchor, day=2)
    _write_label(labels / "train/a.tif", anchor, day=1)
    output = tmp_path / "prepared"
    arguments = {
        "labels": str(labels),
        "output": str(output),
        "spec": str(_save_spec(tmp_path, {"optical": _requirement(url)})),
    }

    result = flow_module.prepare_dense_data(**arguments)

    assert result == str(output / "manifest.parquet")
    assert requests == [["optical"], ["optical"]]
    assert flow_module.prepare_dense_data(**arguments) == result
    assert requests == [["optical"], ["optical"]]
    manifest = gpd.read_parquet(result)
    assert manifest.id.tolist() == ["train/a", "train/b"]
    assert [assets["optical"]["href"] for assets in manifest.assets] == [
        "train/a/optical.tif",
        "train/b/optical.tif",
    ]
    assert all(len(sources) >= 1 for sources in manifest.sources)
    for sample in (output / "train/a", output / "train/b"):
        assert sorted(path.name for path in sample.iterdir()) == [
            "label.tif",
            "optical.tif",
        ]


@pytest.mark.slow
def test_a_sample_folder_without_a_row_is_registered_not_downloaded(
    tmp_path, stac_server, prefect_server
) -> None:
    url, requests, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "a.tif", anchor, day=1)
    output = tmp_path / "prepared"
    arguments = {
        "labels": str(labels),
        "output": str(output),
        "spec": str(_save_spec(tmp_path, {"optical": _requirement(url)})),
    }
    result = flow_module.prepare_dense_data(**arguments)
    Path(result).unlink()

    assert flow_module.prepare_dense_data(**arguments) == result
    assert requests == [["optical"]]
    assert gpd.read_parquet(result).id.tolist() == ["a"]


@pytest.mark.slow
def test_prepare_dense_data_supports_explicit_zarr(
    tmp_path, stac_server, prefect_server
) -> None:
    url, _, anchor = stac_server
    labels = tmp_path / "labels"
    _write_label(labels / "a.tif", anchor)
    output = tmp_path / "prepared"

    result = flow_module.prepare_dense_data(
        labels=str(labels),
        output=str(output),
        spec=str(_save_spec(tmp_path, {"optical": _requirement(url)})),
        format="zarr",
    )

    manifest = gpd.read_parquet(result)
    assert manifest.assets[0]["optical"]["href"] == "a/optical.zarr"
    assert sorted(path.name for path in (output / "a").iterdir()) == [
        "label.zarr",
        "optical.zarr",
    ]


@pytest.mark.slow
def test_a_failure_still_records_samples_that_finished_beside_it(
    tmp_path, prefect_server, monkeypatch
) -> None:
    labels = _labels(tmp_path / "labels", ("a", "b"))
    write = _prepare([])

    def prepare(label_path, spec, output, **options):
        if options["sample_id"] == "a":
            raise RuntimeError("no imagery for a")
        sleep(0.5)
        return write(label_path, spec, output, **options)

    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", prepare)

    with pytest.raises(RuntimeError, match="no imagery for a"):
        _run(tmp_path, labels, max_concurrency=2)

    manifest = tmp_path / "prepared" / "manifest.parquet"
    assert gpd.read_parquet(manifest).id.tolist() == ["b"]


@pytest.mark.slow
def test_a_rerun_takes_caller_columns_from_the_label_table_as_it_is_now(
    tmp_path, prefect_server, monkeypatch
) -> None:
    from geosave_engine.workflow.tasks.labels import read_labels

    labels = read_labels(_labels(tmp_path / "labels", ("a", "b")))
    table = tmp_path / "labels.parquet"
    labels.assign(quality=[0.1, 0.2]).gs.to_geoparquet(table)
    calls: list[str] = []
    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", _prepare(calls))
    _run(tmp_path, table)
    calls.clear()
    labels.assign(quality=[0.9, 0.2], surveyor=["ana", "ben"]).gs.to_geoparquet(
        table, overwrite=True
    )

    manifest = gpd.read_parquet(_run(tmp_path, table))

    assert calls == []
    assert manifest.quality.tolist() == [0.9, 0.2]
    assert manifest.surveyor.tolist() == ["ana", "ben"]
    assert list(manifest)[-1] == "geometry"


@pytest.mark.slow
def test_the_finished_manifest_follows_label_order_and_is_valid_stac(
    tmp_path, prefect_server, monkeypatch
) -> None:
    import pyarrow.parquet as pq
    from pystac import Item
    from stac_geoparquet.arrow import stac_table_to_items

    labels = _labels(tmp_path / "labels")
    write = _prepare([])

    def prepare(label_path, spec, output, **options):
        sleep({"a": 0.6, "b": 0.3, "c": 0.0}[options["sample_id"]])
        return write(label_path, spec, output, **options)

    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", prepare)

    result = _run(tmp_path, labels, max_concurrency=3)

    assert gpd.read_parquet(result).id.tolist() == ["a", "b", "c"]
    for entry in stac_table_to_items(pq.read_table(result)):
        Item.from_dict(entry).validate()


@pytest.mark.slow
def test_a_manifest_holding_other_bands_stops_the_run(
    tmp_path, prefect_server, monkeypatch
) -> None:
    labels = _labels(tmp_path / "labels")
    calls: list[str] = []
    monkeypatch.setattr(flow_module.prepare_dense_sample, "fn", _prepare(calls))
    _run(tmp_path, labels)
    calls.clear()
    other = RasterRequirement(
        variables=("swir", "blue"),
        stac=StacRecipe.model_validate(
            {"collection": "optical", "endpoints": ("https://stac.test",)}
        ),
    )

    with pytest.raises(ValueError, match="does not match model rasters"):
        flow_module.prepare_dense_data(
            labels=str(labels),
            output=str(tmp_path / "prepared"),
            spec=str(_save_spec(tmp_path, {"optical": other})),
        )

    assert calls == []

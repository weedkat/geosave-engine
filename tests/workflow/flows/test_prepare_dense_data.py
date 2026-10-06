from pathlib import Path

import pytest
from prefect import Flow, Task

from geosave_engine.workflow import flows, tasks
from geosave_engine.workflow.flows import prepare_dense_data
from geosave_engine.workflow.tasks import prepare_dense_sample


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


def test_dense_flow_is_explicitly_pending_without_side_effects(tmp_path):
    with pytest.raises(NotImplementedError, match="Dense sample catalog"):
        prepare_dense_data.fn(
            str(tmp_path / "labels"),
            output=str(tmp_path / "prepared"),
            spec=str(tmp_path / "model_spec.yaml"),
        )
    assert list(tmp_path.iterdir()) == []

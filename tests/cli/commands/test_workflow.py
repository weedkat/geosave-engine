import pytest
from typer.testing import CliRunner

from geosave_engine.cli.commands import workflow
from geosave_engine.cli.main import app

runner = CliRunner()


def test_workflow_help_lists_only_runnable_commands() -> None:
    root = runner.invoke(app, ["--help"])
    group = runner.invoke(app, ["workflow", "--help"])

    assert root.exit_code == 0
    assert "workflow" in root.stdout
    assert group.exit_code == 0
    assert "ingest" in group.stdout
    assert "prepare-dense-data" in group.stdout


def test_ingest_help_lists_its_flow_options() -> None:
    result = runner.invoke(app, ["workflow", "ingest", "--help"])

    assert result.exit_code == 0
    for option in ("--anchor", "--output", "--spec", "--sources"):
        assert option in result.stdout


def test_prepare_dense_data_help_lists_safe_concurrency_default() -> None:
    result = runner.invoke(app, ["workflow", "prepare-dense-data", "--help"])

    assert result.exit_code == 0
    for option in (
        "--labels",
        "--output",
        "--spec",
        "--pattern",
        "--sources",
        "--max-concurrency",
    ):
        assert option in result.stdout
    assert "1" in result.stdout


def test_ingest_parses_structured_options_and_prints_result(
    monkeypatch,
) -> None:
    captured = {}

    def run_flow(**arguments):
        captured.update(arguments)
        return "data/raw.zarr"

    monkeypatch.setattr(workflow, "ingest", run_flow)
    result = runner.invoke(
        app,
        [
            "workflow",
            "ingest",
            "--anchor",
            '{"kind":"raster","path":"data/reference.tif"}',
            "--output",
            "data/raw.zarr",
            "--spec",
            "model_spec.yaml",
            "--sources",
            '{"optical":{"query":{"max_items":8}}}',
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "anchor": {"kind": "raster", "path": "data/reference.tif"},
        "output": "data/raw.zarr",
        "spec": "model_spec.yaml",
        "sources": {"optical": {"query": {"max_items": 8}}},
    }
    assert result.stdout == "data/raw.zarr\n"


def test_prepare_dense_data_forwards_typed_options(monkeypatch) -> None:
    captured = {}

    def run_flow(**arguments):
        captured.update(arguments)
        return "data/prepared/manifest.parquet"

    monkeypatch.setattr(workflow, "prepare_dense_data", run_flow)
    result = runner.invoke(
        app,
        [
            "workflow",
            "prepare-dense-data",
            "--labels",
            "data/labels",
            "--output",
            "data/prepared",
            "--spec",
            "model_spec.yaml",
            "--pattern",
            "train/*.tif",
            "--max-concurrency",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "labels": "data/labels",
        "output": "data/prepared",
        "spec": "model_spec.yaml",
        "pattern": "train/*.tif",
        "sources": None,
        "max_concurrency": 2,
    }
    assert result.stdout == "data/prepared/manifest.parquet\n"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--anchor", "{"),
        ("--anchor", '{"kind":"unknown"}'),
        ("--sources", "[]"),
        ("--sources", '{"optical":{"unexpected":true}}'),
    ],
)
def test_ingest_rejects_invalid_json_before_invoking_flow(
    monkeypatch, option, value
) -> None:
    def unexpected_flow(**arguments):
        pytest.fail(f"flow invoked with invalid input: {arguments}")

    monkeypatch.setattr(workflow, "ingest", unexpected_flow)
    arguments = [
        "workflow",
        "ingest",
        "--anchor",
        '{"kind":"raster","path":"reference.tif"}',
        "--output",
        "raw.zarr",
        "--spec",
        "model_spec.yaml",
    ]
    arguments.extend([option, value])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 2


def test_prepare_dense_data_rejects_zero_concurrency_before_flow(
    monkeypatch,
) -> None:
    def unexpected_flow(**arguments):
        pytest.fail(f"flow invoked with invalid input: {arguments}")

    monkeypatch.setattr(workflow, "prepare_dense_data", unexpected_flow)
    result = runner.invoke(
        app,
        [
            "workflow",
            "prepare-dense-data",
            "--labels",
            "labels",
            "--output",
            "prepared",
            "--spec",
            "model_spec.yaml",
            "--max-concurrency",
            "0",
        ],
    )

    assert result.exit_code == 2


def test_flow_failure_exits_without_printing_success_path(monkeypatch) -> None:
    def fail(**arguments):
        raise RuntimeError("ingestion failed")

    monkeypatch.setattr(workflow, "ingest", fail)
    result = runner.invoke(
        app,
        [
            "workflow",
            "ingest",
            "--anchor",
            '{"kind":"raster","path":"reference.tif"}',
            "--output",
            "raw.zarr",
            "--spec",
            "model_spec.yaml",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "raw.zarr" not in result.stdout

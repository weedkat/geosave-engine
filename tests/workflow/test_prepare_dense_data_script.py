from pathlib import Path

from geosave_engine.templates.scaffolds.scripts import (
    prepare_dense_data as script,
)


def test_script_calls_prepare_dense_data(monkeypatch, capsys):
    arguments = {}

    def run_flow(**kwargs):
        arguments.update(kwargs)
        return "prepared/manifest.parquet"

    monkeypatch.setattr(script, "prepare_dense_data", run_flow)

    script.main(
        labels=Path("labels"),
        output=Path("prepared"),
        spec=Path("model_spec.yaml"),
        pattern="train/*.tif",
        format="zarr",
        write_options='{"compress":"ZSTD"}',
    )

    assert arguments == {
        "labels": "labels",
        "output": "prepared",
        "spec": "model_spec.yaml",
        "pattern": "train/*.tif",
        "format": "zarr",
        "write_options": {"compress": "ZSTD"},
    }
    assert capsys.readouterr().out.strip() == "prepared/manifest.parquet"


def test_script_defaults_to_geotiff_and_the_workspace_spec(monkeypatch, capsys):
    arguments = {}

    def run_flow(**kwargs):
        arguments.update(kwargs)
        return "prepared/manifest.parquet"

    monkeypatch.setattr(script, "prepare_dense_data", run_flow)

    script.main()

    assert arguments["spec"] == "configs/model_spec.yaml"
    assert arguments["format"] == "geotiff"
    assert arguments["write_options"] == {}
    assert "metadata" not in arguments
    assert capsys.readouterr().out.strip() == "prepared/manifest.parquet"

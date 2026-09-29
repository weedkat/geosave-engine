from pathlib import Path

from geosave_engine.templates.boilerplate.scripts import ingest_imagery


def test_ingest_script_calls_prepare_dense_data(monkeypatch, capsys):
    captured = {}

    def run_flow(**arguments):
        captured.update(arguments)
        return "prepared/manifest.parquet"

    monkeypatch.setattr(ingest_imagery, "prepare_dense_data", run_flow)

    ingest_imagery.main(
        labels=Path("labels"),
        output=Path("prepared"),
        spec=Path("model_spec.yaml"),
        pattern="train/*.tif",
        metadata=Path("samples.csv"),
        format="zarr",
        write_options='{"compress":"ZSTD"}',
    )

    assert captured == {
        "labels": "labels",
        "output": "prepared",
        "spec": "model_spec.yaml",
        "pattern": "train/*.tif",
        "metadata": "samples.csv",
        "format": "zarr",
        "write_options": {"compress": "ZSTD"},
    }
    assert capsys.readouterr().out.strip() == "prepared/manifest.parquet"


def test_ingest_script_defaults_to_geotiff(monkeypatch, capsys):
    captured = {}

    def run_flow(**arguments):
        captured.update(arguments)
        return "prepared/manifest.parquet"

    monkeypatch.setattr(ingest_imagery, "prepare_dense_data", run_flow)

    ingest_imagery.main()

    assert captured["format"] == "geotiff"
    assert captured["write_options"] == {}
    assert captured["metadata"] is None
    assert capsys.readouterr().out.strip() == "prepared/manifest.parquet"

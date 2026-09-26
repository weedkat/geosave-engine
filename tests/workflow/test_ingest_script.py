from pathlib import Path

from geosave_engine.templates.boilerplate.scripts import ingest_imagery


def test_ingest_script_passes_training_job_inputs(monkeypatch, capsys):
    captured = {}

    def run_flow(**arguments):
        captured.update(arguments)
        return "prepared/manifest.parquet"

    monkeypatch.setattr(ingest_imagery, "prepare_training", run_flow)

    ingest_imagery.main(
        labels=Path("labels"),
        output=Path("prepared"),
        spec=Path("model_spec.yaml"),
        pattern="train/*.tif",
    )

    assert captured == {
        "labels": "labels",
        "sources": {"sentinel_2_l2a": {"query": {}, "load": {}}},
        "output": "prepared",
        "spec": "model_spec.yaml",
        "pattern": "train/*.tif",
    }
    assert capsys.readouterr().out.strip() == "prepared/manifest.parquet"

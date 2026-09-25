from pathlib import Path
import subprocess
import sys

import dask.array as da
import numpy as np
from dask.callbacks import Callback

from geosave_engine.cli.core.workspace import create_workspace
from geosave_engine.workflow.flows import preprocess
from geosave_engine.workflow.specs import ModelSpec, Ref


def test_shipped_model_spec_round_trips_as_inert_declarations(tmp_path):
    path = (
        Path(__file__).parents[3]
        / "src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml"
    )

    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path))

    assert restored == spec
    assert spec.sources["sentinel_2_l2a"].variables == (
        "B02",
        "B03",
        "B04",
        "B08",
    )
    assert spec.inference["image"].call == Ref("image.gs.to_tensor")
    assert spec.inference["image"].kwargs == {"dtype": "float32"}
    assert spec.postprocessing.model_dump() == {}


def test_value_declarations_round_trip_without_loading_calls(tmp_path):
    path = Path(__file__).with_name("fixtures") / "values.yaml"

    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path))

    assert restored == spec
    assert spec.preprocessing["value"].call == Ref("scale")
    assert spec.preprocessing["record"].kwargs["text"] == "!ref value"


def test_shipped_preprocessing_stays_lazy_and_sample_ready(raw, prefect_server):
    optical = raw["optical"].rename({"red": "B04", "nir": "B08"})
    optical["B02"] = optical.B04.copy()
    optical["B03"] = optical.B04.copy()
    for name in ("B02", "B03", "B04", "B08"):
        optical[name] = (optical[name] * 1000).astype("uint16")
        optical[name].attrs.update(scale_factor=0.0001, add_offset=0.0)
    path = (
        Path(__file__).parents[3]
        / "src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml"
    )
    computed = []

    with Callback(posttask=lambda *args: computed.append(True)):
        prepared = preprocess({"sentinel_2_l2a": optical}, ModelSpec.load(path))

    assert computed == []
    assert set(prepared) == {"valid_pixels", "image"}
    assert list(prepared["image"].data_vars) == ["B02", "B03", "B04", "B08"]
    assert isinstance(prepared["image"].B04.data, da.Array)
    np.testing.assert_allclose(prepared["image"].B08.compute(), 0.6)
    np.testing.assert_allclose(optical.B08.compute(), 6000)


def test_value_declarations_execute_with_assignment_semantics(prefect_server):
    path = Path(__file__).with_name("fixtures") / "values.yaml"
    audit = {}
    supplied = {
        "value": 2.0,
        "scale": lambda value, factor: value * factor,
        "audit": audit,
    }

    prepared = preprocess(supplied, ModelSpec.load(path))

    assert supplied["value"] == 2.0
    assert prepared["value"] == 6.0
    assert prepared["squared"] == 36.0
    assert prepared["ratio"] == (6, 1)
    assert prepared["record"]["nested"] == [6.0, {"square": 36.0}]
    assert prepared["record"]["literal"] == {"ref": "value"}
    assert prepared["record"]["text"] == "!ref value"
    assert prepared["record"]["enabled"] is True
    assert prepared["record"]["missing"] is None
    assert prepared["recorded"] is None
    assert audit["latest"] is prepared["record"]
    assert prepared["array"].tolist() == [[6.0, 36.0]]


def test_walkthrough_runs_without_model_or_external_service(tmp_path):
    create_workspace(tmp_path, "semantic_segmentation", "supervised")

    result = subprocess.run(
        [sys.executable, "scripts/prepare_example.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "B08" in result.stdout
    assert "0.6" in result.stdout

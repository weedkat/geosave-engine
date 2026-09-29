from pathlib import Path
import subprocess
import sys

import dask.array as da
import numpy as np
from dask.callbacks import Callback

from geosave_engine.cli.core.workspace import create_workspace
from geosave_engine.model_spec import ModelSpec, Ref


def test_shipped_model_spec_round_trips_as_inert_declarations(tmp_path):
    path = (
        Path(__file__).parents[2]
        / "src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml"
    )

    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path))

    assert restored == spec
    assert spec.rasters["sentinel_2_l2a"].variables == (
        "B02",
        "B03",
        "B04",
        "B08",
    )
    assert spec.preprocessing["valid_pixels"].call == (
        "geosave_engine.geodata.transform.nodata.to_nan"
    )
    assert spec.preprocessing["valid_pixels"].kwargs == {"data": Ref("sentinel_2_l2a")}
    assert spec.preprocessing["image"].call == (
        "geosave_engine.geodata.transform.packing.unpack"
    )
    assert spec.preprocessing["image"].kwargs == {"data": Ref("valid_pixels")}


def test_shipped_preprocessing_stays_lazy_and_sample_ready(raw):
    optical = raw["optical"].rename({"red": "B04", "nir": "B08"})
    optical["B02"] = optical.B04.copy()
    optical["B03"] = optical.B04.copy()
    for name in ("B02", "B03", "B04", "B08"):
        optical[name] = (optical[name] * 1000).astype("uint16")
        optical[name].attrs.update(scale_factor=0.0001, add_offset=0.0)
    path = (
        Path(__file__).parents[2]
        / "src/geosave_engine/templates/tasks/semantic_segmentation/supervised/configs/model_spec.yaml"
    )
    computed = []

    with Callback(posttask=lambda *args: computed.append(True)):
        prepared = ModelSpec.load(path).preprocess({"sentinel_2_l2a": optical})

    assert computed == []
    assert set(prepared) == {"valid_pixels", "image"}
    assert list(prepared["image"].data_vars) == ["B02", "B03", "B04", "B08"]
    assert isinstance(prepared["image"].B04.data, da.Array)
    np.testing.assert_allclose(prepared["image"].B08.compute(), 0.6)
    np.testing.assert_allclose(optical.B08.compute(), 6000)


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

from pathlib import Path

import dask.array as da
import numpy as np
from dask.callbacks import Callback

from geosave_engine.cli.core.workspace import create_workspace
from geosave_engine.workflow.processing import Processor
from geosave_engine.workflow.spec import ModelSpec


def test_documented_example_round_trips_and_stays_lazy(tmp_path, raw):
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
    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path))
    assert restored == spec
    computed = []
    with Callback(posttask=lambda *args: computed.append(True)):
        prepared = Processor(stage="preprocessing", spec=restored)(
            {"sentinel_2_l2a": optical}
        )
    assert computed == []
    assert list(prepared["sentinel_2_l2a"].data_vars) == ["B02", "B03", "B04", "B08"]
    assert prepared["sentinel_2_l2a"].B04.data is optical.B04.data
    assert prepared["sentinel_2_l2a"].B08.data is optical.B08.data
    assert "unused" in optical
    assert list(prepared["image"].data_vars) == ["B02", "B03", "B04", "B08"]
    assert isinstance(prepared["image"].B04.data, da.Array)
    assert prepared["image"].gs.geobox == optical.gs.geobox
    np.testing.assert_allclose(prepared["image"].B08.compute(), 0.6)
    np.testing.assert_allclose(prepared["image"].B04.compute(), 0.2)
    np.testing.assert_allclose(optical.B08.compute(), 6000)


def test_walkthrough_runs_without_model_or_external_service(tmp_path):
    import subprocess
    import sys

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

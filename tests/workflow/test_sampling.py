from dask.callbacks import Callback
import pytest
from geosave_engine.geodata.core.stack import stack
from geosave_engine.workflow.spec import InferenceSpec, TensorInputSpec, TilingSpec


def test_sampling_validates_grid_without_computing_and_ignores_unused_groups(raw, spec):
    from geosave_engine.workflow.sampling import sample_windows

    raw["unused-grid"] = raw["optical"].assign_coords(x=raw["optical"].x + 5)
    reads = []
    with Callback(pretask=lambda *args: reads.append(args)):
        windows = sample_windows(stack(raw), settings=spec.inference)
    assert len(windows) == 1
    assert windows[0].gs.groups == ("optical",)
    assert reads == []


def test_bound_inputs_must_share_reference_grid(raw):
    from geosave_engine.workflow.sampling import sample_windows

    raw["shifted"] = raw["optical"].assign_coords(x=raw["optical"].x + 5)
    settings = InferenceSpec(
        inputs={"image": TensorInputSpec(raster="shifted")},
        tiling=TilingSpec(raster="optical", tile_shape=(2, 2)),
    )
    with pytest.raises(ValueError, match="grid"):
        sample_windows(stack(raw), settings=settings)


def test_temporal_tensor_layout_requires_a_time_dimension(raw):
    from geosave_engine.workflow.sampling import sample_windows

    settings = InferenceSpec(
        inputs={"image": TensorInputSpec(raster="optical", layout="TCHW")},
        tiling=TilingSpec(raster="optical", tile_shape=(2, 2)),
    )
    with pytest.raises(ValueError, match="time|TCHW"):
        sample_windows(stack(raw), settings=settings)

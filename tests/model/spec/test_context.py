import pandas as pd
import pytest

from geosave_engine.model.spec import ModelSpec, Ref


def _spec():
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"image": {"channels": 1}},
            "inputs": {"image": Ref("image")},
            "context": {"call": "builtins.dict", "kwargs": {"time": Ref("row.times")}},
        }
    )


def test_row_context_recipe_round_trips_and_runs_in_inference(tmp_path):
    spec = _spec()
    restored = ModelSpec.load(spec.save(tmp_path))
    row = pd.Series({"times": [2024, 2025]})
    assert restored.model_inputs({"image": "pixels"}, row) == {
        "image": "pixels",
        "time": [2024, 2025],
    }


def test_explicit_cached_context_skips_computation_and_matches_live():
    spec = _spec()
    row = pd.Series({"times": [2024, 2025]})
    cached = spec.model_context(row)
    assert spec.model_inputs(
        {"image": "pixels"}, pd.Series(dtype=object), context=cached
    ) == spec.model_inputs({"image": "pixels"}, row)


def test_context_cannot_replace_pixel_inputs():
    with pytest.raises(ValueError, match="overwrite"):
        _spec().model_inputs(
            {"image": "pixels"}, pd.Series(dtype=object), context={"image": "bad"}
        )


def test_declared_context_needs_a_row():
    spec = _spec()
    with pytest.raises(ValueError, match="row"):
        spec.model_inputs({"image": "pixels"})


def test_context_cannot_read_rasters_as_metadata():
    with pytest.raises(ValueError, match="sample row"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {"image": {"channels": 1}},
                "context": {"call": "builtins.dict", "kwargs": {"image": Ref("image")}},
            }
        )

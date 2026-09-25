from pathlib import Path

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

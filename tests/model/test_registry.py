from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest
import torch
from torch import nn

from geosave_engine.model.chain import Published, chain_step
from geosave_engine.model.factory import BuildSpec
from geosave_engine.model.registry import (
    MODEL_REGISTRY,
    build_model,
    list_models,
    register_model,
)


class Encoder(nn.Module):
    feature_channels: Published[int]

    def __init__(self, feature_channels: int = 2) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.factor = nn.Parameter(torch.tensor(2.0))

    @chain_step()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = image * self.factor
        return features


class Head(nn.Module):
    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.feature_channels


def make_head(feature_channels: int) -> Head:
    return Head(feature_channels)


@pytest.fixture
def model_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    list_models()
    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"TEST": Encoder})
    monkeypatch.setitem(MODEL_REGISTRY, "head", {"TEST": make_head})


@pytest.mark.parametrize("use_path", [False, True])
def test_build_model_builds_serializable_stages_and_wires_published_values(
    use_path: bool, model_factories: None
) -> None:
    stages: dict[str, dict[str, str]] = {
        "encoder": {"name": "test"},
        "head": {"name": "test"},
    }
    if use_path:
        stages = {
            "encoder": {"class_path": f"{__name__}.Encoder"},
            "head": {"class_path": f"{__name__}.Head"},
        }
    original = deepcopy(stages)
    model = build_model(json.loads(json.dumps(stages)))

    result = model(torch.tensor(3.0))
    assert isinstance(result, torch.Tensor)
    assert result.item() == 8.0
    result.backward()
    encoder = model.get_submodule("encoder")
    assert isinstance(encoder, Encoder)
    gradient = encoder.factor.grad
    assert gradient is not None
    assert gradient.item() == 3.0
    assert stages == original


def test_explicit_arguments_override_published_values(model_factories: None) -> None:
    model = build_model(
        {
            "encoder": {"name": "test"},
            "head": {"name": "test", "init_args": {"feature_channels": 7}},
        }
    )
    head = model.get_submodule("head")
    assert isinstance(head, Head)
    assert head.feature_channels == 7


def test_unknown_constructor_arguments_are_not_dropped(model_factories: None) -> None:
    with pytest.raises(TypeError, match="featre_channels") as caught:
        build_model(
            {"encoder": {"name": "test", "init_args": {"featre_channels": 7}}}
        )
    assert caught.value.__notes__ == ["While building stage 'encoder'"]


@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError])
def test_build_model_preserves_constructor_errors(monkeypatch, error_type) -> None:
    failure = error_type("invalid layout")

    def failing_encoder() -> nn.Module:
        raise failure

    list_models()
    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"FAIL": failing_encoder})

    with pytest.raises(error_type) as caught:
        build_model({"encoder": {"name": "fail"}})

    assert caught.value is failure
    assert failure.__notes__ == ["While building stage 'encoder'"]


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_model({})


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"name": "test", "class_path": f"{__name__}.Head"},
        {"name": "missing"},
        {"class_path": "Head"},
        {"name": "test", "unknown": True},
        {"name": "test", "init_args": []},
    ],
)
def test_model_chain_rejects_malformed_stage_selectors(
    spec: dict[str, object], model_factories: None
) -> None:
    with pytest.raises((TypeError, ValueError, KeyError)):
        build_model({"encoder": spec})


def test_build_model_records_an_independent_resolved_recipe(
    model_factories: None,
) -> None:
    specs: dict[str, dict[str, str]] = {
        "encoder": {"name": "test"},
        "head": {"name": "test"},
    }

    model = build_model(specs)
    specs.clear()
    assert list(model.stage_specs) == ["encoder", "head"]


def test_build_model_accepts_validated_build_specs(model_factories: None) -> None:
    model = build_model(
        {
            "encoder": BuildSpec(name="test"),
            "head": BuildSpec(name="test"),
        }
    )

    torch.testing.assert_close(model(torch.tensor(3.0)), torch.tensor(8.0))
    copied = model.stage_specs
    copied.clear()
    assert list(model.stage_specs) == ["encoder", "head"]


def test_register_model_accepts_factory_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "test", {})
    decorated = register_model("test", "factory")(make_head)
    assert decorated is make_head
    assert MODEL_REGISTRY["test"]["FACTORY"] is make_head


def test_build_model_captures_defaults_and_preserves_factory_selector(
    model_factories: None,
) -> None:
    model = build_model({"encoder": {"name": "test"}, "head": {"name": "test"}})

    assert model.stage_specs == {
        "encoder": {"name": "test", "init_args": {"feature_channels": 2}},
        "head": {"name": "test", "init_args": {"feature_channels": 2}},
    }
    assert isinstance(model.get_submodule("encoder"), Encoder)
    assert isinstance(model.get_submodule("head"), Head)


def test_build_model_saves_initial_arguments_before_factory_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(channels: list[int], pretrained: bool = True) -> Encoder:
        channels.append(8)
        encoder = Encoder(feature_channels=sum(channels))
        if pretrained:
            nn.init.constant_(encoder.factor, 3.0)
        return encoder

    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"TEST": factory})
    model = build_model(
        {"encoder": {"name": "test", "init_args": {"channels": [2, 4]}}}
    )

    encoder = model.get_submodule("encoder")
    assert isinstance(encoder, Encoder)
    assert encoder.feature_channels == 14
    assert encoder.factor.item() == 3.0
    assert model.stage_specs == {
        "encoder": {
            "name": "test",
            "init_args": {"channels": [2, 4], "pretrained": False},
        }
    }


@pytest.mark.parametrize(
    ("encoder", "inputs"),
    [
        ({"name": "dinov3", "init_args": {"pretrained": False}}, ["image"]),
        (
            {"name": "prithvi_tl", "init_args": {"model_name": "prithvi_eo_v2_tiny_tl"}},
            ["image", "temporal_coords", "location_coords"],
        ),
        (
            {
                "name": "clay",
                "init_args": {
                    "model_name": "clay_v15_tiny",
                    "waves": [0.49, 0.56, 0.665, 0.842],
                    "gsd": 10.0,
                },
            },
            ["image"],
        ),
    ],
)
def test_registered_encoders_chain_into_a_decoder_and_head(
    encoder: dict[str, Any], inputs: list[str]
) -> None:
    encoder = deepcopy(encoder)
    encoder["init_args"]["in_channels"] = 4
    model = build_model(
        {
            "encoder": encoder,
            "decoder": {"name": "dpt"},
            "head": {
                "name": "segmentation",
                "init_args": {"classes": ["background", "target"]},
            },
        }
    )
    assert list(model.inputs) == inputs

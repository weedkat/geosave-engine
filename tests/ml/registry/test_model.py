from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from geosave_engine.ml.models.contract import ModelChain, Published, chain_step
from geosave_engine.ml.registry import StageSpec, register_model
from geosave_engine.ml.registry.model import MODEL_REGISTRY, build_stages


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
    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"TEST": Encoder})
    monkeypatch.setitem(MODEL_REGISTRY, "head", {"TEST": make_head})


@pytest.mark.parametrize("use_path", [False, True])
def test_model_chain_builds_serializable_stages_and_wires_published_values(
    use_path: bool, model_factories: None
) -> None:
    stages: dict[str, StageSpec] = {
        "encoder": {"name": "test"},
        "head": {"name": "test"},
    }
    if use_path:
        stages = {
            "encoder": {"class_path": f"{__name__}.Encoder"},
            "head": {"class_path": f"{__name__}.Head"},
        }
    original = deepcopy(stages)
    model = ModelChain(stages=json.loads(json.dumps(stages)))

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
    model = ModelChain(
        stages={
            "encoder": {"name": "test"},
            "head": {"name": "test", "init_args": {"feature_channels": 7}},
        }
    )
    head = model.get_submodule("head")
    assert isinstance(head, Head)
    assert head.feature_channels == 7


def test_unknown_constructor_arguments_are_not_dropped(model_factories: None) -> None:
    with pytest.raises(TypeError, match="encoder.*featre_channels"):
        ModelChain(stages={"encoder": {"name": "test", "init_args": {"featre_channels": 7}}})


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ModelChain(stages={})


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
    spec: StageSpec, model_factories: None
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelChain(stages={"encoder": spec})


def test_build_model_is_not_exported() -> None:
    with pytest.raises(ImportError):
        exec("from geosave_engine.ml.registry import build_model")


def test_register_model_accepts_factory_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "test", {})
    decorated = register_model("test", "factory")(make_head)
    assert decorated is make_head
    assert MODEL_REGISTRY["test"]["FACTORY"] is make_head


def test_built_stages_capture_defaults_and_preserve_factory_selector(
    model_factories: None,
) -> None:
    built = build_stages({"encoder": {"name": "test"}, "head": {"name": "test"}})

    assert built.config == {
        "encoder": {"name": "test", "init_args": {"feature_channels": 2}},
        "head": {"name": "test", "init_args": {"feature_channels": 2}},
    }
    assert isinstance(built.modules["encoder"], Encoder)
    assert isinstance(built.modules["head"], Head)


def test_built_stages_save_initial_arguments_before_factory_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(channels: list[int], pretrained: bool = True) -> Encoder:
        channels.append(8)
        encoder = Encoder(feature_channels=sum(channels))
        if pretrained:
            nn.init.constant_(encoder.factor, 3.0)
        return encoder

    monkeypatch.setitem(MODEL_REGISTRY, "encoder", {"TEST": factory})
    built = build_stages(
        {"encoder": {"name": "test", "init_args": {"channels": [2, 4]}}}
    )

    encoder = built.modules["encoder"]
    assert isinstance(encoder, Encoder)
    assert encoder.feature_channels == 14
    assert encoder.factor.item() == 3.0
    assert built.config == {
        "encoder": {
            "name": "test",
            "init_args": {"channels": [2, 4], "pretrained": False},
        }
    }

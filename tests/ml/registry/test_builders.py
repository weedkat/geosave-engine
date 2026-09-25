from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from geosave_engine.ml.models.contract import Published, chain_step
from geosave_engine.ml.registry import (
    BuildSpec,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    register_model,
)
from geosave_engine.ml.registry.model import MODEL_REGISTRY


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
def test_model_builds_from_serializable_spec_and_wires_published_values(
    use_path: bool, model_factories: None
) -> None:
    stages: dict[str, BuildSpec] = {
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
    with pytest.raises(TypeError, match="encoder.*featre_channels"):
        build_model({"encoder": {"name": "test", "init_args": {"featre_channels": 7}}})


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_model({})


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"name": "CELoss", "class_path": "torch.nn.CrossEntropyLoss"},
        {"name": "missing"},
        {"class_path": "CrossEntropyLoss"},
    ],
)
def test_invalid_selectors_are_rejected(spec: BuildSpec) -> None:
    with pytest.raises(ValueError):
        build_loss(spec)


@pytest.mark.parametrize(
    "path", ["torch.optim.AdamW", "torch.nn.functional.cross_entropy"]
)
def test_class_path_requires_a_loss_module_class(path: str) -> None:
    with pytest.raises(TypeError, match="subclass"):
        build_loss({"class_path": path})


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "CELoss", "init_args": {"reduction": "sum", "ignore_index": 255}},
        {
            "class_path": "torch.nn.CrossEntropyLoss",
            "init_args": {"reduction": "sum", "ignore_index": 255},
        },
    ],
)
def test_loss_routes_receive_init_args(spec: BuildSpec) -> None:
    loss = build_loss(spec)
    logits = torch.zeros(2, 3)
    target = torch.tensor([1, 255])
    assert loss(logits, target).item() == pytest.approx(
        torch.log(torch.tensor(3.0)).item()
    )


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "AdamW", "init_args": {"lr": 0.1, "weight_decay": 0.0}},
        {
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 0.1, "weight_decay": 0.0},
        },
    ],
)
def test_optimizer_routes_update_the_supplied_model(spec: BuildSpec) -> None:
    model = nn.Linear(1, 1, bias=False)
    nn.init.ones_(model.weight)
    optimizer = build_optimizer(spec, model)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    assert model.weight.item() == pytest.approx(0.9)


def test_named_optimizer_factory_preserves_parameter_groups() -> None:
    model = nn.ModuleDict({"encoder": nn.Linear(2, 2), "head": nn.Linear(2, 1)})
    optimizer = build_optimizer(
        {
            "name": "AdamW.split",
            "init_args": {"encoder_lr": 0.01, "decoder_lr": 0.1},
        },
        model,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [0.01, 0.1]
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {
        id(p) for p in model["encoder"].parameters()
    }
    assert {id(p) for p in optimizer.param_groups[1]["params"]} == {
        id(p) for p in model["head"].parameters()
    }


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "CosineAnnealingLR", "init_args": {"T_max": 2}},
        {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": 2},
        },
    ],
)
def test_scheduler_routes_use_the_supplied_optimizer(spec: BuildSpec) -> None:
    optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)
    scheduler = build_scheduler(spec, optimizer)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


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
    from geosave_engine.ml.registry.model import build_stages

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
    from geosave_engine.ml.registry.model import build_stages

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

from __future__ import annotations

import pytest
import torch
from torch import nn

from geosave_engine.ml.registry import (
    BuildSpec,
    build_loss,
    build_optimizer,
    build_scheduler,
)


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

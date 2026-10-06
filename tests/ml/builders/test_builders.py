from __future__ import annotations

from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from geosave_engine.ml.builders import (
    build_criterion,
    build_optimizer,
    build_scheduler,
)
from geosave_engine.model.factory import BuildSpec
from geosave_engine.ml.criterion import ProbOhemCrossEntropy2d


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"name": "cross_entropy", "class_path": "torch.nn.CrossEntropyLoss"},
        {"class_path": "CrossEntropyLoss"},
        {"class_path": "torch.nn.CrossEntropyLoss", "unknown": True},
        {"class_path": "torch.nn.CrossEntropyLoss", "init_args": []},
    ],
)
def test_resolve_rejects_invalid_specs(spec: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BuildSpec.model_validate(spec)


def test_build_spec_resolves_registered_names() -> None:
    spec = BuildSpec(name="CrOsS_EnTrOpY")

    assert (
        spec.resolve({"CROSS_ENTROPY": nn.CrossEntropyLoss}, nn.Module)
        is nn.CrossEntropyLoss
    )


def test_build_spec_rejects_unknown_registered_names() -> None:
    spec = BuildSpec(name="missing")

    with pytest.raises(KeyError, match="Unknown name"):
        spec.resolve({"CROSS_ENTROPY": nn.CrossEntropyLoss}, nn.Module)


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "CrOsS_EnTrOpY", "init_args": {"ignore_index": 255}},
        {
            "class_path": "torch.nn.CrossEntropyLoss",
            "init_args": {"ignore_index": 255},
        },
    ],
)
def test_build_criterion_supports_registered_names_and_class_paths(
    spec: dict[str, Any],
) -> None:
    criterion = build_criterion(spec)
    logits = torch.zeros(2, 3)
    target = torch.tensor([1, 255])

    assert criterion(logits, target).item() == pytest.approx(
        torch.log(torch.tensor(3.0)).item()
    )


def test_build_criterion_exposes_ohem() -> None:
    criterion = build_criterion(
        {"name": "ohem", "init_args": {"ignore_index": 255}}
    )

    assert isinstance(criterion, ProbOhemCrossEntropy2d)


def test_build_criterion_rejects_optimizer_classes() -> None:
    with pytest.raises(TypeError, match="Module subclass"):
        build_criterion({"class_path": "torch.optim.AdamW"})


def test_optimizer_groups_exact_children_and_remaining_parameters() -> None:
    model = nn.ModuleDict(
        {"encoder": nn.Linear(2, 2), "head": nn.Linear(2, 1)}
    )
    model["encoder"].bias.requires_grad_(False)

    optimizer = build_optimizer(
        {
            "name": "adamw",
            "init_args": {"lr": 1e-3, "weight_decay": 1e-2},
            "groups": {"head": {"lr": 1e-4, "weight_decay": 0.0}},
        },
        model,
    )

    head, remaining = optimizer.param_groups
    assert head["params"] == [model["head"].weight, model["head"].bias]
    assert remaining["params"] == [model["encoder"].weight]
    assert (head["lr"], head["weight_decay"]) == (1e-4, 0.0)


def test_optimizer_rejects_unknown_and_overlapping_groups() -> None:
    first = nn.Linear(1, 1)
    second = nn.Linear(1, 1)
    second.weight = first.weight
    model = nn.ModuleDict({"first": first, "second": second})

    with pytest.raises(ValueError, match="share parameters"):
        build_optimizer(
            {"name": "adamw", "groups": {"first": {}, "second": {}}},
            model,
        )
    with pytest.raises(KeyError, match="Unknown model groups"):
        build_optimizer({"name": "adamw", "groups": {"missing": {}}}, model)


def test_optimizer_rejects_a_model_without_trainable_parameters() -> None:
    model = nn.Linear(1, 1).requires_grad_(False)

    with pytest.raises(ValueError, match="trainable parameters"):
        build_optimizer({"name": "adamw"}, model)


def test_scheduler_returns_lightning_metadata() -> None:
    optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1)

    configured = build_scheduler(
        {
            "name": "reduce_on_plateau",
            "init_args": {"patience": 2},
            "monitor": "val_loss",
            "interval": "epoch",
            "strict": False,
            "scheduler_name": "validation",
        },
        optimizer,
    )

    assert isinstance(
        configured["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    assert configured["scheduler"].optimizer is optimizer
    assert {
        key: configured[key] for key in ("monitor", "interval", "strict", "name")
    } == {
        "monitor": "val_loss",
        "interval": "epoch",
        "strict": False,
        "name": "validation",
    }

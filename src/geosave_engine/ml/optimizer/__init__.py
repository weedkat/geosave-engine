"""Ways of grouping a model's parameters before an optimizer sees them.

A strategy decides which parameters share which hyperparameters, never the
update rule, so it composes with any optimizer class. Each takes that class
and returns the built optimizer, passing on what it does not consume.

Examples:
    >>> split(torch.optim.AdamW, model, encoder_lr=1e-5, decoder_lr=1e-3)
    AdamW (...)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torch.optim import Optimizer

from geosave_engine.ml.utils.torch_params import (
    freeze_backbone,
    layerwise_param_groups,
    split_encoder_decoder,
    split_no_wd,
)

# Builds one optimizer of the given class over one model's parameters.
type Strategy = Callable[..., Optimizer]


def single(cls: type[Optimizer], model: nn.Module, **kwargs: Any) -> Optimizer:
    """Train every parameter under one learning rate.

    Args:
        cls: Optimizer class to build.
        model: Module whose parameters are optimized.
        **kwargs: Passed to `cls`.

    Returns:
        Optimizer over every parameter as one group.
    """
    return cls(model.parameters(), **kwargs)


def split(
    cls: type[Optimizer],
    model: nn.Module,
    *,
    encoder_lr: float,
    decoder_lr: float,
    **kwargs: Any,
) -> Optimizer:
    """Give the encoder and the rest of the model their own learning rates.

    Args:
        cls: Optimizer class to build.
        model: Module whose parameters are optimized.
        encoder_lr: Learning rate for parameters named encoder or backbone.
        decoder_lr: Learning rate for every other parameter.
        **kwargs: Passed to `cls`.

    Returns:
        Optimizer over two parameter groups.
    """
    encoder, decoder = split_encoder_decoder(model)
    return cls(
        [{"params": encoder, "lr": encoder_lr}, {"params": decoder, "lr": decoder_lr}],
        **kwargs,
    )


def no_wd(
    cls: type[Optimizer],
    model: nn.Module,
    *,
    weight_decay: float = 1e-2,
    **kwargs: Any,
) -> Optimizer:
    """Exempt bias and norm parameters from weight decay.

    Args:
        cls: Optimizer class to build.
        model: Module whose parameters are optimized.
        weight_decay: Rate for the decayed group; the other group gets zero.
        **kwargs: Passed to `cls`.

    Returns:
        Optimizer over a decayed and an undecayed parameter group.
    """
    decayed, undecayed = split_no_wd(model)
    return cls(
        [
            {"params": decayed, "weight_decay": weight_decay},
            {"params": undecayed, "weight_decay": 0.0},
        ],
        **kwargs,
    )


def freeze_encoder(cls: type[Optimizer], model: nn.Module, **kwargs: Any) -> Optimizer:
    """Freeze the encoder, so only the rest of the model trains.

    Args:
        cls: Optimizer class to build.
        model: Module whose encoder or backbone parameters stop requiring grad.
        **kwargs: Passed to `cls`.

    Returns:
        Optimizer over the parameters still requiring grad.
    """
    return cls(freeze_backbone(model), **kwargs)


def layerwise(
    cls: type[Optimizer],
    model: nn.Module,
    *,
    lr: float = 1e-4,
    decoder_lr: float = 1e-3,
    decay_rate: float = 0.75,
    **kwargs: Any,
) -> Optimizer:
    """Decay the learning rate down a ViT-style backbone's blocks.

    Args:
        cls: Optimizer class to build.
        model: Module whose backbone is named `blocks.<i>` or `layers.<i>`.
        lr: Learning rate at the backbone's last block.
        decoder_lr: Learning rate for parameters outside the backbone.
        decay_rate: Factor applied once per block, deepest block first.
        **kwargs: Passed to `cls`.

    Returns:
        Optimizer over one parameter group per block, plus one for the rest.
    """
    return cls(layerwise_param_groups(model, lr, decoder_lr, decay_rate), **kwargs)


STRATEGIES: dict[str, Strategy] = {
    "": single,
    "split": split,
    "no_wd": no_wd,
    "freeze_encoder": freeze_encoder,
    "layerwise": layerwise,
}

__all__ = [
    "STRATEGIES",
    "Strategy",
    "freeze_encoder",
    "layerwise",
    "no_wd",
    "single",
    "split",
]

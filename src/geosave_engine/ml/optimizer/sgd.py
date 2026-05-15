from __future__ import annotations

import torch
import torch.nn as nn

from geosave_engine.utils.torch_params import freeze_backbone, split_encoder_decoder, split_no_wd


def default(
    model: nn.Module,
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.SGD:
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


def split(
    model: nn.Module,
    encoder_lr: float,
    decoder_lr: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.SGD:
    encoder, decoder = split_encoder_decoder(model)
    return torch.optim.SGD(
        [{"params": encoder, "lr": encoder_lr}, {"params": decoder, "lr": decoder_lr}],
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )


def no_wd(
    model: nn.Module,
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.SGD:
    wd, no_wd_params = split_no_wd(model)
    return torch.optim.SGD(
        [{"params": wd, "weight_decay": weight_decay}, {"params": no_wd_params, "weight_decay": 0.0}],
        lr=lr,
        momentum=momentum,
        **kwargs,
    )


def freeze_encoder(
    model: nn.Module,
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.SGD:
    trainable = freeze_backbone(model)
    return torch.optim.SGD(trainable, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

from __future__ import annotations

import torch
import torch.nn as nn

from geosave_engine.utils.torch_params import freeze_backbone, split_encoder_decoder


def default(
    model: nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.RMSprop:
    return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


def split(
    model: nn.Module,
    encoder_lr: float,
    decoder_lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.RMSprop:
    encoder, decoder = split_encoder_decoder(model)
    return torch.optim.RMSprop(
        [{"params": encoder, "lr": encoder_lr}, {"params": decoder, "lr": decoder_lr}],
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )


def freeze_encoder(
    model: nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.RMSprop:
    trainable = freeze_backbone(model)
    return torch.optim.RMSprop(trainable, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

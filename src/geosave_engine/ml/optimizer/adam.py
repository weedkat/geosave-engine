from __future__ import annotations

import torch
import torch.nn as nn

from geosave_engine.utils.torch_params import freeze_backbone, layerwise_param_groups, split_encoder_decoder, split_no_wd


def default(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def split(
    model: nn.Module,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Adam:
    encoder, decoder = split_encoder_decoder(model)
    return torch.optim.Adam(
        [{"params": encoder, "lr": encoder_lr}, {"params": decoder, "lr": decoder_lr}],
        weight_decay=weight_decay,
        **kwargs,
    )


def no_wd(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4, **kwargs) -> torch.optim.Adam:
    wd, no_wd_params = split_no_wd(model)
    return torch.optim.Adam(
        [{"params": wd, "weight_decay": weight_decay}, {"params": no_wd_params, "weight_decay": 0.0}],
        lr=lr,
        **kwargs,
    )


def freeze_encoder(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs) -> torch.optim.Adam:
    trainable = freeze_backbone(model)
    return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay, **kwargs)


def layerwise(
    model: nn.Module,
    lr: float = 1e-4,
    decoder_lr: float = 1e-3,
    weight_decay: float = 0.0,
    decay_rate: float = 0.75,
    **kwargs,
) -> torch.optim.Adam:
    groups = layerwise_param_groups(model, lr, decoder_lr, decay_rate)
    return torch.optim.Adam(groups, weight_decay=weight_decay, **kwargs)

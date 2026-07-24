from __future__ import annotations

import torch
import torch.nn as nn

from geosave_engine.ml.utils.torch_params import freeze_backbone


def default(
    model: nn.Module,
    lr: float = 1e-2,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Adagrad:
    return torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, **kwargs)


def freeze_encoder(
    model: nn.Module,
    lr: float = 1e-2,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Adagrad:
    trainable = freeze_backbone(model)
    return torch.optim.Adagrad(trainable, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, **kwargs)

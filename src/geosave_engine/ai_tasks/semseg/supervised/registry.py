from geosave_engine._core.registry import Registry
import torch
import torch.nn as nn
import torchmetrics
from ..model.dpt.build import registry as dpt_registry
from ..model.smp.build import registry as smp_registry

OPTIM_REGISTRY = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'AdamW': torch.optim.AdamW,
}

LOSS_REGISTRY = {
    'CELoss': nn.CrossEntropyLoss
}

optim_registry = Registry(OPTIM_REGISTRY)
loss_registry = Registry(LOSS_REGISTRY)
model_registry = dpt_registry + smp_registry

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from geosave_engine.ml.core.factory import builder, method_builder

import geosave_engine.ml.optimizer.adamw as adamw
import geosave_engine.ml.optimizer.adam as adam
import geosave_engine.ml.optimizer.sgd as sgd
import geosave_engine.ml.optimizer.rmsprop as rmsprop
import geosave_engine.ml.optimizer.adagrad as adagrad

from geosave_engine.ml.models import (
    dpt_dinov2_small, dpt_dinov2_base,
    dpt_dinov2_large, dpt_dinov2_giant,
    deeplabv3, deeplabv3plus, fpn,
    linknet, manet, pan, pspnet,
    segformer, unet, unetplusplus,
    upernet,
)

from geosave_engine.ml.loss import ProbOhemCrossEntropy2d

# Global registries for dynamic instantiation from config. Alter this if you want to add new models, losses, optimizers, or schedulers without changing factory code.
MODELS = {
    'dpt_dinov2_small': dpt_dinov2_small,
    'dpt_dinov2_base': dpt_dinov2_base,
    'dpt_dinov2_large': dpt_dinov2_large,
    'dpt_dinov2_giant': dpt_dinov2_giant,
    'deeplabv3': deeplabv3,
    'deeplabv3plus': deeplabv3plus,
    'fpn': fpn,
    'linknet': linknet,
    'manet': manet,
    'pan': pan,
    'pspnet': pspnet,
    'segformer': segformer,
    'unet': unet,
    'unetplusplus': unetplusplus,
    'upernet': upernet,
}
LOSSES = {
    'CELoss': CrossEntropyLoss,
    'OHEMLoss': ProbOhemCrossEntropy2d,
}
OPTIMIZERS = {
    'AdamW': adamw,
    'Adam': adam,
    'SGD': sgd,
    'RMSprop': rmsprop,
    'Adagrad': adagrad,
}
SCHEDULERS = {
    "LRScheduler": LRScheduler,
    "CosineAnnealingLR": CosineAnnealingLR,
}

def build_model(config: dict) -> nn.Module:
    return builder(config["name"], config.get("init_args", {}), MODELS)

def build_loss(config: dict) -> nn.Module:
    return builder(config["name"], config.get("init_args", {}), LOSSES)

def build_optimizer(config: dict, model: nn.Module) -> Optimizer:
    return method_builder(config["name"], {**config.get("init_args", {}), "model": model}, OPTIMIZERS)

def build_scheduler(config: dict | None, optimizer: Optimizer) -> LRScheduler | None:
    if config is None:
        return None
    return builder(config["name"], {**config.get("init_args", {}), "optimizer": optimizer}, SCHEDULERS)

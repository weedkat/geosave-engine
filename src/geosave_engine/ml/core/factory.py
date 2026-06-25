import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
)
from torch.nn import CrossEntropyLoss

from geosave_engine.ml.loss import ProbOhemCrossEntropy2d

import geosave_engine.ml.optimizer.adamw as adamw
import geosave_engine.ml.optimizer.adam as adam
import geosave_engine.ml.optimizer.sgd as sgd
import geosave_engine.ml.optimizer.rmsprop as rmsprop
import geosave_engine.ml.optimizer.adagrad as adagrad


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

def uppercase_keys(d: dict) -> dict:
    """Return a copy of d with all keys uppercased."""
    return {k.upper(): v for k, v in d.items()}


def builder(name: str, config: dict, registry: dict):
    """Direct callable lookup. Use for models and losses."""
    reg = uppercase_keys(registry)
    key = name.upper()
    if key not in reg:
        raise ValueError(f"Unknown '{name}'. Available: {list(registry.keys())}")

    return reg[key](**config)


def method_builder(name: str, config: dict, registry: dict):
    """Dot-notation method dispatch ('key.method'). Use for optimizers and schedulers."""
    if "." not in name:
        name = name + ".default" # Allow default method if not specified

    reg = uppercase_keys(registry)
    raw_key, method = name.split(".", 1)
    key = raw_key.upper()
    if key not in reg:
        raise ValueError(f"Unknown key '{raw_key}'. Available: {list(registry.keys())}")

    entry = reg[key]
    if not hasattr(entry, method):
        available = [m for m in dir(entry) if not m.startswith("_")]
        raise ValueError(f"Unknown method '{method}' on '{raw_key}'. Available: {available}")

    return getattr(entry, method)(**config)

def build_model(config: dict, registry: dict) -> nn.Module:
    if "name" not in config:
        raise ValueError("Model config must have a 'name' field.")
    return builder(config["name"], config.get("init_args", {}), registry)

def build_loss(config: dict, registry: dict = LOSSES) -> nn.Module:
    if "name" not in config:
        raise ValueError("Loss config must have a 'name' field.")
    return builder(config["name"], config.get("init_args", {}), registry)

def build_optimizer(config: dict, model: nn.Module, registry: dict = OPTIMIZERS) -> Optimizer:
    if "name" not in config:
        raise ValueError("Optimizer config must have a 'name' field.")
    return method_builder(config["name"], {**config.get("init_args", {}), "model": model}, registry)

def build_scheduler(config: dict | None, optimizer: Optimizer, registry: dict = SCHEDULERS) -> LRScheduler | None:
    if config is None:
        return None
    if "name" not in config:
        raise ValueError("Scheduler config must have a 'name' field.")
    return builder(config["name"], {**config.get("init_args", {}), "optimizer": optimizer}, registry)
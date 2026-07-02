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
    """Return copy of ``d`` with all keys uppercased."""
    return {k.upper(): v for k, v in d.items()}


def builder(name: str, config: dict, registry: dict):
    """Instantiate entry from registry by name.

    Case-insensitive. Use for losses and models.

    Args:
        name: Registry key (e.g. ``"CELoss"``).
        config: Keyword args passed to the constructor.
        registry: Mapping of name → callable.

    Returns:
        Instantiated object from registry.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    reg = uppercase_keys(registry)
    key = name.upper()
    if key not in reg:
        raise ValueError(f"Unknown '{name}'. Available: {list(registry.keys())}")
    return reg[key](**config)


def method_builder(name: str, config: dict, registry: dict):
    """Instantiate via dot-notation method dispatch (``"key.method"``).

    Use for optimizers where variant is a module-level function.
    If no method given, falls back to ``default``.

    Args:
        name: ``"key"`` or ``"key.method"`` (e.g. ``"AdamW.split"``).
        config: Keyword args passed to the method.
        registry: Mapping of name → module with callable methods.

    Returns:
        Result of calling ``registry[key].method(**config)``.

    Raises:
        ValueError: If key or method not found in registry.
    """
    if "." not in name:
        name = name + ".default"

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


def build_model(name: str, config: dict, registry: dict) -> nn.Module:
    """Build model by name from registry.

    Args:
        name: Registry key.
        config: Keyword args passed to the constructor.
        registry: Mapping of name → model class.

    Returns:
        Instantiated ``nn.Module``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return builder(name, config, registry)


def build_loss(name: str, config: dict, registry: dict = LOSSES) -> nn.Module:
    """Build loss by name from registry.

    Args:
        name: Registry key (e.g. ``"CELoss"``).
        config: Keyword args passed to the constructor.
        registry: Mapping of name → loss class. Defaults to ``LOSSES``.

    Returns:
        Instantiated loss ``nn.Module``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return builder(name, config, registry)


def build_optimizer(name: str, model: nn.Module, config: dict, registry: dict = OPTIMIZERS) -> Optimizer:
    """Build optimizer by name from registry.

    Args:
        name: Registry key; supports dot notation (e.g. ``"AdamW.split"``).
        config: Keyword args passed to the optimizer constructor.
        model: Model whose parameters are passed to the optimizer.
        registry: Mapping of name → optimizer module. Defaults to ``OPTIMIZERS``.

    Returns:
        Instantiated ``Optimizer``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return method_builder(name, {**config, "model": model}, registry)


def build_scheduler(name: str, optimizer: Optimizer, config: dict, registry: dict = SCHEDULERS) -> LRScheduler | None:
    """Build LR scheduler by name from registry.

    Args:
        name: Registry key (e.g. ``"CosineAnnealingLR"``).
        optimizer: Optimizer passed to scheduler constructor.
        config: Keyword args passed to the scheduler constructor.
        registry: Mapping of name → scheduler class. Defaults to ``SCHEDULERS``.

    Returns:
        Instantiated ``LRScheduler``, or ``None`` if ``name`` is ``None``.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    return builder(name, {**config, "optimizer": optimizer}, registry)

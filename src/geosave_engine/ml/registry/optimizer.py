from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from geosave_engine.ml.registry.base import method_builder

import geosave_engine.ml.optimizer.adamw as adamw
import geosave_engine.ml.optimizer.adam as adam
import geosave_engine.ml.optimizer.sgd as sgd
import geosave_engine.ml.optimizer.rmsprop as rmsprop
import geosave_engine.ml.optimizer.adagrad as adagrad

OPTIMIZERS = {
    'AdamW': adamw,
    'Adam': adam,
    'SGD': sgd,
    'RMSprop': rmsprop,
    'Adagrad': adagrad,
}


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

from __future__ import annotations

import torch.nn as nn
from torch.nn import CrossEntropyLoss

from geosave_engine.ml.loss import ProbOhemCrossEntropy2d
from geosave_engine.ml.registry.base import builder

LOSSES = {
    'CELoss': CrossEntropyLoss,
    'OHEMLoss': ProbOhemCrossEntropy2d,
}


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

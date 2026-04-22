from __future__ import annotations

import torch
from geosave_engine.ml.core.base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss wrapper based on PyTorch's nn.CrossEntropyLoss."""

    doc_links = ["https://docs.geosave.dev/losses/cross_entropy"]
    loss = torch.nn.CrossEntropyLoss

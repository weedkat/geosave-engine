from __future__ import annotations

import torch
from geosave_engine.core.factory import BaseLossFactory


class CrossEntropyLoss(BaseLossFactory):
    """Cross-entropy loss wrapper based on PyTorch's nn.CrossEntropyLoss."""

    doc_links = ["https://docs.geosave.dev/losses/cross_entropy"]
    loss = torch.nn.CrossEntropyLoss

    @classmethod
    def full(cls, *args, **kwargs):
        """Build a standard CrossEntropyLoss instance for full-output supervision."""
        return cls.loss(*args, **kwargs)
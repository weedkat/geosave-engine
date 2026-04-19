from __future__ import annotations

import torch

from geosave_engine.core.base import BaseOptimizer
from geosave_engine.utils.torch_params import split_encoder_decoder_params


class AdamW(BaseOptimizer):
    """AdamW optimizer wrapper based on PyTorch's torch.optim.AdamW."""

    doc_links = ["https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html"]
    optimizer = torch.optim.AdamW

    @classmethod
    def default(cls, model, *args, **kwargs):
        """Build a standard AdamW optimizer over a single parameter iterable."""
        return cls.optimizer(model.parameters(), *args, **kwargs)

    @classmethod
    def split(cls, model, encoder_lr, decoder_lr, *args, **kwargs):
        """Build AdamW with separate learning rates for encoder and decoder groups."""
        encoder_params, decoder_params = split_encoder_decoder_params(model)
        return cls.optimizer(
            [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": decoder_params, "lr": decoder_lr},
            ],
            *args,
            **kwargs,
        )

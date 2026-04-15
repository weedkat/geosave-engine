from __future__ import annotations

import torch

from geosave_engine.core.factory import BaseOptimizerFactory


class AdamW(BaseOptimizerFactory):
    """AdamW optimizer wrapper based on PyTorch's torch.optim.AdamW."""

    doc_links = ["https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html"]
    optimizer = torch.optim.AdamW

    @classmethod
    def full(cls, params, *args, **kwargs):
        """Build a standard AdamW optimizer over a single parameter iterable."""
        return cls.optimizer(params, *args, **kwargs)

    @classmethod
    def split(cls, encoder_params, decoder_params, *args, **kwargs):
        """
        Build an AdamW optimizer with separate learning rates for encoder and decoder parameters.
        """
        return cls.optimizer([
            {"params": encoder_params, "lr": kwargs.get("encoder_lr", 1e-3)},
            {"params": decoder_params, "lr": kwargs.get("decoder_lr", 1e-3)},
        ], *args, **kwargs)

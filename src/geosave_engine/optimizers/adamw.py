from __future__ import annotations

import torch

from geosave_engine.core.base import BaseOptimizer


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
        """
        Build an AdamW optimizer with separate learning rates for encoder and decoder parameters.
        """
        def get_encoder_decoder_params(model):
            """Split parameters into encoder / decoder groups."""
            if hasattr(model, 'backbone'):
                enc = [p for p in model.backbone.parameters() if p.requires_grad]
                dec = [p for n, p in model.named_parameters() if 'backbone' not in n]
            elif hasattr(model, 'encoder'):
                enc = list(model.encoder.parameters())
                dec = [p for n, p in model.named_parameters() if not n.startswith('encoder')]
            else:
                raise ValueError("Model must have either 'backbone' or 'encoder' attribute to split parameters.")
            return enc, dec

        encoder_params, decoder_params = get_encoder_decoder_params(model)

        return cls.optimizer([
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ], *args, **kwargs)

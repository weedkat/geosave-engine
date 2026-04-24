from __future__ import annotations

import torch


def split_encoder_decoder_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition ``model.parameters()`` into encoder and decoder param lists.

    The model must expose either a ``backbone`` or ``encoder`` attribute; all
    other named parameters are treated as decoder params.

    Raises:
        ValueError: if the model exposes neither ``backbone`` nor ``encoder``.
    """
    if hasattr(model, "backbone"):
        encoder = [p for p in model.backbone.parameters() if p.requires_grad]
        decoder = [p for name, p in model.named_parameters() if "backbone" not in name]
        return encoder, decoder

    if hasattr(model, "encoder"):
        encoder = list(model.encoder.parameters())
        decoder = [p for name, p in model.named_parameters() if not name.startswith("encoder")]
        return encoder, decoder

    raise ValueError(
        "Model must expose either a 'backbone' or 'encoder' attribute to split parameters."
    )

from __future__ import annotations

from typing import Any, Iterable


def split_encoder_decoder_params(model: Any) -> tuple[list[Any], list[Any]]:
    """Partition `model.parameters()` into encoder and decoder param lists.

    The model must expose either a `backbone` or `encoder` attribute; all other
    named parameters are treated as decoder params.
    """
    if hasattr(model, "backbone"):
        encoder: list[Any] = [p for p in model.backbone.parameters() if p.requires_grad]
        decoder: list[Any] = [p for name, p in model.named_parameters() if "backbone" not in name]
        return encoder, decoder

    if hasattr(model, "encoder"):
        encoder = list(model.encoder.parameters())
        decoder = [p for name, p in model.named_parameters() if not name.startswith("encoder")]
        return encoder, decoder

    raise ValueError(
        "Model must expose either a 'backbone' or 'encoder' attribute to split parameters."
    )


def filter_requires_grad(params: Iterable[Any]) -> list[Any]:
    """Keep only parameters with `requires_grad=True`."""
    return [p for p in params if p.requires_grad]

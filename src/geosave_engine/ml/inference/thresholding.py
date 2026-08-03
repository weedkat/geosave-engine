from __future__ import annotations

import torch


def softmax_argmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over the class dim, then argmax + top-class confidence.

    Args:
        logits: `[B, num_classes, H, W]` raw model output.

    Returns:
        `(preds [B, H, W] argmax class, max_probs [B, H, W] top-class confidence)`.
    """
    probs = logits.softmax(dim=1)
    max_probs, preds = probs.max(dim=1)
    return preds, max_probs


def apply_thresholds(
    logits: torch.Tensor,
    thresholds: torch.Tensor,
    ignore_index: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax + per-class confidence threshold + optional nodata mask.

    Args:
        logits: `[B, num_classes, H, W]` raw model output.
        thresholds: `[num_classes]` per-class confidence threshold.
        ignore_index: Class index assigned to low-confidence/masked pixels.
        mask: Optional boolean `[B, H, W]` nodata mask. Masked pixels -> ignore_index.

    Returns:
        `(preds [B, H, W], max_probs [B, H, W] float32)`.
    """
    preds, max_probs = softmax_argmax(logits)

    pixel_thresholds = torch.index_select(thresholds, 0, preds.reshape(-1)).view_as(preds)
    preds = torch.where(max_probs >= pixel_thresholds, preds, preds.new_full((), ignore_index))

    if mask is not None:
        preds = torch.where(mask.bool(), preds.new_full((), ignore_index), preds)

    return preds, max_probs

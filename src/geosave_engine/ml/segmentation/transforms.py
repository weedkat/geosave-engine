"""Interpret dense semantic-segmentation logits."""

import torch


def softmax_argmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top class and confidence for each pixel."""
    probabilities = logits.softmax(dim=1)
    confidence, labels = probabilities.max(dim=1)
    return labels, confidence


def apply_thresholds(
    logits: torch.Tensor,
    thresholds: torch.Tensor,
    ignore_index: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply class thresholds and nodata to dense logits.

    Args:
        logits: Raw model output shaped ``[B, classes, H, W]``.
        thresholds: Confidence threshold for each class.
        ignore_index: Label assigned to rejected or masked pixels.
        mask: Optional boolean nodata mask shaped ``[B, H, W]``.

    Returns:
        Uint8 labels and float32 top-class probabilities shaped ``[B, H, W]``.
    """
    labels, confidence = softmax_argmax(logits)
    pixel_thresholds = torch.index_select(thresholds, 0, labels.reshape(-1)).view_as(
        labels
    )
    labels = torch.where(
        confidence >= pixel_thresholds,
        labels,
        labels.new_full((), ignore_index),
    )

    if mask is not None:
        labels = torch.where(mask.bool(), labels.new_full((), ignore_index), labels)

    return labels.to(torch.uint8), confidence.to(torch.float32)

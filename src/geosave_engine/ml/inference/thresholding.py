from __future__ import annotations

from typing import Callable

import torch
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_jaccard_index

_METRIC_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "f1": multiclass_f1_score,
    "iou": multiclass_jaccard_index,
}


class ClassThresholding:
    """Apply and calibrate per-class confidence thresholds for dense classification.

    ``apply`` turns raw logits into thresholded predictions given a
    thresholds tensor; ``sweep`` searches each class's best threshold from
    already-computed predictions/labels. Neither touches batches, keys, or
    a model — that's the caller's job (see ``SemanticSegmentationTask``'s
    ``postprocess``/``on_fit_end``).

    Args:
        num_classes: Number of classes to calibrate/apply a threshold for.
        ignore_index: Label value excluded from calibration and assigned to
            low-confidence/masked pixels.
        threshold_begin: Start of sweep range.
        threshold_end: End of sweep range.
        threshold_steps: Number of candidate thresholds to try per class.
        metric: Metric to maximise during calibration — ``'f1'`` or ``'iou'``.

    Raises:
        ValueError: If ``metric`` is not ``'f1'`` or ``'iou'``.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        threshold_begin: float = 0.0,
        threshold_end: float = 1.0,
        threshold_steps: int = 100,
        metric: str = "f1",
    ) -> None:
        if metric not in _METRIC_FNS:
            raise ValueError(f"metric must be one of {list(_METRIC_FNS)}, got {metric!r}")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold_range = torch.linspace(threshold_begin, threshold_end, threshold_steps)
        self._metric_fn = _METRIC_FNS[metric]

    def softmax_argmax(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax over the class dim, then argmax + top-class confidence.

        Args:
            logits: ``[B, num_classes, H, W]`` raw model output.

        Returns:
            ``(preds [B, H, W] argmax class, max_probs [B, H, W] top-class confidence)``.
        """
        probs = logits.softmax(dim=1)
        max_probs, preds = probs.max(dim=1)
        return preds, max_probs

    def apply(
        self,
        logits: torch.Tensor,
        thresholds: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmax + per-class confidence threshold + optional nodata mask.

        Args:
            logits: ``[B, num_classes, H, W]`` raw model output.
            thresholds: ``[num_classes]`` per-class confidence threshold.
            mask: Optional boolean ``[B, H, W]`` nodata mask. Masked pixels → ignore_index.

        Returns:
            ``(pred_label [B, H, W], pred_proba [B, H, W] float32)``.
        """
        preds, max_probs = self.softmax_argmax(logits)

        pixel_thresholds = torch.index_select(thresholds, 0, preds.reshape(-1)).view_as(preds)
        preds = torch.where(max_probs >= pixel_thresholds, preds, preds.new_full((), self.ignore_index))

        if mask is not None:
            preds = torch.where(mask.bool(), preds.new_full((), self.ignore_index), preds)

        return preds, max_probs

    def sweep(
        self,
        preds: torch.Tensor,
        max_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Search each class's threshold that maximises ``metric``.

        Args:
            preds: Flat ``[N]`` argmax class predictions (unthresholded).
            max_probs: Flat ``[N]`` top-class confidence, same order as ``preds``.
            labels: Flat ``[N]`` ground-truth class, same order as ``preds``.

        Returns:
            ``[num_classes]`` best threshold per class.
        """
        num_classes, ignore_index = self.num_classes, self.ignore_index
        new_thresholds = torch.full((num_classes,), 0.5)

        # Remap GT nodata (ignore_index) → num_classes so values stay in [0, num_classes].
        # num_classes acts as the abstain/ignore slot; metric ignores it via ignore_index=num_classes.
        labels_r = labels.clone()
        labels_r[labels_r == ignore_index] = num_classes

        for c in range(num_classes):
            best_t, best_score = 0.5, -1.0
            for t in self.threshold_range:
                adjusted = preds.clone()
                # suppress low-confidence class-c predictions → abstain slot
                adjusted[(preds == c) & (max_probs < t)] = num_classes

                score = self._metric_fn(
                    adjusted,
                    labels_r,
                    num_classes=num_classes + 1,  # +1 for abstain/ignore slot at index num_classes
                    average="macro",
                    ignore_index=num_classes,
                ).item()

                if score > best_score:
                    best_score = score
                    best_t = t.item()

            new_thresholds[c] = best_t

        return new_thresholds

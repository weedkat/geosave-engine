from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_jaccard_index

from geosave_engine.ml.inference.thresholding import softmax_argmax

_METRIC_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "f1": multiclass_f1_score,
    "iou": multiclass_jaccard_index,
}


def _sweep(
    preds: torch.Tensor,
    max_probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    threshold_range: torch.Tensor,
    metric_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Search each class's threshold that maximises `metric_fn`.

    Args:
        preds: Flat `[N]` argmax class predictions (unthresholded).
        max_probs: Flat `[N]` top-class confidence, same order as `preds`.
        labels: Flat `[N]` ground-truth class, same order as `preds`.
        num_classes: Number of classes to calibrate a threshold for.
        ignore_index: Label value excluded from calibration.
        threshold_range: Candidate thresholds to try per class.
        metric_fn: Metric to maximise, e.g. `multiclass_f1_score`.

    Returns:
        `[num_classes]` best threshold per class.
    """
    new_thresholds = torch.full((num_classes,), 0.5)

    # Remap GT nodata (ignore_index) → num_classes so values stay in [0, num_classes].
    # num_classes acts as the abstain/ignore slot; metric ignores it via ignore_index=num_classes.
    labels_r = labels.clone()
    labels_r[labels_r == ignore_index] = num_classes

    for c in range(num_classes):
        best_t, best_score = 0.5, -1.0
        for t in threshold_range:
            adjusted = preds.clone()
            # suppress low-confidence class-c predictions → abstain slot
            adjusted[(preds == c) & (max_probs < t)] = num_classes

            score = metric_fn(
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


class ThresholdCalibrator(Callback):
    """Calibrate per-class confidence thresholds from validation data, once, near the end of training.

    Reacts to the same ``on_validation_batch_end`` outputs
    ``DensePredictionLogger`` reads (``{'logits': ..., 'label': ...}``) — no
    manual forward pass, no ``eval()``/``no_grad()``/dataloader plumbing, no
    ``image_key``/``label_key`` coupling.

    Only accumulates on the last validation epoch (``current_epoch ==
    max_epochs - 1``) — every earlier epoch's ``on_validation_batch_end``
    is a no-op, so there's no per-epoch clearing to manage. Skips
    Lightning's own pre-training sanity check validation pass, which fires
    the same hooks with the same ``current_epoch`` before any training has
    happened.

    Sweeps and writes ``pl_module.class_thresholds`` from ``on_validation_end``,
    not ``on_fit_end`` — ``on_fit_end`` fires after every ``ModelCheckpoint``
    save for that run, so a calibrated buffer written there would never reach
    a saved checkpoint. ``GeosaveCLI`` appends ``ModelCheckpoint`` after
    user-declared callbacks (so this one runs first), meaning the buffer is
    already updated by the time that epoch's checkpoint save happens.

    Args:
        num_classes: Number of classes to calibrate a threshold for.
        ignore_index: Label value excluded from calibration.
        threshold_begin: Start of sweep range.
        threshold_end: End of sweep range.
        threshold_steps: Number of candidate thresholds to try per class.
        metric: Metric to maximise during calibration — ``'f1'`` or ``'iou'``.

    Raises:
        ValueError: ``metric`` isn't ``'f1'``/``'iou'``, or ``Trainer(max_epochs=...)``
            isn't set to a real count — can't tell which epoch is the last one.
        TypeError: ``validation_step`` didn't return a ``logits``/``label``
            dict, or ``pl_module`` has no ``class_thresholds`` buffer to write.
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
        super().__init__()
        if metric not in _METRIC_FNS:
            raise ValueError(f"metric must be one of {list(_METRIC_FNS)}, got {metric!r}")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._threshold_range = torch.linspace(threshold_begin, threshold_end, threshold_steps)
        self._metric_fn = _METRIC_FNS[metric]
        self._preds: list[torch.Tensor] = []
        self._max_probs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def _is_last_epoch(self, trainer: Trainer) -> bool:
        if trainer.sanity_checking:
            return False
        if trainer.max_epochs is None or trainer.max_epochs < 0:
            raise ValueError(
                f"{type(self).__name__} needs Trainer(max_epochs=...) set to a real count "
                f"to know which epoch is the last one to calibrate from — got {trainer.max_epochs!r}."
            )
        return trainer.current_epoch == trainer.max_epochs - 1

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._is_last_epoch(trainer):
            return
        # Exclude the other two union members instead of a positive Mapping/dict
        # isinstance check — Tensor structurally overlaps enough of Mapping's
        # protocol to confuse the checker's narrowing on a positive check.
        if outputs is None or isinstance(outputs, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} expects validation_step to return "
                f"a {{'logits': ..., 'label': ...}} dict, got {type(outputs).__name__}."
            )
        logits, label = outputs.get('logits'), outputs.get('label')
        if not (isinstance(logits, torch.Tensor) and isinstance(label, torch.Tensor)):
            raise TypeError(
                f"{type(self).__name__} expects outputs['logits']/['label'] to be tensors, "
                f"got logits={type(logits).__name__}, label={type(label).__name__}."
            )
        preds, max_probs = softmax_argmax(logits)
        self._preds.append(preds.detach().cpu())
        self._max_probs.append(max_probs.detach().cpu())
        self._labels.append(label.detach().cpu())

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._is_last_epoch(trainer) or not self._preds:
            return
        class_thresholds = getattr(pl_module, 'class_thresholds', None)
        if not isinstance(class_thresholds, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} expects pl_module.class_thresholds to be a "
                f"registered buffer (Tensor), got {type(class_thresholds).__name__}."
            )

        # flatten [B,H,W] → [N] for pixel-wise metric computation
        all_preds = torch.cat(self._preds).view(-1)
        all_max_probs = torch.cat(self._max_probs).view(-1)
        all_labels = torch.cat(self._labels).view(-1)

        new_thresholds = _sweep(
            all_preds, all_max_probs, all_labels,
            self.num_classes, self.ignore_index, self._threshold_range, self._metric_fn,
        )
        class_thresholds.copy_(new_thresholds.to(pl_module.device))
        self._preds.clear()
        self._max_probs.clear()
        self._labels.clear()

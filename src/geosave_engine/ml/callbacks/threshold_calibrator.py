from __future__ import annotations

from typing import Any, Mapping

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from geosave_engine.ml.inference.thresholding import ClassThresholding


class ThresholdCalibrator(Callback):
    """Calibrate per-class confidence thresholds from validation data, once, after training.

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

    Args:
        num_classes: Number of classes to calibrate a threshold for.
        ignore_index: Label value excluded from calibration.
        threshold_begin: Start of sweep range.
        threshold_end: End of sweep range.
        threshold_steps: Number of candidate thresholds to try per class.
        metric: Metric to maximise during calibration — ``'f1'`` or ``'iou'``.

    Raises:
        ValueError: ``Trainer(max_epochs=...)`` isn't set to a real count —
            can't tell which epoch is the last one.
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
        self._thresholding = ClassThresholding(
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold_begin=threshold_begin,
            threshold_end=threshold_end,
            threshold_steps=threshold_steps,
            metric=metric,
        )
        self._preds: list[torch.Tensor] = []
        self._max_probs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        if trainer.max_epochs is None or trainer.max_epochs < 0:
            raise ValueError(
                f"{type(self).__name__} needs Trainer(max_epochs=...) set to a real count "
                f"to know which epoch is the last one to calibrate from — got {trainer.max_epochs!r}."
            )
        if trainer.current_epoch != trainer.max_epochs - 1:
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
        preds, max_probs = self._thresholding.softmax_argmax(logits)
        self._preds.append(preds.detach().cpu())
        self._max_probs.append(max_probs.detach().cpu())
        self._labels.append(label.detach().cpu())

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._preds:
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

        new_thresholds = self._thresholding.sweep(all_preds, all_max_probs, all_labels)
        class_thresholds.copy_(new_thresholds.to(pl_module.device))

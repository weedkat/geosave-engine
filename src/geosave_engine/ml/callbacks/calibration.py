from __future__ import annotations

from typing import Callable

import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_jaccard_index


_METRIC_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "f1": multiclass_f1_score,
    "iou": multiclass_jaccard_index,
}


class DenseCalibrationCallback(Callback):
    """Per-class confidence threshold calibration, run once at end of fit.

    Sweeps candidate thresholds over val set after training. For each class,
    picks threshold that maximises ``metric``. Writes result into
    ``pl_module.class_thresholds``. Reads batch via ``pl_module.image_key``/
    ``label_key`` — same keys the task itself uses, no fixed batch layout required.

    Args:
        threshold_begin: Start of sweep range.
        threshold_end: End of sweep range.
        threshold_steps: Number of candidate thresholds to try per class.
        metric: Metric to maximise — ``'f1'`` or ``'iou'``.

    Raises:
        ValueError: If ``metric`` is not ``'f1'`` or ``'iou'``.
    """

    def __init__(
        self,
        threshold_begin: float = 0.0,
        threshold_end: float = 1.0,
        threshold_steps: int = 100,
        metric: str = "f1",
    ) -> None:
        super().__init__()
        if metric not in _METRIC_FNS:
            raise ValueError(f"metric must be one of {list(_METRIC_FNS)}, got {metric!r}")
        self.threshold_range = torch.linspace(threshold_begin, threshold_end, threshold_steps)
        self._metric_fn = _METRIC_FNS[metric]

    def on_fit_end(self, trainer, pl_module) -> None:
        """Collect val predictions, sweep thresholds, write best per class."""
        if trainer.datamodule is None:
            return

        loader = trainer.val_dataloaders
        if isinstance(loader, list):
            loader = loader[0]

        val_preds: list[torch.Tensor] = []
        val_max_probs: list[torch.Tensor] = []
        val_labels: list[torch.Tensor] = []

        pl_module.eval()
        with torch.no_grad():
            for batch in loader:
                image = batch[pl_module.image_key].to(pl_module.device)
                label = batch[pl_module.label_key].to(pl_module.device)
                context = batch.get("context", {})

                logits = pl_module(image, **context)
                probs = logits.softmax(dim=1)
                # max_probs: [B,H,W] top confidence; preds: [B,H,W] argmax class
                max_probs, preds = probs.max(dim=1)

                val_preds.append(preds.detach().cpu())
                val_max_probs.append(max_probs.detach().cpu())
                val_labels.append(label.detach().cpu())

        if not val_preds:
            return

        # flatten [B,H,W] → [N] for pixel-wise metric computation
        all_preds = torch.cat(val_preds).view(-1)
        all_max_probs = torch.cat(val_max_probs).view(-1)
        all_labels = torch.cat(val_labels).view(-1)

        n_classes: int = pl_module.num_classes
        ignore_index: int = pl_module.ignore_index
        new_thresholds = torch.full((n_classes,), 0.5)

        # Remap GT nodata (ignore_index=255) → n_classes so values stay in [0, n_classes].
        # n_classes acts as the abstain/ignore slot; metric ignores it via ignore_index=n_classes.
        labels_r = all_labels.clone()
        labels_r[labels_r == ignore_index] = n_classes

        for c in range(n_classes):
            best_t, best_score = 0.5, -1.0
            for t in self.threshold_range:
                adjusted = all_preds.clone()
                # suppress low-confidence class-c predictions → abstain slot
                adjusted[(all_preds == c) & (all_max_probs < t)] = n_classes

                score = self._metric_fn(
                    adjusted,
                    labels_r,
                    num_classes=n_classes + 1,  # +1 for abstain/ignore slot at index n_classes
                    average="macro",
                    ignore_index=n_classes,
                ).item()

                if score > best_score:
                    best_score = score
                    best_t = t.item()

            new_thresholds[c] = best_t

        pl_module.class_thresholds.copy_(new_thresholds.to(pl_module.device))

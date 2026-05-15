from __future__ import annotations

from typing import Callable

import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_jaccard_index


_METRIC_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "f1": multiclass_f1_score,
    "iou": multiclass_jaccard_index,
}


class CalibrationCallback(Callback):
    """Calibrates per-class confidence thresholds once at the end of fit.

    Runs a single forward pass over the validation dataloader after training,
    sweeps ``threshold_steps`` points in ``[threshold_begin, threshold_end]``
    per class, and picks the threshold that maximises ``metric`` (f1 or iou)
    evaluated over all classes. Learned thresholds are written into
    ``pl_module.class_thresholds``.
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
                image = batch["image"].to(pl_module.device)
                label = batch["label"].to(pl_module.device)
                mask = batch.get("mask")
                if mask is not None:
                    label = label & mask.to(pl_module.device)

                logits = pl_module.model(image)
                probs = logits.softmax(dim=1)
                max_probs, preds = probs.max(dim=1)

                val_preds.append(preds.detach().cpu())
                val_max_probs.append(max_probs.detach().cpu())
                val_labels.append(label.detach().cpu())

        if not val_preds:
            return

        self._compute_and_apply_thresholds(pl_module, val_preds, val_max_probs, val_labels)

    def _compute_and_apply_thresholds(
        self,
        pl_module,
        val_preds: list[torch.Tensor],
        val_max_probs: list[torch.Tensor],
        val_labels: list[torch.Tensor],
    ) -> None:
        all_preds = torch.cat(val_preds).view(-1)
        all_max_probs = torch.cat(val_max_probs).view(-1)
        all_labels = torch.cat(val_labels).view(-1)

        n_classes: int = pl_module.num_classes
        ignore_index: int = pl_module.ignore_index
        new_thresholds = torch.full((n_classes,), 0.5)

        for c in range(n_classes):
            best_t, best_score = 0.5, -1.0
            for t in self.threshold_range:
                adjusted = all_preds.clone()
                adjusted[(all_preds == c) & (all_max_probs < t)] = ignore_index

                score = self._metric_fn(
                    adjusted,
                    all_labels,
                    num_classes=n_classes + 1,  # add one for ignore_index class
                    average="macro",
                ).item()

                if score > best_score:
                    best_score = score
                    best_t = t.item()

            new_thresholds[c] = best_t

        pl_module.class_thresholds.copy_(new_thresholds.to(pl_module.device))

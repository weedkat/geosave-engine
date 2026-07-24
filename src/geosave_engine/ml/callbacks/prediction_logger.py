from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from geosave_engine.utils import colorize
from geosave_engine.geodata.utils.geovis import fig_to_array


class DensePredictionLogger(Callback):
    """Log a colorized label/prediction panel with a class legend, periodically, during val/test.

    Dense/per-pixel classification only — ``validation_step``/``test_step``
    must return ``{'logits': ..., 'label': ...}``, ``logits`` shaped ``[B,
    num_classes, H, W]``. A classification or object detection task needs a
    different callback; its output isn't a per-pixel class map, so nothing
    here would apply.

    Reacts to Lightning's own per-batch hooks — no separate pass over data.
    No ``pl_module`` calls of any kind: no postprocess, no per-class
    thresholds, no nodata masking — those only reflect calibrated
    thresholds, which don't exist until ``on_fit_end`` runs at the very end
    of training, so applying them mid-training would just be a flat 0.5
    cutoff, no more informative than argmax.

    Args:
        color_map: ``{class_id: hex_color}``.
        class_map: ``{class_id: class_name}``, for the legend. A class with
            no entry falls back to its bare id.
        log_image_every_n_epochs: Epoch frequency to log at.

    Raises:
        TypeError: ``validation_step``/``test_step`` didn't return a
            ``logits``/``label`` dict.
        ValueError: ``logits`` aren't ``[B, num_classes, H, W]``.
    """

    def __init__(
        self,
        color_map: dict[int, str],
        class_map: dict[int, str] | None = None,
        log_image_every_n_epochs: int = 2,
    ) -> None:
        super().__init__()
        self.color_map = color_map
        self.class_map = class_map or {}
        self.log_image_every_n_epochs = log_image_every_n_epochs

    def _render(self, label: torch.Tensor, preds: torch.Tensor) -> np.ndarray:
        fig, (ax_label, ax_pred) = plt.subplots(1, 2, figsize=(8, 4))
        ax_label.imshow(colorize(label, self.color_map))
        ax_label.set_title('label', fontsize=10)
        ax_pred.imshow(colorize(preds, self.color_map))
        ax_pred.set_title('prediction', fontsize=10)
        for ax in (ax_label, ax_pred):
            ax.set_xticks([])
            ax.set_yticks([])

        handles = [
            Patch(color=color, label=self.class_map.get(cls, str(cls)))
            for cls, color in sorted(self.color_map.items())
        ]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.9, 1))

        image = fig_to_array(fig)
        plt.close(fig)
        return image

    def _log(
        self,
        trainer: Trainer,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch_idx: int,
        prefix: str,
    ) -> None:
        if batch_idx != 0 or trainer.current_epoch % self.log_image_every_n_epochs != 0:
            return
        # Exclude the other two union members instead of a positive Mapping/dict
        # isinstance check — Tensor structurally overlaps enough of Mapping's
        # protocol to confuse the checker's narrowing on a positive check.
        if outputs is None or isinstance(outputs, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} expects validation_step/test_step to return "
                f"a {{'logits': ..., 'label': ...}} dict, got {type(outputs).__name__}."
            )
        logits, label = outputs.get('logits'), outputs.get('label')
        if not (isinstance(logits, torch.Tensor) and isinstance(label, torch.Tensor)):
            raise TypeError(
                f"{type(self).__name__} expects outputs['logits']/['label'] to be tensors, "
                f"got logits={type(logits).__name__}, label={type(label).__name__}."
            )
        if logits.dim() != 4:
            raise ValueError(
                f"{type(self).__name__} only handles dense output — expected logits shaped "
                f"[B, num_classes, H, W], got {tuple(logits.shape)}."
            )
        if label.dim() == 4:  # (B, 1, H, W) → (B, H, W)
            label = label.squeeze(1)

        preds = logits.argmax(dim=1)
        image = self._render(label[0], preds[0])
        step = trainer.current_epoch
        for lg in trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_image(f'{prefix}/prediction', image, step, dataformats='HWC')
            elif isinstance(lg, MLFlowLogger):
                lg.experiment.log_image(lg.run_id, image, f'{prefix}_prediction_{step}.png')

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._log(trainer, outputs, batch_idx, 'val')

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._log(trainer, outputs, batch_idx, 'test')

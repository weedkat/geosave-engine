from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import xarray as xr
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from geosave_engine.geodata.utils.color import Palette, parse_color


def _colorize(
    mask: np.ndarray | torch.Tensor,
    palette: Palette,
    default: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Map a class-index mask to an RGB image.

    Args:
        mask: ``(H, W)`` array of integer class indices. Accepts numpy array or
            torch Tensor; tensors are converted to numpy automatically.
        palette: Mapping from class index to RGB tuple ``(R, G, B)`` or hex
            string ``"#RRGGBB"``. Values are clamped to ``[0, 255]``.
        default: RGB colour for indices absent from ``palette``. Defaults to
            black ``(0, 0, 0)``.

    Returns:
        ``(H, W, 3)`` uint8 numpy array suitable for JPEG/PNG saving or logger
        consumption.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = np.asarray(mask)
    h, w = mask.shape
    rgb = np.full((h, w, 3), default, dtype=np.uint8)

    parsed: dict[int, tuple[int, int, int]] = {
        idx: parse_color(color) for idx, color in palette.items()
    }

    for idx, color in parsed.items():
        rgb[mask == idx] = color

    return rgb


def _fig_to_array(fig: Figure) -> np.ndarray:
    """Rasterize a drawn figure into an image array.

    Args:
        fig: Figure drawn on an Agg-backed canvas.

    Returns:
        Rendered pixels as uint8, shaped `(height, width, 3)`, alpha dropped
        because both loggers take three channels.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())  # (H, W, 4)
    return rgba[..., :3].copy()  # (H, W, 3)


class DensePredictionLogger(Callback):
    """Log label and argmax prediction panels to TensorBoard or MLflow.

    Validation and test steps must return logits shaped ``[B, C, H, W]`` and
    labels shaped ``[B, H, W]`` or ``[B, 1, H, W]``. Only the first sample of
    each loader is logged, on rank zero, outside sanity checks.

    Args:
        color_map: ``{class_id: hex_color}``.
        class_map: ``{class_id: class_name}``, for the legend. A class with
            no entry falls back to its bare id.
        log_image_every_n_epochs: Epoch frequency to log at.

    Raises:
        TypeError: ``validation_step``/``test_step`` didn't return a
            ``logits``/``label`` dict.
        ValueError: The epoch interval is not positive, or logits are not 4D.
    """

    def __init__(
        self,
        color_map: dict[int, str],
        class_map: dict[int, str] | None = None,
        log_image_every_n_epochs: int = 2,
    ) -> None:
        super().__init__()
        if log_image_every_n_epochs < 1:
            raise ValueError("log_image_every_n_epochs must be positive")
        self.color_map = color_map
        self.class_map = class_map or {}
        self.log_image_every_n_epochs = log_image_every_n_epochs

    def _render(self, label: torch.Tensor, preds: torch.Tensor) -> np.ndarray:
        fig, (ax_label, ax_pred) = plt.subplots(1, 2, figsize=(8, 4))
        ax_label.imshow(_colorize(label, self.color_map))
        ax_label.set_title("label", fontsize=10)
        ax_pred.imshow(_colorize(preds, self.color_map))
        ax_pred.set_title("prediction", fontsize=10)
        for ax in (ax_label, ax_pred):
            ax.set_xticks([])
            ax.set_yticks([])

        handles = [
            Patch(color=color, label=self.class_map.get(cls, str(cls)))
            for cls, color in sorted(self.color_map.items())
        ]
        fig.legend(
            handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=8
        )
        fig.tight_layout(rect=(0, 0, 0.9, 1))

        image = _fig_to_array(fig)
        plt.close(fig)
        return image

    def _log(
        self,
        trainer: Trainer,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch_idx: int,
        prefix: str,
        dataloader_idx: int,
    ) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if batch_idx != 0 or trainer.current_epoch % self.log_image_every_n_epochs != 0:
            return
        loggers = [
            logger
            for logger in trainer.loggers
            if isinstance(logger, (TensorBoardLogger, MLFlowLogger))
        ]
        if not loggers:
            return
        if outputs is None or isinstance(outputs, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} expects validation_step/test_step to return "
                f"a {{'logits': ..., 'label': ...}} dict, got {type(outputs).__name__}."
            )
        logits, label = outputs.get("logits"), outputs.get("label")
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
        for lg in loggers:
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_image(
                    f"{prefix}/dataloader_{dataloader_idx}/prediction",
                    image,
                    step,
                    dataformats="HWC",
                )
            elif isinstance(lg, MLFlowLogger):
                lg.experiment.log_image(
                    lg.run_id,
                    image,
                    f"{prefix}_dataloader_{dataloader_idx}_prediction_{step}.png",
                )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._log(trainer, outputs, batch_idx, "val", dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._log(trainer, outputs, batch_idx, "test", dataloader_idx)


class ChipWriter(BasePredictionWriter):
    """Save each batch of chip predictions, keyed by window id, to merge later.

    Nothing is merged while predicting, so chips of one raster may be
    predicted in any order and on several devices; each process appends to a
    store of its own. `cuts.merge` puts them back together afterwards.

    Args:
        output_dir: Folder taking one `chips-<rank>.zarr` per process.
        name: Name the saved output carries, and the merged raster after it.

    Examples:
        >>> trainer = Trainer(callbacks=[ChipWriter("runs/predict/chips")])
        >>> trainer.predict(module, dataloaders=loader, return_predictions=False)
        >>> outputs = ChipWriter.read("runs/predict/chips")
        >>> merged = cuts.merge(frames, chips, outputs, taper="hann")
    """

    def __init__(self, output_dir: str | Path, *, name: str = "logits") -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.name = name

    def _store(self, trainer: Trainer) -> Path:
        """Return the store this process writes."""
        return self.output_dir / f"chips-{trainer.global_rank}.zarr"

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Refuse to append to chips an earlier run left behind.

        Raises:
            FileExistsError: This process's store already exists.
        """
        store = self._store(trainer)
        if store.exists():
            raise FileExistsError(
                f"{store} holds chips of an earlier run, which a merge would mix "
                f"with this one's; write to another folder or remove it"
            )

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Append one batch of `(outputs, ids)` to this process's store."""
        outputs, ids = prediction
        values = outputs.detach().float().cpu().numpy()
        dims = ("id", "band", "y", "x") if values.ndim == 4 else ("id", "y", "x")
        chips = xr.Dataset(
            {self.name: (dims, values)},
            coords={"id": np.asarray(list(ids), dtype=object)},
        )
        store = self._store(trainer)
        if store.exists():
            chips.to_zarr(store, append_dim="id")
        else:
            chips.to_zarr(store, mode="w")

    @staticmethod
    def read(output_dir: str | Path) -> xr.DataArray:
        """Open every process's saved chips as one lazy array along `id`.

        Args:
            output_dir: Folder a `ChipWriter` wrote to.

        Returns:
            The saved outputs, one per chip, in the order they were written.

        Raises:
            FileNotFoundError: The folder holds no saved chips.
        """
        stores = sorted(Path(output_dir).glob("chips-*.zarr"))
        if not stores:
            raise FileNotFoundError(f"{output_dir} holds no saved chips")
        parts = [xr.open_zarr(store) for store in stores]
        (name,) = parts[0].data_vars
        return xr.concat(parts, "id")[name]

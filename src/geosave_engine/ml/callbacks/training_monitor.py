"""Rich-powered Lightning progress bar with per-epoch telemetry panels."""
from __future__ import annotations

from time import perf_counter
from typing import Sequence

import torch
from lightning.pytorch.callbacks import RichProgressBar
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


METRIC_PREFIXES: tuple[str, ...] = ("train_", "val_", "test_")


class LiveTrainingMonitor(RichProgressBar):
    """Extends ``RichProgressBar`` with a fit-start banner and per-epoch summary tables.

    Drop-in replacement for the default progress bar. Adds:
      - Fit-start banner showing experiment name, model class, max epochs, optimizer(s).
      - Per-epoch summary table with train/val losses + every other train_/val_
        metric in ``trainer.callback_metrics``, plus LR(s) and epoch wall time.
      - End-of-fit test summary (when ``trainer.test`` is invoked next) shows
        every ``test_`` metric.

    Args:
        summary_metrics: Optional explicit ordering of metric keys to surface.
            When ``None`` (default), every scalar in ``trainer.callback_metrics``
            with a ``train_/val_/test_`` prefix is shown.
    """

    def __init__(
        self,
        summary_metrics: Sequence[str] | None = None,
        refresh_rate: int = 1,
        leave: bool = False,
    ) -> None:
        super().__init__(refresh_rate=refresh_rate, leave=leave)
        self.summary_metrics = tuple(summary_metrics) if summary_metrics else None
        self._console = Console()
        self._epoch_start: float | None = None

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer, pl_module) -> None:
        super().on_fit_start(trainer, pl_module)
        self._console.print(self._fit_banner(trainer, pl_module))

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self._epoch_start = perf_counter()

    def render_epoch_summary(self, trainer, metrics: dict) -> None:
        """Print the epoch summary panel from a metric dict.

        Called by the LightningModule's ``on_train_epoch_end`` once both train
        and val metrics are computed — the only hook order in which both are
        available together.
        """
        if trainer.sanity_checking:
            return
        self._console.print(self._epoch_summary(trainer, metrics))

    def on_test_end(self, trainer, pl_module) -> None:
        super().on_test_end(trainer, pl_module)
        self._console.print(self._test_summary(trainer))

    def on_fit_end(self, trainer, pl_module) -> None:
        super().on_fit_end(trainer, pl_module)
        self._console.rule("[bold green]Training complete")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _fit_banner(self, trainer, pl_module) -> Panel:
        exp_name = getattr(trainer, "logger", None)
        exp_name = getattr(exp_name, "name", None) or pl_module.__class__.__name__

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("experiment", str(exp_name))
        table.add_row("model", pl_module.__class__.__name__)
        table.add_row("max_epochs", str(trainer.max_epochs))
        table.add_row("devices", f"{trainer.num_devices} × {trainer.accelerator.__class__.__name__}")
        num_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        table.add_row("trainable_params", f"{num_params:,}")
        return Panel(table, title="[bold]geosave training", border_style="cyan")

    def _epoch_summary(self, trainer, metrics=None) -> Panel:
        metrics = metrics if metrics is not None else trainer.callback_metrics
        keys = self._select_keys(metrics, prefixes=("train_", "val_"))
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("metric", style="bold cyan")
        table.add_column("value", justify="right")

        for key in keys:
            table.add_row(key, self._format_value(metrics.get(key)))

        for idx, lr in enumerate(self._current_lrs(trainer)):
            label = "lr" if len(trainer.optimizers) == 1 else f"lr[{idx}]"
            table.add_row(label, f"{lr:.3e}")

        elapsed = None
        if self._epoch_start is not None:
            elapsed = perf_counter() - self._epoch_start
            self._epoch_start = None
        if elapsed is not None:
            table.add_row("epoch_time_s", f"{elapsed:.1f}")

        title = f"[bold]epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        return Panel(table, title=title, border_style="green")

    def _test_summary(self, trainer) -> Panel:
        metrics = trainer.callback_metrics
        keys = self._select_keys(metrics, prefixes=("test_",))
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("metric", style="bold cyan")
        table.add_column("value", justify="right")
        for key in keys:
            table.add_row(key, self._format_value(metrics.get(key)))
        return Panel(table, title="[bold]test summary", border_style="green")

    def _select_keys(self, metrics, prefixes: tuple[str, ...]) -> list[str]:
        """Return metric keys to display, ordered by user override or prefix."""
        if self.summary_metrics is not None:
            return [k for k in self.summary_metrics if k in metrics]
        keys = [k for k in metrics if any(k.startswith(p) for p in prefixes)]
        scalar_keys = [k for k in keys if self._is_scalar(metrics.get(k))]
        scalar_keys.sort(key=lambda k: (next(i for i, p in enumerate(prefixes) if k.startswith(p)), k))
        return scalar_keys

    @staticmethod
    def _is_scalar(value) -> bool:
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            return value.numel() == 1
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _current_lrs(trainer) -> list[float]:
        lrs: list[float] = []
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                lrs.append(float(group["lr"]))
        return lrs

    @staticmethod
    def _format_value(value) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

"""Rich-powered Lightning progress bar with per-epoch telemetry panels."""
from __future__ import annotations

from time import perf_counter

from lightning.pytorch.callbacks import RichProgressBar
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class LiveTrainingMonitor(RichProgressBar):
    """Extends ``RichProgressBar`` with a fit-start banner and per-epoch summary tables.

    Drop-in replacement for the default progress bar. Adds:
      - Fit-start banner showing experiment name, model class, max epochs, optimizer(s).
      - Per-epoch summary table with train/val/test losses, headline metrics, LR(s),
        and elapsed seconds.

    Args:
        summary_metrics: Metric keys to surface in the epoch summary. Keys are
            matched against ``trainer.callback_metrics``; missing keys are
            rendered as ``—``.
    """

    DEFAULT_SUMMARY_METRICS: tuple[str, ...] = (
        "train_loss",
        "val_loss",
        "val_iou",
        "val_f1",
    )

    def __init__(
        self,
        summary_metrics: tuple[str, ...] = DEFAULT_SUMMARY_METRICS,
        refresh_rate: int = 1,
        leave: bool = False,
    ) -> None:
        super().__init__(refresh_rate=refresh_rate, leave=leave)
        self.summary_metrics = tuple(summary_metrics)
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

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if trainer.sanity_checking:
            return
        self._console.print(self._epoch_summary(trainer))

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

    def _epoch_summary(self, trainer) -> Panel:
        metrics = trainer.callback_metrics
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("metric", style="bold cyan")
        table.add_column("value", justify="right")

        for key in self.summary_metrics:
            value = metrics.get(key)
            table.add_row(key, self._format_value(value))

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

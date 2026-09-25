from __future__ import annotations

from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)
from torchmetrics.wrappers import ClasswiseWrapper


class SemanticSegmentationMetrics(MetricCollection):
    """Collect segmentation metrics selected by name and aggregation mode.

    Args:
        num_classes: Number of output classes.
        ignore_index: Target value excluded from metric computation.
        labels: Class names in class-index order, or a mapping whose values
            supply that order. None uses numeric labels for per-class results.
        metrics: Names with optional dot-separated aggregation modes.
            None or an empty list selects the default metrics.

    Raises:
        ValueError: A metric, aggregation mode, or their combination is invalid.

    Examples:
        >>> metrics = SemanticSegmentationMetrics(
        ...     3, ignore_index=255, metrics=["f1.macro.per_class", "mcc"]
        ... )
        >>> metrics.update(predictions, targets)
        >>> scores = metrics.compute()
    """

    metric_map: dict[str, type[Metric]] = {
        "accuracy": MulticlassAccuracy,
        "f1": MulticlassF1Score,
        "iou": MulticlassJaccardIndex,
        "precision": MulticlassPrecision,
        "recall": MulticlassRecall,
        "mcc": MulticlassMatthewsCorrCoef,
        "kappa": MulticlassCohenKappa,
        "auroc": MulticlassAUROC,
    }

    modes: dict[str, list[str]] = {
        "macro": ["accuracy", "f1", "iou", "precision", "recall", "auroc"],
        "micro": ["accuracy", "f1", "iou", "precision", "recall", "auroc"],
        "per_class": ["f1", "iou", "precision", "recall", "auroc"],
    }
    default_metrics: list[str] = [
        "accuracy.macro",
        "f1.macro",
        "iou.macro",
        "precision.macro",
        "recall.macro",
        "mcc",
        "kappa",
    ]

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        labels: list[str] | dict[str, str] | None = None,
        metrics: list[str] | None = None,
    ) -> None:

        metrics = metrics or self.default_metrics

        if isinstance(labels, dict):
            labels = list(labels.values())

        collection: dict[str, Metric | MetricCollection] = {}

        for entry in metrics:
            parts = entry.split(".")
            name = parts[0]
            requested_modes = list(dict.fromkeys(parts[1:]))

            if name not in self.metric_map:
                raise ValueError(
                    f"Unknown metric {name!r}; must be one of {list(self.metric_map)}"
                )

            metric_cls = self.metric_map[name]
            if not requested_modes:
                collection[name] = metric_cls(
                    num_classes=num_classes, ignore_index=ignore_index
                )
                continue

            for mode in requested_modes:
                if mode not in self.modes:
                    raise ValueError(
                        f"Unknown mode {mode!r}; must be one of {list(self.modes)}"
                    )

                eligible = self.modes[mode]

                if name not in eligible:
                    raise ValueError(
                        f"Metric {name!r} does not support mode {mode!r}; must be one of {eligible}"
                    )

                metric = metric_cls(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average=None if mode == "per_class" else mode,
                )
                if mode == "per_class":
                    metric = ClasswiseWrapper(metric, labels=labels, prefix=f"{name}_")
                collection[f"{name}_{mode}"] = metric

        super().__init__(collection)

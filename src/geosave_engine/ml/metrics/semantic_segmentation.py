from __future__ import annotations

import inspect

from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score,
    MulticlassJaccardIndex, MulticlassMatthewsCorrCoef,
    MulticlassPrecision, MulticlassRecall, MulticlassAUROC
)
from torchmetrics.wrappers import ClasswiseWrapper

from typing import Any, Callable

Config = dict[str, Any]  # for metric init kwargs

def filter_kwargs_for_class(cls: Callable, kwargs: dict) -> Config:
    """Filters a dictionary to only include keys that match a class constructor."""
    signature = inspect.signature(cls.__init__)
    valid_keys = set(signature.parameters.keys()) - {"self", "args", "kwargs"}
    
    return {k: v for k, v in kwargs.items() if k in valid_keys}
    

def macro_fn(metric_cls: Metric, config: Config) -> Metric:
    kwargs = filter_kwargs_for_class(metric_cls, config)
    return metric_cls(average='macro', **kwargs)

def micro_fn(metric_cls: Metric, config: Config) -> Metric:
    kwargs = filter_kwargs_for_class(metric_cls, config)
    return metric_cls(average='micro', **kwargs)

def per_class_fn(metric_cls: Metric, config: Config) -> Metric:
    kwargs = filter_kwargs_for_class(metric_cls, config)
    return ClasswiseWrapper(
        metric_cls(average=None, **kwargs), 
        labels=config.get('labels'),
        prefix=config.get('prefix')
    )
    

class SemanticSegmentationMetrics(MetricCollection):
    """Build MetricCollection from dot-notation metric names.

    Dot-notation: ``"<name>"`` or ``"<name>.<mode>[.<mode>...]"``.
      ``"f1.macro"``            — f1 macro
      ``"f1.macro.per_class"``  — f1 macro + per-class
      ``"mcc"``                 — no mode; added directly with common kwargs only
    """
    metric_map: dict[str, type] = {
        "accuracy": MulticlassAccuracy,
        "f1": MulticlassF1Score,
        "iou": MulticlassJaccardIndex,
        "precision": MulticlassPrecision,
        "recall": MulticlassRecall,
        "mcc": MulticlassMatthewsCorrCoef,
        "kappa": MulticlassCohenKappa,
        "auroc": MulticlassAUROC,
    }

    macro_metrics: tuple[list[str], Callable] = (
        ["accuracy", "f1", "iou", "precision", "recall", "auroc"],
        macro_fn
    )
    micro_metrics: tuple[list[str], Callable] = (
        ["accuracy", "f1", "iou", "precision", "recall", "auroc"],
        micro_fn
    )
    per_class_metrics: tuple[list[str], Callable] = (
        ["f1", "iou", "precision", "recall", "auroc"],
        per_class_fn
    )
    modes: dict[str, tuple[list[str], Callable]] = {
        "macro": macro_metrics,
        "micro": micro_metrics,
        "per_class": per_class_metrics
    }
    default_metrics: list[str] = [
        "accuracy.macro", 
        "f1.macro", 
        "iou.macro", 
        "precision.macro", 
        "recall.macro", 
        "mcc", "kappa"
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
        
        common = {"num_classes": num_classes, "ignore_index": ignore_index, "labels": labels}
        collection = {}

        for entry in metrics:
            parts = entry.split(".")
            name, modes = parts[0], set(parts[1:])

            if name not in self.metric_map:
                raise ValueError(f"Unknown metric {name!r}; must be one of {list(self.metric_map)}")

            if not modes:
                metric = self.metric_map[name]
                collection[name] = metric(**filter_kwargs_for_class(metric, common))
                continue

            for mode in modes:
                if mode not in self.modes:
                    raise ValueError(f"Unknown mode {mode!r}; must be one of {list(self.modes)}")
                
                eligible, build_fn = self.modes[mode]
                
                if name not in eligible:
                    raise ValueError(f"Metric {name!r} does not support mode {mode!r}; must be one of {eligible}")
                
                common.update({"prefix": f"{name}_"})
                metric = self.metric_map[name]
                collection[f"{name}_{mode}"] = build_fn(metric, common)

        super().__init__(collection)
    
    def compute(self) -> dict[str, Any]:
        """Compute metrics and flatten nested metric dicts."""
        result = super().compute()
        flat: dict[str, Any] = {}

        for name, value in result.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[name] = value

        return flat


if __name__ == "__main__":
    # Example usage
    import torch

    metrics = SemanticSegmentationMetrics(
        num_classes=3, 
        ignore_index=255, 
        labels=["water", "trees", "urban"], 
        # metrics=["f1.macro", "iou.per_class"]
    )
    tensor = torch.randn(4, 3, 256, 256)  # Example predictions (logits)
    target = torch.randint(low=0, high=3, size=(4, 256, 256))  # Example ground truth
    metrics.update(tensor, target)
    print(metrics.compute())
from __future__ import annotations

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score,
    MulticlassJaccardIndex, MulticlassMatthewsCorrCoef,
    MulticlassPrecision, MulticlassRecall,
)
from torchmetrics.wrappers import ClasswiseWrapper

# {name: (Class, default_modes)}. None in modes = scalar (no average param).
_METRIC_MAP: dict[str, tuple] = {
    "accuracy":  (MulticlassAccuracy,         {"macro", "per_class"}),
    "f1":        (MulticlassF1Score,          {"macro", "per_class"}),
    "iou":       (MulticlassJaccardIndex,     {"macro", "per_class"}),
    "precision": (MulticlassPrecision,        {"macro"}),
    "recall":    (MulticlassRecall,           {"macro"}),
    "mcc":       (MulticlassMatthewsCorrCoef, {None}),
    "kappa":     (MulticlassCohenKappa,       {None}),
}

_SCALAR_METRICS = {"mcc", "kappa"}


class SemanticSegmentationMetrics(MetricCollection):
    """Semantic segmentation metric suite with per-class and aggregate variants.

    Dot-notation overrides via ``metrics``:
      ``"f1.macro"``               — f1 macro only
      ``"f1.macro.per_class"``     — f1 macro + per-class
      ``"accuracy.exclude"``       — exclude accuracy
      ``"precision.per_class"``    — override precision to per-class only

    MCC and Kappa are always scalar; mode tokens are ignored for them.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        labels: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        config = {name: set(modes) for name, (_, modes) in _METRIC_MAP.items()}

        if metrics: # apply overrides
            for entry in metrics:
                parts = entry.split(".")
                name, modes = parts[0], set(parts[1:])

                if name not in config:
                    raise ValueError(f"Unknown metric {name!r}. Valid: {sorted(config)}")

                if "exclude" in modes:
                    config.pop(name)
                elif modes and name not in _SCALAR_METRICS:
                    config[name] = modes

        common = {"num_classes": num_classes, "ignore_index": ignore_index}
        collection = {}

        for name, modes in config.items():
            cls, _ = _METRIC_MAP[name]
            for mode in modes:
                if mode is None:  # scalar (mcc/kappa)
                    collection[name] = cls(**common)
                elif mode == "per_class":
                    collection[name] = ClasswiseWrapper(cls(average="none", **common), labels=labels)
                else:  # macro/micro/weighted
                    collection[f"{name}_{mode}"] = cls(average=mode, **common)

        super().__init__(collection)

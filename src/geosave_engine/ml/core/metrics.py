from __future__ import annotations

from typing import Sequence

from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.wrappers import ClasswiseWrapper


def get_segmentation_metrics(
    num_classes: int,
    class_names: Sequence[str],
    ignore_index: int | None,
) -> MetricCollection:
    """Standard segmentation metric collection: accuracy, IoU, F1, per-class IoU."""
    return MetricCollection(
        {
            "accuracy": Accuracy(
                task="multiclass", num_classes=num_classes, ignore_index=ignore_index
            ),
            "iou": JaccardIndex(
                task="multiclass", num_classes=num_classes, ignore_index=ignore_index
            ),
            "f1": MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index),
            "per_class_iou": ClasswiseWrapper(
                JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average=None,
                ),
                labels=list(class_names),
            ),
        }
    )

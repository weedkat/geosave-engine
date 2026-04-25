from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.wrappers import ClasswiseWrapper


@dataclass
class SegmentationMetrics:
    """Pair of segmentation metric collections.

    ``scalar`` aggregates aggregate metrics (accuracy, iou, f1) and is suitable
    for ``self.log_dict(self.metrics, on_epoch=True)`` auto-logging.
    ``per_class`` is a :class:`ClasswiseWrapper` whose ``compute`` returns a
    dict — Lightning's auto-logger rejects dicts, so the LightningModule
    manually flattens it inside ``on_*_epoch_end``.
    """

    scalar: MetricCollection
    per_class: ClasswiseWrapper

    def update(self, *args, **kwargs) -> None:
        self.scalar.update(*args, **kwargs)
        self.per_class.update(*args, **kwargs)

    def reset(self) -> None:
        self.scalar.reset()
        self.per_class.reset()

    def clone(self, prefix: str | None = None) -> SegmentationMetrics:
        return SegmentationMetrics(
            scalar=self.scalar.clone(prefix=prefix),
            per_class=self.per_class.clone(prefix=prefix),  # type: ignore[arg-type]
        )


def get_segmentation_metrics(
    num_classes: int,
    class_names: Sequence[str],
    ignore_index: int | None,
) -> SegmentationMetrics:
    """Standard segmentation metrics: accuracy, IoU, F1, per-class IoU."""
    scalar = MetricCollection(
        {
            "accuracy": Accuracy(
                task="multiclass", num_classes=num_classes, ignore_index=ignore_index
            ),
            "iou": JaccardIndex(
                task="multiclass", num_classes=num_classes, ignore_index=ignore_index
            ),
            "f1": MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index),
        }
    )
    per_class = ClasswiseWrapper(
        JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None,
        ),
        labels=list(class_names),
    )
    return SegmentationMetrics(scalar=scalar, per_class=per_class)

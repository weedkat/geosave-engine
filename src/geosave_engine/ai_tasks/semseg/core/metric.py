from torchmetrics import MetricCollection, Accuracy, JaccardIndex
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.wrappers import ClasswiseWrapper

def get_metrics(num_classes, class_names, ignore_index):
  
    metrics = {
        'accuracy': Accuracy(task='multiclass', num_classes=num_classes, ignore_index=ignore_index),
        'iou': JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=ignore_index),
        'f1': MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index),
        'per_class_iou': ClasswiseWrapper(
            JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=ignore_index, average=None),
            labels=class_names # This maps index 0 to "Forest", etc.
        )
    }

    return MetricCollection(metrics)
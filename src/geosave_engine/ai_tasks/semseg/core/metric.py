from torchmetrics import MetricCollection, Accuracy, JaccardIndex, Dice
from torchmetrics.wrappers import ClasswiseWrapper

def get_metrics(num_classes, class_names, ignore_index):
    # Base arguments for all metrics
    base_kwargs = {
        'task': 'multiclass',
        'num_classes': num_classes,
        'ignore_index': ignore_index
    }

    metrics = {
        'accuracy': Accuracy(**base_kwargs),
        'iou': JaccardIndex(**base_kwargs),
        'dice': Dice(**base_kwargs),
        'per_class_iou': ClasswiseWrapper(
            JaccardIndex(**base_kwargs, average=None),
            labels=class_names # This maps index 0 to "Forest", etc.
        )
    }

    return MetricCollection(metrics)
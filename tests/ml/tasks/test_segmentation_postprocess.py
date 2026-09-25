import torch

from geosave_engine.ml.tasks.semantic_segmentation import (
    apply_thresholds,
    softmax_argmax,
)


def test_task_postprocessing_applies_thresholds_and_nodata_without_mutating_logits():
    logits = torch.tensor([[[[3.0, 0.0, 0.0]], [[0.0, 0.0, 3.0]]]])
    original = logits.clone()
    labels, confidence = softmax_argmax(logits)
    assert labels.tolist() == [[[0, 0, 1]]]
    result, scores = apply_thresholds(
        logits,
        torch.tensor([0.8, 0.8]),
        255,
        mask=torch.tensor([[[False, False, True]]]),
    )
    assert result.tolist() == [[[0, 255, 255]]]
    torch.testing.assert_close(scores, confidence)
    torch.testing.assert_close(logits, original)

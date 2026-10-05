import torch

from geosave_engine.ml.segmentation.transforms import (
    apply_thresholds,
    softmax_argmax,
)


def test_thresholds_and_nodata_preserve_logits_and_return_storage_dtypes():
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
    assert result.dtype == torch.uint8
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, confidence)
    torch.testing.assert_close(logits, original)

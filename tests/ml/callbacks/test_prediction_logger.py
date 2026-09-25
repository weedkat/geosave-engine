from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from geosave_engine.ml.callbacks import DensePredictionLogger


@pytest.mark.parametrize("interval", [0, -1])
def test_interval_must_be_positive(interval: int) -> None:
    with pytest.raises(ValueError, match="log_image_every_n_epochs must be positive"):
        DensePredictionLogger({0: "#000000"}, log_image_every_n_epochs=interval)


@pytest.mark.parametrize("reason", ["sanity", "rank", "logger", "batch", "epoch"])
def test_skipped_batches_do_not_render_or_require_outputs(reason: str) -> None:
    callback = DensePredictionLogger({0: "#000000"})
    trainer = Mock(spec=Trainer)
    trainer.sanity_checking = reason == "sanity"
    trainer.is_global_zero = reason != "rank"
    trainer.current_epoch = 1 if reason == "epoch" else 0
    trainer.loggers = [] if reason == "logger" else [Mock(spec=TensorBoardLogger)]

    callback.on_validation_batch_end(
        trainer, LightningModule(), None, None, 1 if reason == "batch" else 0
    )
    for logger in trainer.loggers:
        logger.experiment.add_image.assert_not_called()


def test_loader_images_have_distinct_names_and_argmax_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = DensePredictionLogger({0: "#000000", 1: "#ffffff"})
    trainer = Mock(spec=Trainer)
    trainer.sanity_checking = False
    trainer.is_global_zero = True
    trainer.current_epoch = 2
    logger = Mock(spec=TensorBoardLogger)
    trainer.loggers = [logger]
    label = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.tensor([[[[3.0, 0.0], [1.0, 2.0]], [[0.0, 4.0], [2.0, 0.0]]]])
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def render(actual_label: torch.Tensor, actual_preds: torch.Tensor) -> np.ndarray:
        torch.testing.assert_close(actual_label, label[0])
        torch.testing.assert_close(actual_preds, label[0])
        return image

    monkeypatch.setattr(callback, "_render", render)
    for loader in (0, 1):
        callback.on_validation_batch_end(
            trainer,
            LightningModule(),
            {"logits": logits, "label": label},
            None,
            0,
            loader,
        )
    assert [call.args[0] for call in logger.experiment.add_image.call_args_list] == [
        "val/dataloader_0/prediction",
        "val/dataloader_1/prediction",
    ]
    assert logger.experiment.add_image.call_args.args[1] is image
    assert logger.experiment.add_image.call_args.args[2] == 2

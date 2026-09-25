"""Ordinary PyTorch Lightning task for a generated custom workspace."""

from __future__ import annotations

from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW


class CustomTask(LightningModule):
    """Classify four input features with a small feed-forward network.

    Replace ``network`` and ``_loss`` with the application's model and loss;
    ``training_step`` and ``validation_step`` keep the standard Lightning flow.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 8,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *, features: Tensor) -> Tensor:
        """Return logits for a batch of feature vectors."""
        return self.network(features)

    def _loss(self, batch: tuple[dict[str, Tensor], Tensor]) -> Tensor:
        """Calculate the classification loss for one batch."""
        inputs, target = batch
        logits = self(features=inputs["features"])
        return self.criterion(logits, target)

    def training_step(
        self, batch: tuple[dict[str, Tensor], Tensor], batch_idx: int
    ) -> Tensor:
        """Calculate and log the training loss."""
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[dict[str, Tensor], Tensor], batch_idx: int
    ) -> Tensor:
        """Calculate and log the validation loss."""
        loss = self._loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> AdamW:
        """Return the task-owned AdamW optimizer."""
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

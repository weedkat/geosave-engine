import pytest
from pathlib import Path

import torch
from geosave_engine.ai_tasks.semseg.supervised.loader import DataModule
from geosave_engine.ai_tasks.semseg.supervised.main import SemSegModel
import lightning as L


dataset_dir = Path("dataset/isprs_postdam").resolve()
metadata_path = Path("config/isprs_postdam/metadata.yaml").resolve()

dm = DataModule(data_dir=dataset_dir, metadata=metadata_path)

model = SemSegModel(
    arch="dpt",
    optim="AdamW",
    loss="CELoss",
    metadata=metadata_path,
)

trainer = L.Trainer(
    accelerator="cpu",
    limit_train_batches=5,  # Only run 5 batches per epoch
    limit_val_batches=2,    # Only run 2 validation batches
    max_epochs=1            # Just do one lap
)

trainer.fit(model, datamodule=dm)


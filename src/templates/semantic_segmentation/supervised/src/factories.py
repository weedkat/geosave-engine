from __future__ import annotations

from geosave_engine.losses.cross_entropy_loss import CrossEntropyLoss
from geosave_engine.optimizers.adamw import AdamW

LOSS_FACTORY = [CrossEntropyLoss]
OPTIM_FACTORY = [AdamW]

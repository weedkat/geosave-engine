from __future__ import annotations

from lightning.pytorch.callbacks import Callback


class PredictionWriter(Callback):
    """Writes prediction artifacts from `predict` runs (implementation pending)."""

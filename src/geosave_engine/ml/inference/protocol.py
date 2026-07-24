from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Predictable(Protocol):
    """Structural contract a deployable task must satisfy.

    Anything with this shape works with predict-time serving code — a
    direct method call, no ``Trainer``, no ``DataModule`` — whether or not
    it happens to be a ``SemanticSegmentationTask``. Checked at MLflow
    registration time (see ``cli.commands.upload``) so an incompatible
    custom task fails loud there, not later at serve time.
    """

    def predict(
        self,
        image: torch.Tensor,
        context: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class GeosaveModel(Protocol):
    """Structural contract for a model exposing its inference stages separately.

    Complements ``Predictable`` (the minimal "can this thing produce a
    finished prediction" gate) — this is for callers that want the pieces
    on their own: raw preprocessing, a raw one-tile forward pass, or raw
    postprocessing, rather than one composed ``predict()`` call. A task
    implementing ``predict()`` internally without exposing these three
    separately still satisfies ``Predictable`` but not this.
    """

    def preprocess(self, image: torch.Tensor) -> torch.Tensor: ...

    def forward(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...

    def postprocess(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

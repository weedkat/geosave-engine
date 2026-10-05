from __future__ import annotations

import torch
from torch import nn

from geosave_engine.model.chain import Published, chain_step


class Encoder(nn.Module):
    channels: Published[int]

    def __init__(self, channels: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.pretrained = pretrained
        self.scale = nn.Parameter(torch.tensor(2.0 if pretrained else 0.0))

    @chain_step(outputs=("features",))
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.scale


class Head(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.channels

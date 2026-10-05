"""Smoke native terminal outputs using the implemented library chain."""

import json

import torch
from torch import nn

from geosave_engine.model.chain import ModelChain, chain_step


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    @chain_step(outputs=("features",))
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.scale


class Dense(nn.Module):
    @chain_step(head=True)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1, keepdim=True)


class Objects(nn.Module):
    @chain_step(head=True)
    def forward(self, features: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": value.new_empty((0, 4)),
                "scores": value.new_empty((0,)),
                "labels": value.new_empty((0,), dtype=torch.long),
            }
            for value in features
        ]


class Bad(nn.Module):
    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return image


image = torch.rand(2, 3, 8, 8)
single_dense = ModelChain(encoder=Encoder(), cover=Dense())
assert single_dense(image=image).shape == (2, 1, 8, 8)
single_objects = ModelChain(encoder=Encoder(), objects=Objects())
assert len(single_objects(image=image)) == 2
mixed = ModelChain(encoder=Encoder(), cover=Dense(), objects=Objects())
values = mixed(image=image)
assert set(values) == {"cover", "objects"}
assert all(record["boxes"].shape == (0, 4) for record in values["objects"])
values["cover"].sum().backward()
assert mixed.encoder.scale.grad is not None
assert mixed.encoder.scale.grad.abs() > 0
assert set(ModelChain(encoder=Encoder())(image=image)) == {"features"}
two_dense = ModelChain(encoder=Encoder(), cover=Dense(), amount=Dense())
assert set(two_dense(image=image)) == {"cover", "amount"}
try:
    ModelChain(bad=Bad())(image=image)
except TypeError as error:
    assert "expected list" in str(error)
else:
    raise AssertionError("Bad outer return type was accepted")

print(
    json.dumps(
        {
            "single_tensor": True,
            "single_native_detections": True,
            "mixed_native_outputs": True,
            "two_tensors": True,
            "encoder_only": True,
            "empty_detections": True,
            "gradient": True,
            "outer_type_validation": True,
            "implementation": ModelChain.__module__,
        },
        indent=2,
    )
)

import torch
from geosave_engine.core.factory import BaseLossFactory


class CrossEntropyLoss(BaseLossFactory):
    """Cross-entropy loss wrapper based on PyTorch's nn.CrossEntropyLoss."""

    task = {
        "semantic segmentation": ["supervised"],
        "pixelwise regression": []
        }
    doc_links = ["https://docs.geosave.dev/losses/cross_entropy"]
    loss = torch.nn.CrossEntropyLoss

    @classmethod
    def build(cls, *args, **kwargs):
        return cls.loss(*args, **kwargs)
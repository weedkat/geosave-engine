from __future__ import annotations

import re
import torch.nn as nn


def split_encoder_decoder(model: nn.Module) -> tuple[list, list]:
    """Split parameters into encoder and decoder groups by name."""
    encoder, decoder = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name or "backbone" in name:
            encoder.append(param)
        else:
            decoder.append(param)
    return encoder, decoder


def split_no_wd(model: nn.Module) -> tuple[list, list]:
    """Split parameters into weight-decayed and non-weight-decayed groups.

    Bias and 1-D params (norms) are excluded from weight decay.
    """
    wd, no_wd = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_wd.append(param)
        else:
            wd.append(param)
    return wd, no_wd


def freeze_backbone(model: nn.Module) -> list:
    """Freeze encoder/backbone params and return remaining trainable params."""
    for name, param in model.named_parameters():
        if "encoder" in name or "backbone" in name:
            param.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


def layerwise_param_groups(
    model: nn.Module,
    base_lr: float,
    decoder_lr: float,
    decay_rate: float = 0.75,
) -> list[dict]:
    """Build per-layer param groups with exponentially decaying LR.

    Assigns each backbone block a LR of ``base_lr * decay_rate^(num_blocks - i)``.
    Decoder and non-backbone params use ``decoder_lr``.
    Designed for ViT-style backbones (blocks.0, blocks.1, ...).
    """
    block_indices: list[int] = []
    for name, _ in model.named_parameters():
        match = re.search(r"(?:blocks|layers)\.(\d+)\.", name)
        if match:
            block_indices.append(int(match.group(1)))

    num_blocks = max(block_indices) + 1 if block_indices else 0

    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match = re.search(r"(?:blocks|layers)\.(\d+)\.", name)
        if match:
            idx = int(match.group(1))
            lr = base_lr * (decay_rate ** (num_blocks - 1 - idx))
            key = f"block_{idx}"
        else:
            lr = decoder_lr
            key = "other"

        if key not in groups:
            groups[key] = {"params": [], "lr": lr}
        groups[key]["params"].append(param)

    return list(groups.values())

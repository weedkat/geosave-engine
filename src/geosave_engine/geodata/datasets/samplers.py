from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


def stack_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack list of sample dicts into one batched dict for DataLoader.

    Tensor values are stacked on dim 0; dict values collated recursively; non-tensor values gathered as list.

    Args:
        samples: Iterable of sample dicts from StackDataset.__getitem__.

    Returns:
        {
            "<key>": torch.Tensor,  # stacked if value was Tensor
            "<key>": dict,          # recursively collated if value was dict
            "<key>": list,          # gathered as list otherwise
        }.
    """
    sample_list = list(samples)
    sample_keys = set(sample_list[0].keys())
    out: dict[str, Any] = {}
    for key in sample_keys:
        values = [s[key] for s in sample_list]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        elif isinstance(values[0], dict):
            out[key] = stack_samples(values)
        else:
            out[key] = values
    return out

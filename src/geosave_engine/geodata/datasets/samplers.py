from __future__ import annotations

from typing import Any, Iterable, Mapping

from torch.utils.data import default_collate


def stack_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack list of sample dicts into one batched dict for DataLoader.

    Args:
        samples: Iterable of dicts from `GeoTileStack.to_tensor()`.

    Returns:
        {
            "layers": {
                "<layer>": torch.Tensor,  # (batch, band, y, x)
            },
            "anchor": [GeoAnchor],  # one per sample, in batch order
            "model_context": {
                "<key>": torch.Tensor | list | str | None,
            },
        }

    Raises:
        ValueError: Samples have different keys or inconsistent null values.
    """
    sample_list = list(samples)
    if not sample_list:
        return {}

    keys = tuple(sample_list[0])
    if any(set(sample) != set(keys) for sample in sample_list[1:]):
        raise ValueError("all samples must have the same keys")

    out: dict[str, Any] = {}
    for key in keys:
        # a GeoAnchor isn't a tensor/number/mapping, so it rides as a plain per-sample list
        if key == "anchor":
            out[key] = [sample[key] for sample in sample_list]
        else:
            out[key] = _collate_values([sample[key] for sample in sample_list])

    return out


def _collate_values(values: list[Any]) -> Any:
    """Collate matching nested values while preserving null fields."""
    first = values[0]
    if first is None:
        if any(value is not None for value in values[1:]):
            raise ValueError("sample field mixes null and non-null values")
        return None
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(not isinstance(value, Mapping) or set(value) != set(keys) for value in values[1:]):
            raise ValueError("sample mappings must have the same keys")
        return {key: _collate_values([value[key] for value in values]) for key in keys}
    return default_collate(values)

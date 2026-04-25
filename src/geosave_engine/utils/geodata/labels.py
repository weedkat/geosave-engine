"""Label raster utilities: pixel-value remapping."""
from __future__ import annotations

import numpy as np


def remap_label_array(
    arr: np.ndarray,
    mapping: dict[int, int],
    default: int = 255,
) -> np.ndarray:
    """Vectorised LUT remap.

    Pixels whose value is not present in ``mapping`` are replaced by ``default``.
    Output dtype is uint8; raises ``ValueError`` if any value exceeds 255.
    """
    src_max = max(int(arr.max()), max(mapping.keys(), default=0))
    if src_max > 255 or default > 255 or any(v > 255 for v in mapping.values()):
        raise ValueError("label remap targets must fit in uint8 (≤255)")

    lut = np.full(src_max + 1, default, dtype=np.uint8)
    for src_val, dst_val in mapping.items():
        lut[src_val] = dst_val
    return lut[arr.astype(np.int64)]


def build_remap_from_class_meta(class_meta: list[dict]) -> dict[int, int]:
    """Build ``{source_class_id: class_id}`` from a metadata table."""
    return {int(c["source_class_id"]): int(c["class_id"]) for c in class_meta}

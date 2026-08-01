from __future__ import annotations

from pathlib import Path
from typing import Any

from geosave_engine.geodata.datasets.base_dataset import BaseDataset


class CocoDataset(BaseDataset):
    """SKELETON — spec not settled, do not implement against this yet.

    COCO-format annotation dataset — one json for the whole split
    (`images`/`annotations` lists), keyed by `file_name` stem. Open:

      - `bbox` stays COCO's own `[x, y, w, h]` (pixel, top-left origin), or
        converted to something else?
      - Keyed on `file_name` stem — confirmed right, or should it use
        COCO's own `image_id` and let the raster side adapt instead?
      - Segmentation/keypoints/other COCO annotation types in scope at
        all, or bbox-only for now?

    Args:
        path: COCO-style annotations json.
    """

    def __init__(self, path: str | Path) -> None:
        raise NotImplementedError("spec not settled — see open questions in class docstring")

    def render(self, key: str) -> dict[str, Any]:
        """Render one sample.

        Args:
            key: Image stem — one of `self.keys`.

        Returns:
            TBD — see open questions in class docstring.
        """
        raise NotImplementedError("spec not settled — see open questions in class docstring")

    def to_row(self, key: str) -> dict[str, Any]:
        """Manifest row for `key`.

        Args:
            key: Image stem — one of `self.keys`.
        """
        raise NotImplementedError("spec not settled — see open questions in class docstring")

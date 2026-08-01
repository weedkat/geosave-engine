from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from geosave_engine.geodata.datasets.base_dataset import BaseDataset, extract_key, filter_by_split


class YoloDataset(BaseDataset):
    """YOLO-format annotation dataset — one `.txt` per image, one line per box.

    Native layout, read as-is: each line is `class_id cx cy w h`
    (normalized, center format), no pixel-space conversion and no class-name
    resolution here — both need context this class alone doesn't have
    (image size; a `classes.txt`/`data.yaml` mapping). `class_id` stays a
    raw int, `box` stays normalized. Keyed by the label file's stem —
    matches an image dataset's own sample stem when they share a name, so
    `IntersectionDataset`/`__and__` joins them directly.

    `root` should point at the labels folder itself, not a mixed
    images+labels+notes directory — `.txt` is generic enough that a stray
    `classes.txt` sitting elsewhere in the tree would otherwise get globbed
    as if it were a real label file.

    Args:
        root: Directory to `rglob` for `.txt` label files.
        class_id_name: Dict key `render()`/`to_row()` return class ids under.
        box_name: Dict key `render()`/`to_row()` return boxes under.
        key_pattern: Regex to extract the sample key from each label file's
            name. None strips `.txt` and uses the rest — see `extract_key`.
        split: Text file of label stems to keep, one per line. None keeps
            every `.txt` file found under `root`.

    Raises:
        ValueError: A label line doesn't split into exactly 5 fields
            (`class_id cx cy w h`) — usually means `root` picked up a
            non-label `.txt` file.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        class_id_name: str = "class_id",
        box_name: str = "box",
        key_pattern: str | None = None,
        split: str | Path | None = None,
    ) -> None:
        self.split = split
        self.root = Path(root)
        self.class_id_name = class_id_name
        self.box_name = box_name
        self.key_pattern = key_pattern

        label_files: dict[str, Path] = {
            extract_key(p.name, key_pattern): p for p in sorted(self.root.rglob("*.txt"))
        }
        label_files = filter_by_split(label_files, split)

        samples: dict[str, dict[str, list]] = {}
        for stem, label_file in label_files.items():
            class_ids: list[int] = []
            boxes: list[list[float]] = []
            for lineno, line in enumerate(label_file.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"{label_file}:{lineno}: expected 'class_id cx cy w h' (5 fields), got "
                        f"{len(fields)}: {line!r} — wrong file globbed as a YOLO label?"
                    )
                class_id, cx, cy, w, h = fields
                class_ids.append(int(class_id))
                boxes.append([float(cx), float(cy), float(w), float(h)])
            samples[stem] = {"class_id": class_ids, "box": boxes}

        self.samples = samples
        self.reindex(samples)

    def render(self, key: str) -> dict[str, Any]:
        """Render one sample.

        Args:
            key: Label file stem — one of `self.keys`.

        Returns:
            `{class_id_name: Tensor[N] int64, box_name: Tensor[N,4] float32}`
            — one entry per box, `[cx, cy, w, h]` normalized per row.
            `N=0`, correctly shaped (`torch.Size([0])`/`torch.Size([0, 4])`),
            for an image with no boxes.
        """
        sample = self.samples[key]
        return {
            self.class_id_name: torch.tensor(sample["class_id"], dtype=torch.int64),
            self.box_name: torch.tensor(sample["box"], dtype=torch.float32).reshape(-1, 4),
        }

    def to_row(self, key: str) -> dict[str, Any]:
        """Manifest row for `key` — the parsed boxes themselves.

        Args:
            key: Label file stem — one of `self.keys`.
        """
        sample = self.samples[key]
        return {self.class_id_name: sample["class_id"], self.box_name: sample["box"]}

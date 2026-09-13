"""Reading a cut's tiles as model input, one tile per sample.

Examples:
    Predicting two scenes and merging each back into a raster::

        tiles = Tiles([scene_a[["B04", "B08"]], scene_b[["B04", "B08"]]], (256, 256), overlap=32)
        loader = DataLoader(TileDataset(tiles), batch_size=8)
        merger = tiles.merger(window="hann")

        for batch in loader:
            predictions = model(batch["image"]).cpu().numpy()
            merger.add(dict(zip(batch["index"].tolist(), predictions)))

        predictions = merger.merge()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torch

    from geosave_engine.geodata.transform.tiling import Tiles


class TileDataset(Dataset[dict[str, Any]]):
    """Read a cut's tiles as model input, each under the number it answers.

    A sample states the tile number it was asked by, which is what routes its
    result back to the raster it came from. Tiles stay lazy until a sample is
    read, so a worker reads only the pixels its own tile covers.

    Args:
        tiles: Cut to read. Select and order the variables a model expects
            before cutting, so the selection happens once rather than per tile.
        dtype: Tensor dtype. None casts to `torch.float32`.

    Examples:
        >>> samples = TileDataset(Tiles([scene[["B04", "B08"]]], (256, 256)))
        >>> len(samples), sorted(samples[0])
        (9, ['image', 'index'])
    """

    def __init__(self, tiles: Tiles, *, dtype: torch.dtype | None = None) -> None:
        """Read nothing yet, holding the cut its samples come from."""
        self.tiles = tiles
        self.dtype = dtype

    def __len__(self) -> int:
        """Count the samples, one per tile of the cut.

        Returns:
            Tile count across every raster the cut holds.
        """
        return len(self.tiles)

    def __repr__(self) -> str:
        """Describe how many samples the cut being read holds."""
        return f"{type(self).__name__}({len(self)} samples)"

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read the tile this number names as model input.

        Args:
            index: Tile number in `range(len(self))`.

        Returns:
            {
                "image": tensor shaped `(*axes, band, y, x)`, or one such
                    per group where the raster is a stack,
                "index": the number this sample was asked by,
            }

        Raises:
            IndexError: `index` falls outside the cut.

        Examples:
            >>> samples[3]["image"].shape
            torch.Size([2, 256, 256])
        """
        return {
            "image": self.tiles[index].gs.to_tensor(dtype=self.dtype),
            "index": index,
        }

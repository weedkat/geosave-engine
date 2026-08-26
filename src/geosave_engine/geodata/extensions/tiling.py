"""TilingInfo: where one tile sits in its group's grid. See TilingInfo for details."""
from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2b
from typing import TYPE_CHECKING, ClassVar, Literal

import orjson

from geosave_engine.geodata.extensions.base import GeoExtension

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox
    from tiler import Tiler

    from geosave_engine.geodata.utils.datetime import DateRange

# How GeoRaster.tiles() fills a trailing window's overhang — tiler.Tiler's own modes.
TilerMode = Literal["reflect", "edge", "constant"]


class TilingInfo(GeoExtension):
    """Which GeoRaster.tiles() call produced one tile, and where in its own grid.

    Never propagates from a tile back up to a composed array — see
    `combine`. All fields required together; a partial stamp can't place a
    tile back.

    Args:
        group_id: Shared ID for every tile of one cut. `from_grid` derives
            one from the cut itself, so the same surface cut the same way
            in another process gets the same group.
        tile_id: This tile's own position within that call, row-major.
        data_shape: The whole raster's own (height, width) at tiling time.
        tile_shape: One tile's own (height, width), before edge handling.
        overlap: Pixels consecutive tiles share — px, a [0,1) fraction, or (row, col).
        mode: Trailing-edge behavior used to build the group.
    """

    NAMESPACE: ClassVar[str] = "tiling"

    group_id: str
    tile_id: int
    data_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    overlap: int | float | tuple[int, int]
    mode: TilerMode

    @classmethod
    def from_grid(
        cls,
        geobox: GeoBox,
        tile_shape: tuple[int, int],
        overlap: int | float | tuple[int, int],
        mode: TilerMode,
        *,
        time: DateRange | None = None,
        name: str | None = None,
        group_id: str | None = None,
    ) -> TilingInfo:
        """Stamp one whole cut, `tile_id` at 0 for the caller to advance per tile.

        Args:
            geobox: Grid being cut, whole.
            tile_shape: One tile's own (height, width), before edge handling.
            overlap: Pixels consecutive tiles share.
            mode: Trailing-edge behavior.
            time: Span the cut covers, so two cuts of one surface over
                different spans land in different groups. None for a
                timeless cut.
            name: Extra text folded into the derived ID, separating two
                otherwise identical cuts. None derives from the cut alone.
            group_id: Use this ID as-is instead of deriving one.

        Returns:
            Stamp for the cut, `tile_id` 0.

        Examples:
            >>> cut = TilingInfo.from_grid(raster.data.odc.geobox, (512, 512), 0, "reflect")
            >>> tile_stamp = cut.model_copy(update={"tile_id": 7})
        """
        return cls(
            group_id=group_id or _cut_digest(geobox, tile_shape, overlap, mode, time, name),
            tile_id=0,
            data_shape=geobox.shape.yx,
            tile_shape=tile_shape,
            overlap=overlap,
            mode=mode,
        )

    def tiler(self, channels: int | None = None) -> Tiler:
        """Rebuild the tiler this group was cut with.

        Args:
            channels: Flattened non-spatial length (bands × timesteps) to
                hold out of the tiling, for laying whole arrays back down.
                None tiles the `(y, x)` grid alone, which is all that
                window origins need.

        Returns:
            Tiler over `data_shape`, cutting `tile_shape` windows. Its
            leading axis is the held-out channel axis when `channels` is
            given.
        """
        from tiler import Tiler

        if channels is None:
            return Tiler(
                data_shape=self.data_shape,
                tile_shape=self.tile_shape,
                overlap=self.overlap,
                mode=self.mode,
            )
        return Tiler(
            data_shape=(channels, *self.data_shape),
            tile_shape=(channels, *self.tile_shape),
            overlap=(0, *self.overlap) if isinstance(self.overlap, tuple) else self.overlap,
            channel_dimension=0,
            mode=self.mode,
        )

    @classmethod
    def combine(cls, values: Sequence[GeoExtension | None]) -> None:
        """Never propagate a tiling stamp onto a composed array.

        A stamp places one tile in its own group's grid; a composed array
        isn't a member of that grid anymore.

        Args:
            values: This namespace's value from each array being composed.

        Returns:
            None, always.
        """
        return None


def _cut_digest(
    geobox: GeoBox,
    tile_shape: tuple[int, int],
    overlap: int | float | tuple[int, int],
    mode: TilerMode,
    time: DateRange | None,
    name: str | None,
) -> str:
    """Stable ID for one cut, identical across processes and machines.

    Args:
        geobox: Grid being cut, whole.
        tile_shape: One tile's own (height, width).
        overlap: Pixels consecutive tiles share.
        mode: Trailing-edge behavior.
        time: Span the cut covers, or None.
        name: Caller-supplied separator, or None.

    Returns:
        16 lowercase hex characters.
    """
    # Python's own hash() is reseeded every process, so the digest is built from canonical bytes instead.
    spec = {
        "crs": str(geobox.crs),
        "affine": [float(value) for value in tuple(geobox.affine)[:6]],
        "shape": list(geobox.shape.yx),
        "tile_shape": list(tile_shape),
        "overlap": list(overlap) if isinstance(overlap, tuple) else overlap,
        "mode": mode,
        "time": None if time is None else [time[0].isoformat(), time[1].isoformat()],
        "name": name,
    }
    return blake2b(orjson.dumps(spec, option=orjson.OPT_SORT_KEYS), digest_size=8).hexdigest()

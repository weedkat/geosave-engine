"""GeoStitcher: lay prediction tiles back onto their surfaces, batch by batch. See GeoStitcher."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.extensions import GeoExtension
from geosave_engine.geodata.utils.spatial.geobox import geobox_matches
from geosave_engine.geodata.utils.array import same_nodata

from .raster import GeoRaster
from .tile import GeoTile
from .vector import GeoVector

if TYPE_CHECKING:
    from tiler import Merger, Tiler

    from geosave_engine.geodata.extensions import TilingInfo


@dataclass
class _Group:
    """One tiling group's in-flight merge state.

    Args:
        info: The group's shared tiling config.
        config: Tiling fields every tile in the group must share.
        tiler: Tiler rebuilt over the whole surface, channels held out.
        merger: Accumulator every added tile is laid into.
        geobox: Reconstructed whole-surface grid.
        lead: Non-spatial dim names, in the tiles' own order.
        lead_shape: Length of each lead dim.
        lead_coords: Coordinate values for every non-spatial dim.
        dtype: Pixel dtype every tile must share.
        nodata: Nodata declaration every tile must share.
        expected: How many tiles the surface was cut into.
        pixel_shape: One tile's own array shape, which all must share.
        seen: `tile_id`s laid in so far.
        reference: Lowest-`tile_id` tile seen — names the grid and header.
        vectors: `(tile_id, vector)` for every tile that carried one.
    """

    info: TilingInfo
    config: tuple[object, ...]
    tiler: Tiler
    merger: Merger
    geobox: GeoBox
    lead: tuple[str, ...]
    lead_shape: tuple[int, ...]
    lead_coords: dict[str, np.ndarray]
    dtype: np.dtype[Any]
    nodata: float | int | None
    expected: int
    pixel_shape: tuple[int, ...]
    seen: set[int] = field(default_factory=set)
    reference: GeoTile | None = None
    vectors: list[tuple[int, GeoVector]] = field(default_factory=list)


class GeoStitcher:
    """Lay prediction tiles back onto the surfaces they were cut from, without holding a run in memory.

    Tiles route by their own `tiling.group_id`, so batches may mix groups and
    arrive in any order. A tile merges on `add` and may be dropped straight
    after. Read surfaces out with `drain` as they complete, or `flush` at the end.

    Args:
        window: How pixels several tiles cover are weighed against each
            other — a `tiler.Merger` window name (`"hann"`,
            `"overlap-tile"`, ...) or a `(tile_height, tile_width)` array.
            None weighs every tile equally. Needs tiles cut with an overlap.

    Examples:
        >>> stitcher = GeoStitcher()
        >>> for prediction_tiles in loader:
        ...     stitcher.add(*prediction_tiles)
        ...     for raster in stitcher.drain():
        ...         raster.to_cog(out / f"{raster.stem}.tif")
        >>> for raster in stitcher.flush():
        ...     raster.to_cog(out / f"{raster.stem}.tif")
    """

    def __init__(
        self,
        *,
        window: str | np.ndarray | None = None,
    ) -> None:
        self._window = window
        self._groups: dict[str, _Group] = {}

    def __repr__(self) -> str:
        pending = ", ".join(f"{gid[:8]}: {len(g.seen)}/{g.expected}" for gid, g in self._groups.items())
        return f"{type(self).__name__}(window={self._window!r}, groups={{{pending}}})"

    def __len__(self) -> int:
        """How many groups are in flight."""
        return len(self._groups)

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Every in-flight `group_id`, in first-seen order."""
        return tuple(self._groups)

    def add(self, *tiles: GeoTile) -> None:
        """Lay tiles into their own groups' accumulators.

        Args:
            *tiles: Tiles to merge, any order, any mix of groups. Every
                one needs a `tiling` stamp — only a `GeoRaster.tiles()`
                tile has one, and a prediction keeps it when re-anchored
                with `tile.anchor.to_geotile(pred, bands=("class",), attrs=prediction_attrs)`.

        Raises:
            ValueError: a tile carries no `tiling`, disagrees with its
                group's tiling config or pixel shape, repeats a `tile_id`
                already added, or `window` was given for tiles cut
                without an overlap.
        """
        for tile in tiles:
            stamp = tile.tiling
            if stamp is None:
                raise ValueError(
                    f"tile at {tile.stem} carries no tiling stamp — only a GeoRaster.tiles() tile merges back"
                )

            group = self._groups.get(stamp.group_id)
            is_new = group is None
            if group is None:
                group = self._open_group(stamp, tile)
            self._validate(group, stamp, tile)

            values = tile.data.values.reshape(group.tiler.tile_shape)
            group.merger.add(stamp.tile_id, values)
            group.seen.add(stamp.tile_id)
            if is_new:
                self._groups[stamp.group_id] = group

            # the lowest tile_id names the grid and header, so the result never follows arrival order
            if group.reference is None or stamp.tile_id < _stamp_id(group.reference):
                group.reference = tile
            if tile.vector is not None:
                group.vectors.append((stamp.tile_id, tile.vector))

    def is_complete(self, group_id: str) -> bool:
        """Whether every tile the surface was cut into has been added.

        Args:
            group_id: Group to check.

        Returns:
            True when the group holds all its tiles. False for a group
            still missing one, and for a `group_id` nothing was added for.
        """
        group = self._groups.get(group_id)
        return group is not None and len(group.seen) == group.expected

    def missing(self, group_id: str) -> tuple[int, ...]:
        """Tile IDs a group still needs before it can merge.

        Turns a half-finished run into a resumable one: re-cut the surface,
        add whatever tiles were written, then compute only these.

        Args:
            group_id: Group to inspect.

        Returns:
            Missing `tile_id`s, ascending. Empty for a complete group and
            for a `group_id` nothing was added for.
        """
        group = self._groups.get(group_id)
        if group is None:
            return ()
        return tuple(sorted(set(range(group.expected)) - group.seen))

    def drain(self) -> Iterator[GeoRaster]:
        """Yield every group that holds all its tiles, releasing each one.

        Call between batches to write finished surfaces and free their
        accumulators while the rest of the run is still going.

        Yields:
            One GeoRaster per completed group, in first-seen order.
            Nothing when no group is complete yet.
        """
        for group_id in [gid for gid in self._groups if self.is_complete(gid)]:
            yield self._close(group_id)

    def flush(self, *, allow_partial: bool = False) -> Iterator[GeoRaster]:
        """Release every group still in flight, one surface each.

        Every group is checked before the first surface is merged, so a
        rejected group raises from the call itself, not part-way through
        iterating.

        Args:
            allow_partial: True merges a group missing tiles, leaving its
                declared nodata wherever no tile landed. False rejects one.

        Returns:
            Iterator over one GeoRaster per group, in first-seen order.

        Raises:
            ValueError: a group is missing tiles and `allow_partial` is
                False, or it is missing tiles and its tiles declare no
                nodata to fill the holes they leave.
        """
        for group_id, group in self._groups.items():
            if len(group.seen) == group.expected:
                continue
            missing = self.missing(group_id)
            listed = f"{list(missing[:10])}{'...' if len(missing) > 10 else ''}"
            if not allow_partial:
                raise ValueError(
                    f"group {group_id!r} is missing {len(missing)} of {group.expected} tiles "
                    f"(tile_ids {listed}) — merging it would leave "
                    "nodata holes; pass allow_partial=True to accept them"
                )
            # without a sentinel the holes would merge as zeros and read as real pixels
            if group.nodata is None:
                raise ValueError(
                    f"group {group_id!r} is missing {len(missing)} of {group.expected} tiles "
                    f"(tile_ids {listed}) and its tiles declare no nodata to fill the holes with — "
                    "rebase(nodata=...) the raster before tiles(), or add the missing tiles"
                )
        return self._close_all()

    def _close_all(self) -> Iterator[GeoRaster]:
        """Merge and release every in-flight group, in first-seen order.

        Yields:
            One GeoRaster per group.
        """
        for group_id in list(self._groups):
            yield self._close(group_id)

    def _open_group(self, stamp: TilingInfo, tile: GeoTile) -> _Group:
        """Start an accumulator for a group's first tile.

        Args:
            stamp: That tile's own tiling stamp.
            tile: The tile, read for its non-spatial dims and pixel shape.

        Returns:
            New group state. The caller registers it after the first tile
            is added successfully.

        Raises:
            ValueError: `window` was given for tiles cut without an overlap.
        """
        from tiler import Merger

        overlapping = any(stamp.overlap) if isinstance(stamp.overlap, tuple) else stamp.overlap > 0
        if self._window is not None and not overlapping:
            raise ValueError("window= weighs tiles against each other, which needs tiles cut with an overlap")

        # every non-spatial axis flattens into one channel axis, which Tiler holds out of the tiling
        lead = tuple(str(dim) for dim in tile.data.dims[:-2])
        lead_shape = tuple(int(tile.data.sizes[dim]) for dim in lead)
        tiler = stamp.tiler(channels=math.prod(lead_shape))

        group = _Group(
            info=stamp,
            config=_tiling_config(stamp),
            tiler=tiler,
            merger=Merger(tiler, window=self._window),
            geobox=_whole_geobox(stamp, tile, tiler),
            lead=lead,
            lead_shape=lead_shape,
            lead_coords={dim: np.asarray(tile.data.coords[dim].values) for dim in lead},
            dtype=tile.data.dtype,
            nodata=tile.nodata,
            expected=len(tiler),
            pixel_shape=tuple(tile.data.shape),
        )
        return group

    @staticmethod
    def _validate(group: _Group, stamp: TilingInfo, tile: GeoTile) -> None:
        """Check one tile belongs to the group it routed to.

        Args:
            group: The group the tile routed to.
            stamp: That tile's own tiling stamp.
            tile: The tile itself.

        Raises:
            ValueError: the stamp disagrees with the group's config, the
                pixel shape differs, or the `tile_id` was already added.
        """
        if _tiling_config(stamp) != group.config:
            raise ValueError(
                f"tile at {tile.stem} wasn't cut by the same tiles() call as group {group.info.group_id!r}"
            )
        if not 0 <= stamp.tile_id < group.expected:
            raise ValueError(
                f"tile_id {stamp.tile_id} is outside group {group.info.group_id!r}'s "
                f"0..{group.expected - 1} range"
            )
        if tuple(tile.data.dims[:-2]) != group.lead:
            raise ValueError(
                f"tile at {tile.stem} has non-spatial dims {tuple(tile.data.dims[:-2])}, "
                f"not the group's {group.lead}"
            )
        if tuple(tile.data.shape) != group.pixel_shape:
            raise ValueError(
                f"tile at {tile.stem} holds {tuple(tile.data.shape)} pixels, not the group's {group.pixel_shape}"
            )
        if tile.data.dtype != group.dtype:
            raise ValueError(
                f"tile at {tile.stem} has dtype {tile.data.dtype}, not the group's {group.dtype}"
            )
        if not same_nodata(tile.nodata, group.nodata):
            raise ValueError(
                f"tile at {tile.stem} declares nodata {tile.nodata!r}, not the group's {group.nodata!r}"
            )
        for dim, expected in group.lead_coords.items():
            if not np.array_equal(tile.data.coords[dim].values, expected):
                raise ValueError(f"tile at {tile.stem} carries different {dim!r} coordinates from its group")
        if not geobox_matches(tile.anchor.geobox, _tile_geobox(group, stamp.tile_id)):
            raise ValueError(f"tile at {tile.stem} isn't at tile_id {stamp.tile_id}'s expected grid position")
        if stamp.tile_id in group.seen:
            raise ValueError(
                f"tile_id {stamp.tile_id} was already added to group {group.info.group_id!r} — "
                "adding it twice would double-weight those pixels"
            )

    def _close(self, group_id: str) -> GeoRaster:
        """Merge one group's accumulator into a surface and drop it.

        Args:
            group_id: Group to finish. Must be in flight.

        Returns:
            GeoRaster on the group's whole grid, `tiling` cleared, pixels
            in the tiles' own dtype. A pixel no tile weighed holds the
            group's nodata.
        """
        group = self._groups[group_id]
        reference = group.reference
        if reference is None:
            raise ValueError(f"group {group_id!r} holds no tile to rebuild from")

        height, width = group.info.data_shape
        merged = group.merger.merge(dtype=reference.data.dtype)

        # a pixel no tile weighed has no value to carry, so it reads as nodata
        if reference.nodata is not None:
            weights = np.asarray(group.merger.weights_sum)[:, :height, :width]
            merged[weights == 0] = reference.nodata

        geobox = group.geobox

        coords = {dim: reference.data.coords[dim] for dim in group.lead if dim in reference.data.coords}
        vectors = [vector for _, vector in sorted(group.vectors, key=lambda pair: pair[0])]
        result = GeoRaster(
            data=xr.DataArray(
                merged.reshape(*group.lead_shape, height, width),
                dims=(*group.lead, "y", "x"),
                coords={**coords, **dict(xr_coords(geobox, always_yx=True))},
                attrs={
                    key: value
                    for key, value in reference.data.attrs.items()
                    if key not in GeoExtension.registry()
                },
            ),
            anchor=reference.anchor.rebase(
                geobox=geobox,
                vector=GeoVector.concat(*vectors),
                tiling=None,
            ),
        )
        del self._groups[group_id]
        return result


def _tiling_config(stamp: TilingInfo) -> tuple[object, ...]:
    """Tiling fields shared by every tile in one group."""
    return stamp.data_shape, stamp.tile_shape, stamp.overlap, stamp.mode


def _whole_geobox(stamp: TilingInfo, tile: GeoTile, tiler: Tiler) -> GeoBox:
    """Reconstruct a group's whole grid from one correctly stamped tile."""
    if not 0 <= stamp.tile_id < len(tiler):
        raise ValueError(
            f"tile_id {stamp.tile_id} is outside group {stamp.group_id!r}'s 0..{len(tiler) - 1} range"
        )
    row, col = tiler.get_tile_bbox(stamp.tile_id)[0]
    return GeoBox(
        shape=stamp.data_shape,
        affine=tile.anchor.affine * Affine.translation(-int(col), -int(row)),
        crs=tile.anchor.geobox.crs,
    )


def _tile_geobox(group: _Group, tile_id: int) -> GeoBox:
    """Expected tile grid at one position in a group."""
    row, col = group.tiler.get_tile_bbox(tile_id)[0]
    return GeoBox(
        shape=group.info.tile_shape,
        affine=group.geobox.affine * Affine.translation(int(col), int(row)),
        crs=group.geobox.crs,
    )


def _stamp_id(tile: GeoTile) -> int:
    """That tile's own position in its group's grid.

    Args:
        tile: Tile already known to carry a tiling stamp.

    Returns:
        Its `tile_id`.

    Raises:
        ValueError: the tile carries no `tiling`.
    """
    if tile.tiling is None:
        raise ValueError(f"tile at {tile.stem} lost its tiling stamp mid-merge")
    return tile.tiling.tile_id

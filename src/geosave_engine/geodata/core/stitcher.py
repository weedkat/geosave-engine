"""Rebuilding rasters from the tiles they were cut into."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs import StitchWindow, Tiling, rebase

from .raster import UNPLACED_DIMENSIONS, GeoRaster, raster

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import xarray as xr
    from tiler import Merger, Tiler


def _shared_dtype(arrays: Sequence[xr.DataArray]) -> np.dtype:
    """Return the one dtype every array of a tile shares.

    Args:
        arrays: Arrays to compare, at least one.

    Returns:
        Shared pixel dtype.

    Raises:
        ValueError: The arrays carry more than one dtype, so merging them
            would promote them.
    """
    dtypes = {array.dtype for array in arrays}
    if len(dtypes) > 1:
        raise ValueError(
            f"tile variables carry different dtypes "
            f"({sorted(str(dtype) for dtype in dtypes)}); merging would promote them"
        )
    return dtypes.pop()


def _shared_fill(arrays: Sequence[xr.DataArray]) -> float | int | None:
    """Return the one fill value every array of a tile declares.

    Args:
        arrays: Arrays to compare, at least one.

    Returns:
        Shared fill value, or None when none declare one.

    Raises:
        ValueError: The arrays declare more than one fill value, so absence
            would mean two things across one raster.
    """
    declared = [_declared_fill(array) for array in arrays]
    first, *rest = declared
    if any(not _same_fill(value, first) for value in rest):
        raise ValueError(
            f"tile variables declare different fill values "
            f"({sorted(set(declared), key=str)}); absence means one value"
        )
    return first


def _same_fill(a: float | int | None, b: float | int | None) -> bool:
    """Whether two declared fill values are the same, NaN included.

    Plain `==` reads two declared NaNs as disagreeing, since `nan != nan`; NaN
    is also the only value unequal to itself, so that check doubles as the NaN
    test.

    Args:
        a: First declared value, or None for undeclared.
        b: Second declared value, or None for undeclared.

    Returns:
        True when `a` and `b` are the same fill value.
    """
    return a == b or (a != a and b != b)


# What a tile states about where it sits, rather than how the raster was cut.
_POSITION_FIELDS = {"source_id", "tile_index"}

_FILL_VALUE_ATTRIBUTE = "_FillValue"
_ODC_NODATA_ATTRIBUTE = "nodata"


def _declared_fill(array: xr.DataArray) -> float | int | None:
    """Read the fill value a variable declares, keeping its stored dtype.

    `odc.nodata` coerces the value to float, which promotes an integer raster
    wherever the result is filled, so the declaration is read as stored.

    Args:
        array: Data variable that may declare absence.

    Returns:
        The declared value, or None when the variable declares none.
    """
    declared = array.attrs.get(_FILL_VALUE_ATTRIBUTE)
    return declared if declared is not None else array.attrs.get(_ODC_NODATA_ATTRIBUTE)


@dataclass
class _Group:
    """One group's in-flight merge state.

    Args:
        layout: Tiling record every tile of the group states.
        tiler: Layout rebuilt over the whole raster, channels held out.
        merger: Accumulator every added tile is laid into.
        geobox: Whole-raster grid, or None when the tiles are unplaced.
        leading: Non-spatial dim names, in the tiles' own order.
        leading_shape: Length of each non-spatial dim.
        leading_coords: Coordinate values every non-spatial dim must carry.
        variables: Data variable names every tile carries, in order.
        dtype: Pixel dtype every variable of every tile must share.
        fill: Value marking absence, which every variable must agree on.
        placement: The layout every tile must state, its own position aside.
        seen: `tile_id`s laid in so far.
        attrs: Dataset attrs read off the lowest-`tile_id` tile.
    """

    layout: Tiling
    tiler: Tiler
    merger: Merger
    geobox: GeoBox | None
    leading: tuple[str, ...]
    leading_shape: tuple[int, ...]
    leading_coords: dict[str, np.ndarray]
    variables: tuple[str, ...]
    dtype: np.dtype[Any]
    fill: float | int | None
    placement: dict[str, Any]
    seen: set[int] = field(default_factory=set)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def expected(self) -> int:
        """Return how many tiles the raster was cut into."""
        return len(self.tiler)

    @property
    def spatial(self) -> tuple[str, str]:
        """Return the names of the two spatial dims the tiles carry."""
        return UNPLACED_DIMENSIONS if self.geobox is None else self.geobox.dimensions


class GeoStitcher:
    """Lay tiles back onto the rasters they were cut from, group by group.

    Tiles route by their own `Tiling.group_id`, so batches may mix groups
    and arrive in any order. A tile merges on `add` and may be dropped straight
    after, so a run never holds more than the surfaces still in flight.

    Args:
        window: How pixels that several tiles cover are weighed against each
            other. None weighs every tile alike, which rebuilds the source
            exactly. A tapering window smooths the seams and needs an overlap.

    Examples:
        >>> stitcher = GeoStitcher()
        >>> for batch in loader:
        ...     stitcher.add(*batch)
        ...     for rebuilt in stitcher.drain():
        ...         rebuilt.gs.to_cog(out / "prediction.tif")
        >>> rest = list(stitcher.flush())
    """

    def __init__(self, *, window: StitchWindow | np.ndarray | None = None) -> None:
        """Start a stitcher holding no group.

        Args:
            window: Weighting applied across each tile before averaging
                overlaps. None weighs every pixel alike.

        Raises:
            ValueError: `window` names no window `tiler` supports.
        """
        from tiler import Merger

        if isinstance(window, str) and window not in Merger.SUPPORTED_WINDOWS:
            raise ValueError(
                f"{window!r} is not a window tiler knows; pick one of "
                f"{sorted(Merger.SUPPORTED_WINDOWS)}"
            )
        self._window = window
        self._groups: dict[str, _Group] = {}

    def __len__(self) -> int:
        """Count the groups in flight."""
        return len(self._groups)

    def __repr__(self) -> str:
        """Describe the groups in flight and how complete each one is."""
        pending = ", ".join(
            f"{group_id}: {len(group.seen)}/{group.expected}"
            for group_id, group in self._groups.items()
        )
        return f"{type(self).__name__}(window={self._window!r}, groups={{{pending}}})"

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Return every in-flight group identifier, in first-seen order."""
        return tuple(self._groups)

    def add(self, *tiles: xr.Dataset) -> None:
        """Lay tiles into their own groups' accumulators.

        Args:
            *tiles: Tiles to merge, in any order and any mix of groups. Every
                one states the `Tiling` that `tile` stamped on it.

        Raises:
            ValueError: A tile states no tiling, disagrees with its group's
                layout, variables, dims, coordinates, dtype, fill value, or
                grid position, repeats a `tile_id` already added, or `window`
                was given for tiles cut without an overlap.
        """
        for tile in tiles:
            record = GeoRaster(tile).attrs.root.get(Tiling)
            if not isinstance(record, Tiling):
                raise ValueError(
                    "a tile states no tiling, so it cannot be placed back into a "
                    "raster; stitch tiles that ds.gs.tile produced"
                )

            opened = self._groups.get(record.source_id)
            group = opened or self._open(record, tile)
            self._check(group, record, tile)

            group.merger.add(record.tile_index, self._cube(group, tile))
            group.seen.add(record.tile_index)
            if opened is None:
                self._groups[record.source_id] = group

            # The lowest tile_id names the attrs, so a result never follows arrival order.
            if record.tile_index == min(group.seen):
                group.attrs = dict(tile.attrs)

    def is_complete(self, group_id: str) -> bool:
        """Report whether a group holds every tile its raster was cut into.

        Args:
            group_id: Group to check.

        Returns:
            True when the group holds all its tiles, False for one still
            missing a tile and for a group nothing was added for.
        """
        group = self._groups.get(group_id)
        return group is not None and len(group.seen) == group.expected

    def missing(self, group_id: str) -> tuple[int, ...]:
        """Name the tiles a group still needs before it can merge.

        Args:
            group_id: Group to inspect.

        Returns:
            Missing `tile_id`s in ascending order, empty for a complete group
            and for a group nothing was added for.
        """
        group = self._groups.get(group_id)
        if group is None:
            return ()
        return tuple(sorted(set(range(group.expected)) - group.seen))

    def drain(self) -> Iterator[xr.Dataset]:
        """Release every group that holds all its tiles.

        Call it between batches to write finished rasters and free their
        accumulators while the rest of the run is still going.

        Yields:
            One raster per completed group, in first-seen order.
        """
        for group_id in [name for name in self._groups if self.is_complete(name)]:
            yield self._close(group_id)

    def flush(self, *, allow_partial: bool = False) -> tuple[xr.Dataset, ...]:
        """Release every group still in flight.

        Every group is checked before the first raster is merged, so a rejected
        group raises from the call rather than part-way through iterating.

        Args:
            allow_partial: True merges a group missing tiles, leaving its
                declared fill value wherever no tile landed. False rejects one.

        Returns:
            One raster per group, in first-seen order. Every group is released,
            so the stitcher holds nothing afterwards.

        Raises:
            ValueError: A group is missing tiles and `allow_partial` is False,
                or it is missing tiles and states no fill value to mark the
                holes they leave.
        """
        for group_id, group in self._groups.items():
            absent = self.missing(group_id)
            if not absent:
                continue
            listed = f"{list(absent[:10])}{'...' if len(absent) > 10 else ''}"
            if not allow_partial:
                raise ValueError(
                    f"group {group_id!r} is missing {len(absent)} of "
                    f"{group.expected} tiles (tile_ids {listed}); merging it would "
                    f"leave holes, so pass allow_partial=True to accept them"
                )
            if group.fill is None:
                raise ValueError(
                    f"group {group_id!r} is missing {len(absent)} of "
                    f"{group.expected} tiles (tile_ids {listed}) and states no fill "
                    f"value to mark the holes with; declare one with "
                    f"Packing(fill_value=...) before tiling, or add the missing tiles"
                )
        return tuple(self._close_all())

    def _close_all(self) -> Iterator[xr.Dataset]:
        """Merge and release every in-flight group, in first-seen order.

        Yields:
            One raster per group.
        """
        for group_id in list(self._groups):
            yield self._close(group_id)

    def _cube(self, group: _Group, tile: xr.Dataset) -> np.ndarray:
        """Flatten one tile's variables and non-spatial axes into one cube.

        Args:
            group: Group the tile belongs to.
            tile: Tile to read.

        Returns:
            Array shaped as the group's tiler expects, channels first.
        """
        return (
            GeoRaster(tile)
            .to_numpy(group.variables)
            .reshape(tuple(group.tiler.tile_shape))
        )

    def _open(self, record: Tiling, tile: xr.Dataset) -> _Group:
        """Start an accumulator for a group's first tile.

        Args:
            record: That tile's own tiling record.
            tile: The tile, read for its dims, variables, and grid.

        Returns:
            New group state, which the caller registers once the tile is added.

        Raises:
            ValueError: `window` was given for tiles cut without an overlap,
                the variables carry different dtypes, or they declare
                different fill values.
        """
        from tiler import Merger

        overlapping = (
            any(record.tile_overlap)
            if isinstance(record.tile_overlap, tuple)
            else record.tile_overlap > 0
        )
        if self._window is not None and not overlapping:
            raise ValueError(
                "a window weighs tiles against each other, which needs tiles cut "
                "with an overlap; tile with overlap=... or stitch without a window"
            )

        grid = tile.odc.geobox
        spatial = GeoRaster(tile).grid_dims
        variables = GeoRaster(tile).variables
        leading = tuple(
            str(dim) for dim in tile[variables[0]].dims if dim not in spatial
        )
        leading_shape = tuple(int(tile.sizes[dim]) for dim in leading)
        tiler = record.tiler(channels=len(variables) * math.prod(leading_shape))

        arrays = [tile[name] for name in variables]
        return _Group(
            layout=record,
            tiler=tiler,
            merger=Merger(tiler, window=self._window),
            geobox=None if grid is None else _whole_geobox(record, grid, tiler),
            leading=leading,
            leading_shape=leading_shape,
            leading_coords={
                dim: np.asarray(tile.coords[dim].values)
                for dim in leading
                if dim in tile.coords
            },
            variables=variables,
            dtype=_shared_dtype(arrays),
            fill=_shared_fill(arrays),
            placement=record.model_dump(exclude=_POSITION_FIELDS),
        )

    def _check(self, group: _Group, record: Tiling, tile: xr.Dataset) -> None:
        """Refuse a tile that does not belong where it routed.

        Args:
            group: Group the tile routed to.
            record: That tile's own tiling record.
            tile: The tile itself.

        Raises:
            ValueError: The record disagrees with the group's layout, the
                variables, dims, coordinates, dtype, or fill value differ, the
                tile sits at another grid position, or its `tile_id` was
                already added.
        """
        if record.model_dump(exclude=_POSITION_FIELDS) != group.placement:
            raise ValueError(
                f"a tile of group {record.source_id!r} was cut on a different "
                f"layout from the group's; stitch one tiling operation at a time"
            )
        if not 0 <= record.tile_index < group.expected:
            raise ValueError(
                f"tile_id {record.tile_index} is outside group {record.source_id!r}'s "
                f"0..{group.expected - 1} range"
            )
        if record.tile_index in group.seen:
            raise ValueError(
                f"tile_id {record.tile_index} was already added to group "
                f"{record.source_id!r}; adding it twice double-weights those pixels"
            )
        if GeoRaster(tile).variables != group.variables:
            raise ValueError(
                f"a tile of group {record.source_id!r} carries "
                f"{GeoRaster(tile).variables} but the group carries "
                f"{group.variables}; every tile holds the same variables"
            )
        carried = tuple(
            str(dim)
            for dim in tile[group.variables[0]].dims
            if dim not in group.spatial
        )
        if carried != group.leading:
            raise ValueError(
                f"a tile of group {record.source_id!r} carries non-spatial dims "
                f"{carried}, not the group's {group.leading}"
            )
        arrays = [tile[name] for name in group.variables]
        if _shared_dtype(arrays) != group.dtype:
            raise ValueError(
                f"a tile of group {record.source_id!r} carries dtype "
                f"{tile[group.variables[0]].dtype}, not the group's {group.dtype}; "
                f"merging would promote them"
            )
        if not _same_fill(_shared_fill(arrays), group.fill):
            raise ValueError(
                f"a tile of group {record.source_id!r} declares a different fill "
                f"value from the group's {group.fill!r}; absence means one value "
                f"across a raster"
            )
        drifted = sorted(
            dim
            for dim, labels in group.leading_coords.items()
            if not np.array_equal(np.asarray(tile.coords[dim].values), labels)
        )
        if drifted:
            raise ValueError(
                f"a tile of group {record.source_id!r} carries different {drifted} "
                f"coordinates from its group; it was cut from another raster"
            )
        if group.geobox is not None and tile.odc.geobox != _tile_geobox(
            group, record.tile_index
        ):
            raise ValueError(
                f"tile_id {record.tile_index} of group {record.source_id!r} is not on "
                f"the grid its position names; it was cut from another raster"
            )

    def _close(self, group_id: str) -> xr.Dataset:
        """Merge one group's accumulator into a raster and drop it.

        Args:
            group_id: Group to finish, which must be in flight.

        Returns:
            Raster on the group's whole grid, without the tiling record, in the
            tiles' own dtype. A pixel no tile weighed holds the fill value.

        Raises:
            ValueError: A tapering window left pixels unweighed and no fill
                value marks them absent.
        """
        group = self._groups.pop(group_id)
        height, width = group.layout.source_shape
        merged = group.merger.merge(unpad=True, dtype=group.dtype)

        fill = group.fill
        # An even weighting covers every pixel of a whole raster, so nothing is unweighed.
        weighed = self._window is None and len(group.seen) == group.expected
        unweighted = (
            np.zeros((), dtype=bool)
            if weighed
            else np.asarray(group.merger.weights_sum)[:, :height, :width] == 0
        )
        if unweighted.any():
            if fill is None:
                raise ValueError(
                    f"group {group_id!r} leaves {int(unweighted.sum())} pixels "
                    f"unweighted, which a tapering window does at the raster's outer "
                    f"frame; stitch without a window, or declare a fill value with "
                    f"Packing(fill_value=...) to mark them absent"
                )
            # A pixel no tile weighed carries no value, so it reads as absent.
            merged[unweighted] = fill

        # Both accumulators are whole-raster sized and nothing reads them again.
        del group.merger

        # to_numpy laid the variables after the leading axes, so they unflatten there.
        cube = merged.reshape(*group.leading_shape, len(group.variables), height, width)
        rebuilt = raster(
            dict(zip(group.variables, np.moveaxis(cube, -3, 0), strict=True)),
            group.geobox,
            nodata=fill,
            **{dim: group.leading_coords.get(dim) for dim in group.leading},
        )
        rebuilt.attrs = {**rebuilt.attrs, **group.attrs}
        # The rebuilt raster is whole, so the tile stamp that placed one tile is gone.
        return rebase(rebuilt, tiling=None)


def _whole_geobox(record: Tiling, grid: GeoBox, tiler: Tiler) -> GeoBox:
    """Reconstruct a group's whole grid from one placed tile.

    Args:
        record: That tile's own tiling record.
        grid: That tile's own grid.
        tiler: Layout the group was cut on.

    Returns:
        Grid covering the whole raster the tiles were cut from.
    """
    row, column = tiler.get_tile_bbox(record.tile_index)[0]
    return GeoBox(
        shape=record.source_shape,
        affine=grid.translate_pix(-int(column), -int(row)).affine,
        crs=grid.crs,
    )


def _tile_geobox(group: _Group, tile_id: int) -> GeoBox:
    """Name the grid a tile at one position must sit on.

    Args:
        group: Group the tile belongs to, placed.
        tile_id: Position to place.

    Returns:
        Grid covering that position, cut from the whole grid the way `tile`
        cut it.
    """
    whole = cast("GeoBox", group.geobox)  # the caller tests the group is placed
    row, column = group.tiler.get_tile_bbox(tile_id)[0]
    height, width = group.layout.tile_shape
    return whole[int(row) : int(row) + height, int(column) : int(column) + width]

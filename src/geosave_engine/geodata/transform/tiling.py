"""Cutting rasters into model-sized tiles and merging results back onto them.

Examples:
    Two scenes cut together, run through a model, merged back into two
    rasters::

        tiles = Tiles([scene_a, scene_b], (256, 256), overlap=32)
        merger = tiles.merger()

        for index in range(len(tiles)):
            merger.add({index: predict(tiles[index])})

        predictions = merger.merge()   # {0: ..., 1: ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import xarray as xr
from odc.geo.xr import xr_coords

from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.core.convention import BAND_DIMENSION, CRS_COORDINATE
from geosave_engine.geodata.core.stack import stack

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from odc.geo.geobox import GeoBox
    from tiler import Merger, Tiler

    from geosave_engine.geodata.core.array import DataArray

# numpy's padding kinds, as xarray spells them.
type TilingMode = Literal["reflect", "edge", "constant", "wrap"]

# tiler.Merger's own window names, weighing a tile from its centre to its edge.
type StitchWindow = Literal[
    "boxcar",  # 1.0 throughout: a plain average, rebuilding a raster exactly
    "hamming",  # gentlest taper, still 0.080 at the edge
    "triang",  # linear ramp, 0.008 at the edge
    "bartlett",  # linear ramp, 0.0 at the edge
    "hann",  # cosine, 0.500 a quarter in, 0.0 at the edge
    "barthann",  # as hann, 0.500 a quarter in
    "bohman",  # 0.318 a quarter in
    "parzen",  # 0.253 a quarter in
    "blackman",  # 0.340 a quarter in
    "nuttall",  # 0.227 a quarter in
    "blackmanharris",  # strongest taper, 0.217 a quarter in
    "overlap-tile",  # 1 inside the shared band, 0 across it: one tile per pixel
]

_PADDING_KINDS = get_args(TilingMode.__value__)

# What a raster may be. A stack shares one grid, so one tiler cuts every group.
type Raster = xr.Dataset | xr.DataArray | xr.DataTree


def _frame(tile_shape: tuple[int, int], overlap: int) -> list[tuple[int, int]]:
    """Measure the frame a window needs around every raster.

    A window fades a tile to nothing at its own edges, which neighbouring
    tiles make up for. A raster's outermost pixels have no neighbour, so they
    are framed with pixels to fade away instead.

    Args:
        tile_shape: Tile height and width in pixels.
        overlap: Pixels neighbouring tiles share. Zero needs no frame, because
            a cut without overlap takes no window.

    Returns:
        Pixels to add before and after, per axis.

    Examples:
        >>> _frame((256, 256), 32)
        [(112, 112), (112, 112)]
    """
    if not overlap:
        return [(0, 0), (0, 0)]
    from tiler import Tiler

    layout = Tiler(tile_shape, tile_shape, overlap=overlap)
    return [(int(near), int(far)) for near, far in layout.calculate_padding()[1]]


def _cut_pixels[T: (xr.Dataset, xr.DataArray)](
    pixels: T,
    window: dict[str, slice],
    widths: dict[str, tuple[int, int]],
    mode: TilingMode,
    coords: dict[Hashable, xr.DataArray],
) -> T:
    """Take one window out of a raster, extend it, and relabel what it spans.

    Args:
        pixels: Raster to reshape.
        window: Row and column slices to take, which clip at the raster's own
            edge. Empty takes the whole raster.
        widths: Pixels to add before and after, per grid dimension.
        mode: How the added pixels are fabricated.
        coords: Coordinates the result carries, since padding leaves the ones
            it invents unlabelled.

    Returns:
        The windowed, extended and relabelled raster.
    """
    pixels = pixels.isel(window)
    if any(before or after for before, after in widths.values()):
        pixels = pixels.pad(widths, mode=mode)
    return pixels.assign_coords(coords)


def _cut(
    raster: Raster,
    window: dict[str, slice],
    widths: dict[str, tuple[int, int]],
    mode: TilingMode,
    coords: dict[Hashable, xr.DataArray],
) -> Raster:
    """Reshape a raster of any kind, a stack group by group.

    Args:
        raster: Raster to reshape, of any kind.
        window: Row and column slices to take, empty for the whole raster.
        widths: Pixels to add before and after, per grid dimension.
        mode: How the added pixels are fabricated.
        coords: Coordinates the result carries.

    Returns:
        Raster of the same kind as `raster`.
    """
    if isinstance(raster, xr.DataTree):
        return stack(
            {
                name: _cut_pixels(group, window, widths, mode, coords)
                for name, group in raster.gs.rasters.items()
            }
        )
    # Narrowed one kind at a time: _reshape keeps its kind, so takes no union.
    if isinstance(raster, xr.Dataset):
        return _cut_pixels(raster, window, widths, mode, coords)
    return _cut_pixels(raster, window, widths, mode, coords)


@dataclass(frozen=True, eq=False)
class _Raster:
    """One raster prepared for cutting.

    Args:
        pixels: The raster, carrying a frame where the tiles overlap.
        tiler: Layout over the framed raster.
        coords: Its coordinates, run out to wherever the last tile reaches, so
            a tile slices its own rather than deriving them.
        grid: The raster's own geobox, naming its dimensions and where a
            merged result lands back.
    """

    pixels: Raster
    tiler: Tiler
    coords: dict[Hashable, xr.DataArray]
    grid: GeoBox


class Tiles:
    """Number every tile of several rasters under one sequence.

    A tile is cut on access, stays lazy while its raster is, and states
    nothing about where it came from — the number it was asked by is what
    routes its result back. Rasters need not share a shape.

    Args:
        rasters: Georeferenced rasters to cut, each a Dataset, a DataArray or
            a stack DataTree, in the order they are numbered.
        tile_shape: Tile height and width in pixels.
        overlap: Pixels neighbouring tiles share, which a window needs to weigh
            them against each other. Sharing any also frames every raster, so
            no pixel of one sits on a tile's own edge.
        mode: How a raster is extended wherever a tile reaches past it.

    Raises:
        ValueError: `rasters` is empty, one carries no geobox, `mode` names no
            padding kind, or `tile_shape` exceeds a raster in either axis.

    Examples:
        >>> tiles = Tiles([scene_a, scene_b], (256, 256), overlap=32)
        >>> len(tiles), tiles[0].odc.geobox.shape
        (32, Shape2d(x=256, y=256))
    """

    def __init__(
        self,
        rasters: Sequence[Raster],
        tile_shape: tuple[int, int],
        *,
        overlap: int = 0,
        mode: TilingMode = "reflect",
    ) -> None:
        """Cut nothing yet, holding one tiler per raster and their numbering."""
        from tiler import Tiler

        if not rasters:
            raise ValueError("no rasters to cut; pass at least one")
        if mode not in _PADDING_KINDS:
            raise ValueError(
                f"{mode!r} is not a padding kind; extend a raster with one of "
                f"{_PADDING_KINDS}"
            )

        # Declared, because inference widens a Literal on assignment to an attribute.
        self.tile_shape: tuple[int, int] = tile_shape
        self.overlap = overlap
        self.mode: TilingMode = mode
        self._frame = _frame(tile_shape, overlap)

        prepared: list[_Raster] = []
        for ordinal, pixels in enumerate(rasters):
            try:
                grid = pixels.gs.geobox
            except ValueError as absent:
                raise ValueError(
                    f"raster {ordinal} carries no geobox, so its tiles could not be "
                    f"placed; write a CRS onto it first"
                ) from absent
            rows, columns = grid.dimensions
            if tile_shape[0] > grid.shape[0] or tile_shape[1] > grid.shape[1]:
                raise ValueError(
                    f"tile shape {tile_shape} is larger than raster {ordinal}'s own "
                    f"{tuple(grid.shape)}; a tile is cut from the raster, so it "
                    f"cannot exceed it"
                )

            # GeoBox.pad takes columns before rows, unlike the frame's own order.
            framed = grid.pad(self._frame[1][0], self._frame[0][0])
            if overlap:
                pixels = _cut(
                    pixels,
                    {},
                    {rows: self._frame[0], columns: self._frame[1]},
                    mode,
                    xr_coords(framed),
                )
            tiler = Tiler(
                data_shape=tuple(framed.shape),
                tile_shape=tile_shape,
                overlap=overlap,
                mode=mode,
            )

            far = tiler.get_tile_bbox(len(tiler) - 1)[1]
            span = framed[0 : int(far[0]), 0 : int(far[1])]
            prepared.append(_Raster(pixels, tiler, xr_coords(span), grid))

        self._rasters = tuple(prepared)

    def __len__(self) -> int:
        """Count the tiles of every raster together.

        Returns:
            Total tile count across all rasters.
        """
        return sum(len(raster.tiler) for raster in self._rasters)

    def __repr__(self) -> str:
        """Describe the rasters cut and how many tiles they carry."""
        return (
            f"{type(self).__name__}({len(self._rasters)} rasters, "
            f"tile_shape={self.tile_shape}, {len(self)} tiles)"
        )

    def __getitem__(self, index: int) -> Raster:
        """Cut the tile this number names.

        Args:
            index: Tile number in `range(len(self))`.

        Returns:
            Tile of `tile_shape` on its own geobox, of whichever kind its
            raster is, extended wherever it reaches past that raster.

        Raises:
            IndexError: `index` falls outside the numbering.
        """
        ordinal, tile_id = self.locate(index)
        raster = self._rasters[ordinal]
        rows, columns = raster.grid.dimensions
        near, far = raster.tiler.get_tile_bbox(tile_id)
        top, left = int(near[0]), int(near[1])
        bottom, right = int(far[0]), int(far[1])

        # isel clips at the raster's edge, so a trailing tile is padded back out.
        reach = raster.tiler.data_shape
        return _cut(
            raster.pixels,
            {rows: slice(top, bottom), columns: slice(left, right)},
            {
                rows: (0, max(0, bottom - int(reach[0]))),
                columns: (0, max(0, right - int(reach[1]))),
            },
            self.mode,
            {
                rows: raster.coords[rows][top:bottom],
                columns: raster.coords[columns][left:right],
                CRS_COORDINATE: raster.coords[CRS_COORDINATE],
            },
        )

    def merger(self, *, window: StitchWindow | None = None) -> TileMerger:
        """Build the accumulator that lays this cut's results back down.

        Args:
            window: How pixels several tiles cover are weighed against each
                other. None weighs every tile alike, rebuilding a raster
                exactly. A tapering window blends the seams, down to
                `"overlap-tile"`, which lets one tile alone answer each pixel.

        Returns:
            Accumulator holding no raster yet.

        Raises:
            ValueError: `window` is given for a cut whose tiles do not overlap,
                which leaves it nothing to weigh.
        """
        if window is not None and not self.overlap:
            raise ValueError(
                f"{window!r} weighs tiles against each other, which needs tiles cut "
                f"with an overlap; cut with overlap=... or merge without a window"
            )
        return TileMerger(self, window=window)

    def locate(self, index: int) -> tuple[int, int]:
        """Resolve a tile number into the raster holding it and its tile id.

        Args:
            index: Tile number in `range(len(self))`.

        Returns:
            The raster's place in the order it was cut in, and the tile's id
            within that raster.

        Raises:
            IndexError: `index` falls outside the numbering.

        Examples:
            >>> tiles.locate(40)
            (1, 8)
        """
        remaining = index + len(self) if index < 0 else index
        if remaining >= 0:
            for ordinal, raster in enumerate(self._rasters):
                if remaining < len(raster.tiler):
                    return ordinal, remaining
                remaining -= len(raster.tiler)
        raise IndexError(
            f"tile {index} is outside this cut, which numbers 0 to {len(self) - 1}"
        )


class TileMerger:
    """Lay tile results back onto the rasters they were cut from.

    A result is routed by the tile number it answers, so results may arrive in
    any order and several rasters may arrive together. A raster's accumulator
    opens on its first result, taking its band count and dtype from it.

    Args:
        tiles: Cut the results answer.
        window: How pixels several tiles cover are weighed against each other.
            None weighs every tile alike.

    Examples:
        >>> merger = tiles.merger()
        >>> merger.add(dict(zip(batch["index"].tolist(), predictions)))
        >>> merger.merge()[0].odc.geobox == scene_a.odc.geobox
        True
    """

    def __init__(self, tiles: Tiles, *, window: StitchWindow | None = None) -> None:
        """Start an accumulator holding no raster."""
        self._tiles = tiles
        self._window = window
        self._mergers: dict[_Raster, Merger] = {}
        self._tile_ids: dict[_Raster, set[int]] = {}

    def __repr__(self) -> str:
        """Describe how many of the cut's rasters have taken a result."""
        return (
            f"{type(self).__name__}({len(self._mergers)} of "
            f"{len(self._tiles._rasters)} rasters in flight)"
        )

    def add(self, results: Mapping[int, np.ndarray]) -> None:
        """Take each result under the tile number it answers.

        Args:
            results: Tile number mapped to that tile's result, shaped `(y, x)`
                or `(band, y, x)`. A batch states one entry per row.

        Raises:
            IndexError: A number falls outside the cut.
            ValueError: A result does not match the tile shape.

        Examples:
            >>> merger.add({3: prediction})
            >>> merger.add(dict(zip(batch["index"].tolist(), predictions)))
        """
        for number, result in results.items():
            pixels = np.asarray(result)
            ordinal, tile_id = self._tiles.locate(int(number))
            raster = self._tiles._rasters[ordinal]
            if raster not in self._mergers:
                self._mergers[raster] = self._build_merger(raster, pixels)
                self._tile_ids[raster] = set()
            self._mergers[raster].add(tile_id, pixels)
            self._tile_ids[raster].add(tile_id)

    @property
    def pending(self) -> dict[int, int]:
        """Count the tiles each unfinished raster is still waiting on.

        Returns:
            Raster place in the cut mapped to how many of its tiles are
            missing, empty once every raster has been merged and released.

        Examples:
            >>> merger.pending
            {1: 12}
        """
        waiting = {}
        for ordinal, raster in enumerate(self._tiles._rasters):
            if raster in self._mergers:
                missing = len(raster.tiler) - len(self._tile_ids[raster])
                if missing:
                    waiting[ordinal] = missing
        return waiting

    def merge(self) -> dict[int, DataArray]:
        """Rebuild and release every raster whose tiles have all arrived.

        A raster still waiting on tiles is left in flight, so results may be
        drained as a run goes and each raster freed once it is whole. A raster
        no result has named at all is not this merger's to rebuild.

        Returns:
            Raster place in the cut mapped to its one rebuilt band, on that
            raster's own geobox. A result carrying bands keeps them on a
            leading `band` axis.

        Examples:
            >>> for ordinal, prediction in merger.merge().items():
            ...     prediction.gs.to_cog(f"{ordinal}.tif")
            >>> merger.pending
            {}
        """
        rebuilt: dict[int, DataArray] = {}
        for ordinal, raster in enumerate(self._tiles._rasters):
            if raster not in self._mergers:
                continue
            if len(self._tile_ids[raster]) < len(raster.tiler):
                continue
            pixels = self._mergers.pop(raster).merge(
                unpad=True, extra_padding=self._tiles._frame
            )
            del self._tile_ids[raster]
            leading = {BAND_DIMENSION: None} if pixels.ndim == 3 else {}
            rebuilt[ordinal] = array(pixels, raster.grid, **leading)
        return rebuilt

    def _build_merger(self, raster: _Raster, pixels: np.ndarray) -> Merger:
        """Build one raster's accumulator, shaped by the first result it takes.

        Args:
            raster: The raster the result belongs to.
            pixels: Its first result, shaped `(y, x)` or `(band, y, x)`.

        Returns:
            Accumulator over that raster's tiler, carrying no tile yet.
        """
        from tiler import Merger

        return Merger(
            raster.tiler,
            window=self._window,
            logits=pixels.shape[0] if pixels.ndim == 3 else 0,
            # float32 alone would round an int32 or float64 result.
            data_dtype=np.promote_types(pixels.dtype, np.float32),
            save_visits=False,
        )

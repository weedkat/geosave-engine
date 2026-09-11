"""Cutting a raster or a raster stack into tiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from odc.geo.xr import xr_coords

from geosave_engine.geodata.attrs import Tiling
from geosave_engine.geodata.core.raster import GeoRaster

_FILL_VALUE_ATTRIBUTE = "_FillValue"
_ODC_NODATA_ATTRIBUTE = "nodata"


def declared_fill(array: xr.DataArray) -> float | int | None:
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


if TYPE_CHECKING:
    import xarray as xr

    from geosave_engine.geodata.attrs import TilingMode


def tile(
    raster: xr.Dataset,
    shape: tuple[int, int],
    *,
    group_id: str,
    overlap: int | float | tuple[int, int] = 0,
    mode: TilingMode = "reflect",
) -> list[xr.Dataset]:
    """Cut a raster into equally shaped tiles.

    The trailing edges are padded rather than truncated, so every tile matches
    `shape` and a model reading them sees one input spec. Tiles read as
    ordinary rasters and stay lazy until their pixels are needed.

    Args:
        raster: Dataset to cut, placed or unplaced.
        shape: Tile height and width in pixels.
        group_id: Identifier every tile of this operation shares, used to route
            tiles back to this raster when stitching.
        overlap: Shared pixels as a count, a fraction in `[0, 1)`, or row-column
            counts.
        mode: How the trailing edges are padded. `"constant"` extends every
            variable with its own declared fill value.

    Returns:
        Tiles in row-major order, each stamped with its own `Tiling` and
        placed when `raster` is placed.

    Raises:
        ValueError: `raster` names no spatial dimensions, `shape` exceeds its
            own shape in either axis, or `mode` is `"constant"` while a
            variable declares no fill value.

    Examples:
        >>> tiles = transform.tiles.tile(ds, (256, 256), group_id="scene-001")
        >>> len(tiles), tiles[3].gs.geobox.shape
        (16, Shape2d(x=256, y=256))
    """
    grid = raster.odc.geobox
    rows, columns = GeoRaster(raster).grid_dims
    absent = [name for name in (rows, columns) if name not in raster.dims]
    if absent:
        raise ValueError(
            f"{absent} are not dimensions of this raster, so it has no spatial "
            f"axes to cut; it carries {tuple(raster.dims)}"
        )

    source = (raster.sizes[rows], raster.sizes[columns])
    if shape[0] > source[0] or shape[1] > source[1]:
        raise ValueError(
            f"tile shape {shape} is larger than the raster's own {source}; a tile "
            f"is cut from the raster, so it cannot exceed it"
        )

    layout = Tiling(
        source_id=group_id,
        tile_index=0,
        source_shape=source,
        tile_shape=shape,
        tile_overlap=overlap,
        tile_padding=mode,
    )
    tiler = layout.tiler()
    corners = [tiler.get_tile_bbox(tile_id) for tile_id in range(len(tiler))]
    overhang = (
        max(int(far[0]) for _, far in corners) - source[0],
        max(int(far[1]) for _, far in corners) - source[1],
    )

    options: dict[str, Any] = {}
    if mode == "constant":
        fills = {
            str(name): declared_fill(array) for name, array in raster.data_vars.items()
        }
        undeclared = sorted(name for name, fill in fills.items() if fill is None)
        if undeclared:
            raise ValueError(
                f"{undeclared} declare no fill value, so 'constant' padding has "
                f"nothing to extend them with; declare one with "
                f"Packing(fill_value=...) or pad with 'edge' or 'reflect'"
            )
        options["constant_values"] = fills

    # xarray does not short-circuit a zero-width pad; it copies every variable.
    padded = (
        raster
        if overhang == (0, 0)
        else raster.pad(
            {rows: (0, overhang[0]), columns: (0, overhang[1])}, mode=mode, **options
        )
    )

    tiles: list[xr.Dataset] = []
    for tile_id, (near, far) in enumerate(corners):
        top, left, bottom, right = int(near[0]), int(near[1]), int(far[0]), int(far[1])
        cut = padded.isel(
            {rows: slice(top, bottom), columns: slice(left, right)}
        ).gs.rebase(layout.model_copy(update={"tile_index": tile_id}))
        if grid is None:
            tiles.append(cut)
            continue
        # Padding mirrors the coordinate values, so the grid restates them.
        placed = cut.assign_coords(xr_coords(grid[top:bottom, left:right]))
        tiles.append(placed.gs.write_crs())
    return tiles


def tile_stack(
    tree: xr.DataTree,
    shape: tuple[int, int],
    *,
    group_id: str,
    overlap: int | float | tuple[int, int] = 0,
    mode: TilingMode = "reflect",
) -> list[xr.DataTree]:
    """Cut every group of a raster stack on one shared layout.

    A stack holds one exact grid, so each group is cut at the same positions
    and a tile pairs the groups covering the same ground.

    Args:
        tree: Raster stack to cut.
        shape: Tile height and width in pixels.
        group_id: Identifier every tile of this operation shares.
        overlap: Shared pixels as a count, a fraction in `[0, 1)`, or row-column
            counts.
        mode: How the trailing edges are padded. `"constant"` extends every
            variable with its own declared fill value.

    Returns:
        Stacks in row-major order, each holding every group cut to one tile and
        stamped with the shared `Tiling`.

    Raises:
        ValueError: `tree` holds no group, a group carries no locatable grid or
            sits on a different one, `shape` exceeds the stack's own shape, or
            `mode` is `"constant"` while a variable declares no fill value.

    Examples:
        >>> tiles = transform.tiles.tile_stack(scene, (256, 256), group_id="s")
        >>> tiles[3].gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    from geosave_engine.geodata.core.stack import GeoStack, stack

    groups = GeoStack(tree).groups
    if not groups:
        raise ValueError("a raster stack needs at least one group to cut")

    cut = {
        name: tile(
            tree[name].to_dataset(),
            shape,
            group_id=group_id,
            overlap=overlap,
            mode=mode,
        )
        for name in groups
    }
    return [
        stack({name: cut[name][tile_id] for name in groups})
        for tile_id in range(len(cut[groups[0]]))
    ]

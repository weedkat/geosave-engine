"""Describe and extract spatial chips from native rasters without computing pixels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

if TYPE_CHECKING:
    from tiler import Tiler


type PaddingMode = Literal["constant", "edge", "reflect", "wrap"]


def crop(
    data: xr.Dataset | xr.DataArray | xr.DataTree,
    offset: tuple[int, int],
    shape: tuple[int, int],
    *,
    padding: Sequence[tuple[int, int]] = ((0, 0), (0, 0)),
    mode: PaddingMode = "constant",
    constant_value: int | float | None = None,
) -> xr.Dataset | xr.DataArray | xr.DataTree:
    """Read a pixel window with the same padding used by its native Tiler.

    Args:
        data: Raster or stack; leading dimensions remain intact.
        offset: Row and column offsets in the original raster, before padding.
        shape: Requested height and width.
        padding: Halo widths before and after each spatial axis.
        mode: Padding mode for halo and fringe pixels.
        constant_value: Constant fill. None uses xarray's missing-value fill.

    Returns:
        Lazy window of the same native type. Georeferenced rasters retain
        the requested grid, including padded pixels outside the parent.
    """
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import map_groups

        return map_groups(
            data,
            lambda raster: cast(
                "xr.Dataset",
                crop(
                    raster,
                    offset,
                    shape,
                    padding=padding,
                    mode=mode,
                    constant_value=constant_value,
                ),
            ),
        )

    dims = data.gs.grid_dims
    grid = data.gs.geobox

    # Halo offsets are relative to the original parent, not its padded shape.
    if any(before or after for before, after in padding):
        data = _pad(data, dict(zip(dims, padding, strict=True)), mode, constant_value)
    starts = [
        start + before for start, (before, _) in zip(offset, padding, strict=True)
    ]
    data = data.isel(
        {
            dim: slice(start, start + size)
            for dim, start, size in zip(dims, starts, shape, strict=True)
        }
    )

    # Edge tiles may extend beyond the parent even when no halo was added.
    fringe = {
        dim: (0, size - data.sizes[dim]) for dim, size in zip(dims, shape, strict=True)
    }
    if any(after for _, after in fringe.values()):
        data = _pad(data, fringe, mode, constant_value)

    # Reflected/wrapped coordinate labels repeat; the output grid does not.
    if isinstance(grid, GeoBox):
        data = data.assign_coords(
            xr_coords(grid.translate_pix(offset[1], offset[0]).crop(shape))
        )
    return data


def _pad(
    data: xr.Dataset | xr.DataArray,
    widths: Mapping[str, tuple[int, int]],
    mode: PaddingMode,
    constant_value: int | float | None,
) -> xr.Dataset | xr.DataArray:
    """Pad pixels lazily, including reflection wider than the source."""
    # Dask truncates repeated reflection/wrapping. Pad small index arrays with
    # NumPy, then let xarray select pixels without computing the raster.
    if mode in ("reflect", "wrap"):
        return data.isel(
            {
                dim: np.pad(np.arange(data.sizes[dim]), width, mode=mode)
                for dim, width in widths.items()
            }
        )
    return data.pad(widths, mode=mode, constant_values=constant_value)


def chip_windows(
    parents: Mapping[str, xr.Dataset | xr.DataArray | xr.DataTree],
    tilers: Mapping[str, Tiler],
    *,
    padding: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> gpd.GeoDataFrame:
    """List the chip windows of each parent as rows.

    Args:
        parents: Original prepared scenes or frames keyed by persistent ID.
        tilers: Native spatial Tiler for each parent.
        padding: Halo widths used to extend each tiler's data shape.
            None means no halo; native fringe padding needs no entry.

    Returns:
        Metadata-only reference with an `id` column. Footprints use WGS84;
        unreferenced parents have null geometry and projection fields.

    Raises:
        ValueError: Parents are empty, IDs are empty, keys differ, or a
            tiler does not describe two spatial dimensions.

    Examples:
        >>> chips = chip_windows(parents, tilers)
        >>> chips.set_index("id").loc["scene-a/tile-0", "tile_id"]
        0
    """
    if not parents or parents.keys() != tilers.keys():
        raise ValueError("tilers must match at least one parent")
    if any(not isinstance(key, str) or not key for key in parents):
        raise ValueError("parent IDs must be non-empty strings")
    if padding is not None and padding.keys() != parents.keys():
        raise ValueError("padding must match parent keys")
    # Layout bounds describe pixels independently of geographic coordinates.
    rows = []
    for parent_id, parent in parents.items():
        tiler = tilers[parent_id]
        if len(tiler.data_shape) != 2:
            raise ValueError("reference tilers must have two spatial dimensions")
        widths = [(0, 0), (0, 0)] if padding is None else padding[parent_id]
        grid = parent.gs.geobox
        metadata = _raster_metadata(parent)
        for tile_id in range(len(tiler)):
            near, far = tiler.get_tile_bbox(tile_id)
            row_off, col_off = (
                int(start) - before
                for start, (before, _) in zip(near, widths, strict=True)
            )
            height, width = map(int, far - near)
            row = {
                "id": f"{parent_id}/tile-{tile_id}",
                "parent_id": parent_id,
                "tile_id": tile_id,
                "row_off": row_off,
                "col_off": col_off,
                "height": height,
                "width": width,
                "raster_metadata": metadata,
                "padding": [list(width) for width in widths],
                "padding_mode": tiler.mode,
                "padding_value": float(tiler.constant_value)
                if tiler.mode == "constant"
                else None,
                **_tile_grid(grid, (row_off, col_off), (height, width)),
            }
            rows.append(row)
    # Pixel-only references have an active geometry column, but no CRS.
    geographic = any(row["geometry"] is not None for row in rows)
    return gpd.GeoDataFrame(
        rows, geometry="geometry", crs="EPSG:4326" if geographic else None
    )


def _tile_grid(
    grid: GeoBox | None,
    offset: tuple[int, int],
    shape: tuple[int, int],
) -> dict[str, object]:
    """Describe a tile's exact grid, or null fields for unreferenced pixels."""
    if not isinstance(grid, GeoBox) or grid.crs is None:
        return {
            "geometry": None,
            "proj:shape": None,
            "proj:transform": None,
            "proj:code": None,
            "proj:wkt2": None,
        }

    tile = grid.translate_pix(offset[1], offset[0]).crop(shape)
    epsg = grid.crs.epsg
    return {
        "geometry": tile.extent.to_crs("EPSG:4326").geom,
        "proj:shape": shape,
        "proj:transform": tuple(tile.transform)[:6],
        "proj:code": f"EPSG:{epsg}" if epsg is not None else None,
        "proj:wkt2": grid.crs.to_wkt() if epsg is None else None,
    }


def _raster_metadata(
    data: xr.Dataset | xr.DataArray | xr.DataTree,
) -> dict[str, object]:
    """Record ordered time context for native tile references.

    Args:
        data: Native raster or stack supplying time coordinates and band names.

    Returns:
        Metadata by raster name, retaining each raster's timestamp order.
    """
    rasters = data.gs.rasters if isinstance(data, xr.DataTree) else {"image": data}
    metadata = {}
    for name, raster in rasters.items():
        times = raster.coords.get("time")
        labels = [] if times is None else np.atleast_1d(times.values)
        metadata[name] = {
            "times": np.datetime_as_string(labels).tolist()
            if times is not None and np.issubdtype(times.dtype, np.datetime64)
            else list(labels),
            "time_dimension": "time" in raster.dims,
            "time_dtype": str(times.dtype) if times is not None else None,
            "bands": list(raster.data_vars)
            if isinstance(raster, xr.Dataset)
            else [raster.name],
        }
    return metadata

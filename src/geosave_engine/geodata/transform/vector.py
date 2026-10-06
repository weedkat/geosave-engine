"""Convert explicitly between geolocated rasters and vector features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

from geosave_engine.geodata.transform import nodata

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from geosave_engine.geodata import GeoDataFrame
    from geosave_engine.geodata.core.anchor import GeoAnchor


def vectorize(
    flags: xr.DataArray,
    *,
    value_name: str = "value",
    mask: xr.DataArray | np.ndarray | None = None,
    connectivity: Literal[4, 8] = 4,
) -> GeoDataFrame:
    """Polygonize contiguous values from one geolocated flag plane.

    This operation computes lazy flags because geometry depends on their
    values.

    Args:
        flags: Two-dimensional categorical or flag array.
        value_name: Property column receiving each region's value.
        mask: Optional exact-grid mask selecting additional valid pixels.
        connectivity: Four- or eight-neighbour region connectivity.

    Returns:
        One row per contiguous flag region, in the raster CRS.

    Raises:
        ValueError: The array, mask, name, connectivity, or dtype is
            unsuitable for polygonization.
    """
    from rasterio.features import shapes
    from shapely.geometry import shape

    # Polygonization needs one realized numeric plane on a regular grid.
    geobox = flags.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError("flags carry no locatable grid")
    if flags.ndim != 2 or tuple(flags.dims) != tuple(geobox.dimensions):
        raise ValueError(
            "vectorize needs one two-dimensional spatial plane on "
            f"{geobox.dimensions}; select extra dimensions from {flags.dims} first"
        )
    if value_name == "geometry":
        raise ValueError("value_name must not replace the geometry column")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    values = np.asarray(flags.compute().values)
    if values.dtype == np.bool_:
        values = values.astype("uint8")
    supported = {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("float32"),
    }
    if values.dtype.kind in "iu" and values.dtype not in supported:
        # Rasterio polygonizes no other integer, and an argmax emits int64.
        bounds = np.iinfo("int32")
        if values.size and (values.min() < bounds.min or values.max() > bounds.max):
            raise ValueError(
                f"flags are {values.dtype} and hold values beyond int32, the "
                f"widest integer raster polygonization supports"
            )
        values = values.astype("int32")
    if values.dtype not in supported:
        raise ValueError(f"raster polygonization does not support dtype {values.dtype}")
    valid = np.isfinite(values)
    fill = nodata.fill_value(flags)
    if fill is not None:
        valid &= values != fill
    if mask is not None:
        if isinstance(mask, xr.DataArray):
            if mask.gs.geobox != geobox:
                raise ValueError("mask must be on the exact flags grid")
            selected = np.asarray(mask.compute().values)
        else:
            selected = np.asarray(mask)
        if selected.shape != values.shape:
            raise ValueError(
                f"mask shape {selected.shape} does not match flags {values.shape}"
            )
        valid &= selected.astype(bool)

    # Rasterio emits one geometry for each contiguous region.
    features = list(
        shapes(
            values,
            mask=valid,
            transform=geobox.transform,
            connectivity=connectivity,
        )
    )
    if not features:
        return cast(
            "GeoDataFrame",
            gpd.GeoDataFrame(
                {value_name: pd.Series(dtype=values.dtype)},
                geometry=gpd.GeoSeries([], crs=geobox.crs),
            ),
        )

    # Keep Rasterio's scalar values beside native Shapely geometries.
    scalar = values.dtype.type
    return cast(
        "GeoDataFrame",
        gpd.GeoDataFrame(
            {value_name: [scalar(value).item() for _, value in features]},
            # Regions touching at a corner trace one self-touching ring.
            geometry=gpd.GeoSeries(
                [shape(geometry) for geometry, _ in features], crs=geobox.crs
            ).make_valid(),
        ),
    )


def rasterize(
    vector: gpd.GeoDataFrame,
    like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
    *,
    column: str | None = None,
    fill: int | float | bool = 0,
    dtype: DTypeLike | None = None,
    all_touched: bool = False,
) -> xr.DataArray:
    """Burn vector geometries or one numeric property onto an exact grid.

    Args:
        vector: Features to burn.
        like: Anchor or geolocated xarray object supplying the target grid.
        column: Property to burn. None returns a boolean presence mask.
        fill: Value outside geometries.
        dtype: Output dtype for property values.
        all_touched: Burn every touched pixel instead of pixel centres.

    Returns:
        Eager geolocated array on the exact target grid.

    Raises:
        KeyError: The selected property is absent.
        ValueError: A geometry is null, empty, or invalid, or values or fill
            cannot be represented by the dtype.
    """
    from rasterio.features import rasterize as burn

    from geosave_engine.geodata.core.anchor import GeoAnchor
    from geosave_engine.geodata.core.array import array

    # The target owns the exact output grid; only burn geometry is reprojected.
    geobox = like.geobox if isinstance(like, GeoAnchor) else like.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError("rasterize target carries no regular grid CRS")
    # A geometry naming no ground burns wrong pixels without a word.
    shapes = vector.geometry
    if shapes.isna().any() or shapes.is_empty.any() or not shapes.is_valid.all():
        raise ValueError(
            "the vector holds a null, empty, or invalid geometry; drop those "
            "rows or repair them with geometry.make_valid() before rasterizing"
        )
    frame = vector.to_crs(geobox.crs)

    output_dtype: np.dtype
    if column is None:
        output_dtype = np.dtype("uint8")
        values = np.ones(len(frame), dtype=output_dtype)
        output_name = "mask"
        output_fill: int | float | bool = bool(fill)
    else:
        if column not in frame:
            raise KeyError(f"the vector has no {column!r} column")
        series = cast("pd.Series", frame[column])
        if series.isna().any():
            raise ValueError("rasterize values must not be null")
        try:
            output_dtype = np.dtype(
                dtype
                if dtype is not None
                else np.result_type(np.asarray(series.tolist()), type(fill))
            )
        except (TypeError, ValueError) as error:
            raise ValueError("rasterize needs a numeric raster dtype") from error
        if output_dtype.kind not in "iuf":
            raise ValueError(
                f"rasterize needs a numeric raster dtype, got {output_dtype}"
            )

        # Refuse wrapping, truncation, or opaque values before casting once.
        nodata.check_fill_fits(fill, output_dtype, column)
        for value in series.unique():
            scalar = value.item() if isinstance(value, np.generic) else value
            if not isinstance(scalar, (int, float)):
                raise ValueError("rasterize needs a numeric raster dtype")
            nodata.check_fill_fits(scalar, output_dtype, column)
        values = np.asarray(series.tolist()).astype(output_dtype)
        output_name = column
        output_fill = cast(
            "int | float | bool", np.asarray(fill).astype(output_dtype).item()
        )

    # Rasterio preserves row order, so later rows win where geometries overlap.
    if frame.empty:
        pixels = np.full(geobox.shape.yx, output_fill, dtype=output_dtype)
    else:
        pixels = burn(
            zip(frame.geometry, values, strict=True),
            out_shape=geobox.shape.yx,
            transform=geobox.transform,
            fill=output_fill,  # type: ignore[arg-type]
            dtype=output_dtype,
            all_touched=all_touched,
        )
        if pixels is None:
            raise RuntimeError("Rasterio returned no rasterized array")
    if column is None:
        pixels = pixels.view(bool)

    burned = array(pixels, geobox).rename(output_name)
    if isinstance(like, GeoAnchor):
        return burned
    # A sliced raster's coordinates drift from its geobox's own by float error,
    # so the result takes the target's, which is what aligns with it.
    source = like.dataset if isinstance(like, xr.DataTree) else like
    return burned.assign_coords({dim: source.coords[dim] for dim in geobox.dimensions})


def crop[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, region: Geometry | gpd.GeoSeries | gpd.GeoDataFrame, *, mask: bool = True
) -> T:
    """Cut a raster, band, or stack down to the extent of a region.

    Args:
        data: Dataset or DataArray on a regular grid, or a stack of them,
            which is cut group by group.
        region: Geometry, or geometries, to cut against, reprojected onto
            `data`'s CRS where they sit in another. Disconnected geometries
            share one cut, and `mask` blanks the pixels between them.
        mask: Also make nodata the pixels outside the geometries, which then
            take their own variable's fill value.

    Returns:
        `data` covering the region's extent, its dtype unchanged.

    Raises:
        TypeError: `region` carries no CRS, as a bare shapely geometry.
        ValueError: `data` carries no CRS, `region` is empty or does not
            overlap `data`, the crop leaves no regular grid to mask on, or
            `mask` is set while a variable carries no fill value.

    Examples:
        >>> crop(scene, field_boundaries).gs.geobox.shape
        (64, 48)
        >>> crop(scene, box(10, 10, 30, 30, "EPSG:32633"))
    """
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import map_groups

        return cast(
            "T", map_groups(data, lambda raster: crop(raster, region, mask=mask))
        )

    crs = data.odc.crs
    if crs is None:
        raise ValueError(
            f"{type(data).__name__} carries no CRS; write one with "
            f"gs.write_crs before cropping"
        )
    # A region follows the raster's grid; the raster's pixels never move here.
    vector = _as_frame(region).to_crs(crs)

    # odc's own apply_mask writes NaN, which promotes every integer variable.
    cut = cast("T", data.odc.crop(vector.gs.footprint, apply_mask=False))
    if isinstance(cut, xr.Dataset):
        cut = nodata.drop_source(cut)
    if not mask:
        return cut
    if not isinstance(cut.odc.geobox, GeoBox):
        raise ValueError(
            f"crop left {type(cut).__name__} with no regular grid, so mask has "
            f"nothing to rasterize the vector onto"
        )
    # Sliced coordinates drift from the grid's own by float error, so the
    # mask is handed over by position, which names the grid alone.
    return nodata.mask(cut, rasterize(vector, cut).values)


def _as_frame(region: Geometry | gpd.GeoSeries | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(region, gpd.GeoDataFrame):
        return region
    if isinstance(region, Geometry) and region.crs is not None:
        return gpd.GeoDataFrame(geometry=[region.geom], crs=region.crs.proj)
    if isinstance(region, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=region)
    # Shapely carries no CRS, and assuming one would cut the wrong ground.
    raise TypeError(
        f"{type(region).__name__} carries no CRS; pass an odc Geometry, "
        f"GeoSeries, or GeoDataFrame"
    )

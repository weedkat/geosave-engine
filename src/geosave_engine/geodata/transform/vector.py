"""Convert explicitly between geolocated rasters and vector features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.transform import nodata

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from geosave_engine.geodata.core.anchor import GeoAnchor
    from geosave_engine.geodata.core.vector import GeoVector


def vectorize(
    flags: xr.DataArray,
    *,
    value_name: str = "value",
    mask: xr.DataArray | np.ndarray | None = None,
    connectivity: Literal[4, 8] = 4,
) -> GeoVector:
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

    from geosave_engine.geodata.core.vector import GeoVector

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
    if values.dtype not in supported:
        raise ValueError(
            f"raster polygonization does not support dtype {values.dtype}"
        )
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
        return GeoVector(
            gpd.GeoDataFrame(
                {value_name: pd.Series(dtype=values.dtype)},
                geometry=gpd.GeoSeries([], crs=geobox.crs),
            )
        )

    # Keep Rasterio's scalar values beside native Shapely geometries.
    scalar = values.dtype.type
    return GeoVector(
        gpd.GeoDataFrame(
            {value_name: [scalar(value).item() for _, value in features]},
            geometry=[shape(geometry) for geometry, _ in features],
            crs=geobox.crs,
        )
    )


def rasterize(
    vector: GeoVector,
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
        ValueError: Values or fill cannot be represented by the dtype.
    """
    from rasterio.features import rasterize as burn

    from geosave_engine.geodata.core.anchor import GeoAnchor
    from geosave_engine.geodata.core.array import array

    # The target owns the exact output grid; only burn geometry is reprojected.
    geobox = like.geobox if isinstance(like, GeoAnchor) else like.gs.anchor.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError("rasterize target carries no regular grid CRS")
    frame = vector.to_crs(geobox.crs).gdf

    output_dtype: np.dtype
    if column is None:
        output_dtype = np.dtype("uint8")
        values = np.ones(len(frame), dtype=output_dtype)
        output_name = "mask"
        output_fill: int | float | bool = bool(fill)
    else:
        if column not in frame:
            raise KeyError(f"GeoVector has no {column!r} column")
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
        pixels = pixels.astype(bool)

    # Return the same native geolocated DataArray used by other transforms.
    return array(pixels, geobox).rename(output_name)

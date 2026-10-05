"""Feature-specific raster preparation and output construction."""

from __future__ import annotations

import numpy as np
import xarray as xr

from geosave_engine.geodata import attrs


def prepared_reflectance(
    raster: xr.Dataset, *variables: str
) -> tuple[xr.DataArray, ...]:
    """Select reflectance variables after requiring explicit preparation."""
    bands = tuple(raster[variable] for variable in variables)
    for band in bands:
        packing = attrs.Packing.from_attrs(band.attrs)
        nodata = attrs.Nodata.from_attrs(band.attrs)
        packed = packing is not None and (
            packing.scale_factor is not None or packing.add_offset is not None
        )
        filled = (
            nodata is not None
            and nodata.fill_value is not None
            and not np.isnan(nodata.fill_value)
        )
        if packed or filled:
            raise ValueError(
                f"{band.name!r} still holds stored values; call "
                ".gs.to_nan().gs.unpack() before computing reflectance features"
            )
    return tuple(band.astype(np.float32) for band in bands)


def feature_raster(
    source: xr.Dataset,
    field: xr.DataArray,
    *,
    name: str,
    reference: xr.DataArray,
) -> xr.Dataset:
    """Build one named feature raster from a derived field."""
    if not isinstance(name, str) or not name:
        raise ValueError("Feature name must be a non-empty string")
    result = field.rename(name)
    result.attrs = {}
    grid_mapping = reference.encoding.get(
        "grid_mapping", reference.attrs.get("grid_mapping")
    )
    if grid_mapping is not None:
        result.encoding["grid_mapping"] = grid_mapping
    return attrs.rebase(result.to_dataset(), source.gs.attrs.root)

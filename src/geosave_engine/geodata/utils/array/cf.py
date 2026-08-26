"""Convert between canonical Spatial arrays and CF raster datasets."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import orjson
import xarray as xr

from geosave_engine.geodata.extensions import Legend

from .canonical import SPATIAL_DIMS, ensure_crs, validate_spatial
from .nodata import same_nodata

CF_CONVENTIONS = "CF-1.8"

# Dataset variables cannot reuse a coordinate/dimension name this format owns.
_RESERVED_VAR_NAMES = frozenset({*SPATIAL_DIMS, "spatial_ref"})
_BAND_ORDER_ATTR = "_geosave_band_order"
# Attrs this adapter stamps on write and so must strip on read; a copy off disk is stale.
_ADAPTER_ATTRS = frozenset(
    {"Conventions", _BAND_ORDER_ATTR, "flag_values", "flag_meanings", "flag_colors"}
)
_CF_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PACKING_ATTRS = frozenset({"scale_factor", "add_offset"})
_SPATIAL_ALIASES: dict[str, str] = {
    "latitude": "y",
    "lat": "y",
    "longitude": "x",
    "lon": "x",
}
_STRUCTURAL_VAR_ATTRS = frozenset(
    {"_FillValue", "grid_mapping", "coordinates", "scale_factor", "add_offset"}
)


def da_to_cf(da: xr.DataArray) -> xr.Dataset:
    """Convert a canonical Spatial array to one CF variable per band.

    `da(band, y, x)` becomes `red(y, x)`, `nir(y, x)`, and so on.
    Band-aligned coordinates become each variable's own attributes.

    Args:
        da: Canonical `(band, y, x)` or `(time, band, y, x)` array.

    Returns:
        Dataset with the source's attrs dataset-global, `Conventions`
        set, nodata (`_FillValue`) re-stamped per variable, and a private
        attribute preserving band order across stores.

    Raises:
        ValueError: `da` violates the canonical Spatial representation or
            a band name conflicts with the CF layout.
    """
    ds = ensure_crs(_split_bands(validate_spatial(da)))
    return ds.assign_attrs(
        Conventions=CF_CONVENTIONS,
        **{_BAND_ORDER_ATTR: orjson.dumps([str(name) for name in ds.data_vars]).decode()},
    )


def cf_to_da(ds: xr.Dataset) -> xr.DataArray:
    """Convert a CF raster dataset into the canonical Spatial representation.

    Args:
        ds: Dataset containing one variable per band, or one already-banded
            variable. Spatial aliases such as `latitude`/`longitude` are
            accepted at this adapter seam.

    Returns:
        Canonical `(band, y, x)` or `(time, band, y, x)` DataArray. The
        attrs this adapter writes — `Conventions`, the band-order record
        and the `flag_*` render mirror — are dropped; every other attr is
        the source's own and passes through.

    Raises:
        ValueError: The dataset is empty, its variables disagree on dtype
            or nodata, its recorded order is invalid, or its result violates
            the canonical Spatial representation.
    """
    renames = {
        alias: canonical
        for alias, canonical in _SPATIAL_ALIASES.items()
        if alias in ds.dims
    }
    if renames:
        ds = ds.rename(renames)
    if not ds.data_vars:
        raise ValueError("CF raster dataset contains no data variables")

    encoded_order = ds.attrs.get(_BAND_ORDER_ATTR)
    if encoded_order is not None:
        try:
            order = orjson.loads(encoded_order)
        except (TypeError, orjson.JSONDecodeError) as error:
            raise ValueError(f"CF {_BAND_ORDER_ATTR} must be a JSON list of band names") from error
        names = list(map(str, ds.data_vars))
        if (
            not isinstance(order, list)
            or not all(isinstance(name, str) for name in order)
            or len(order) != len(names)
            or set(order) != set(names)
        ):
            raise ValueError(f"CF {_BAND_ORDER_ATTR} {order!r} does not match data variables {names}")
        ds = ds[order]

    ds = ds.copy(deep=False)
    ds.attrs = {key: value for key, value in ds.attrs.items() if key not in _ADAPTER_ATTRS}
    ds = ensure_crs(ds)

    only = next(iter(ds.data_vars.values())) if len(ds.data_vars) == 1 else None
    if only is not None and "band" in only.dims:
        da = only.copy(deep=False)
        da.attrs = {**ds.attrs, **only.attrs}
        ordered = tuple(dim for dim in SPATIAL_DIMS if dim in da.dims)
        da = da.transpose(*ordered)
    else:
        if any("band" in variable.dims for variable in ds.data_vars.values()):
            raise ValueError("CF variables cannot mix a band dimension with one-variable-per-band data")
        ds = _normalize_variable_layout(ds)
        _reject_packed_variables(ds)
        _require_shared_dtype(ds.data_vars.values())
        nodata = _shared_nodata(ds.data_vars.values())
        band_coords = _band_aligned_coords(ds)
        da = ds[list(ds.data_vars)].to_array(dim="band")
        da = da.transpose(*(dim for dim in SPATIAL_DIMS if dim in da.dims))
        da.attrs = dict(ds.attrs)
        if band_coords:
            da = da.assign_coords(band_coords)
        if nodata is not None:
            da = da.rio.write_nodata(nodata, inplace=True)

    da.encoding = {}
    return validate_spatial(da)


def _split_bands(da: xr.DataArray) -> xr.Dataset:
    """Give each band along `da`'s band axis its own Dataset variable.

    Args:
        da: Array already in GDAL form.

    Returns:
        Dataset carrying da's attrs dataset-global, nodata re-stamped on
        every variable. Band-aligned coords become each variable's own
        attributes.

    Raises:
        ValueError: a band name conflicts with the CF layout.
    """
    nodata = da.rio.nodata

    names = [str(band) for band in da.band.values]
    reserved = sorted(set(names) & _RESERVED_VAR_NAMES)
    if reserved:
        raise ValueError(
            f"Band names {reserved} are reserved by the CF layout — rename them before writing"
        )
    invalid = [name for name in names if _CF_NAME.fullmatch(name) is None]
    if invalid:
        raise ValueError(
            f"Band names {invalid} are not portable CF variable names — rename them to start with a letter "
            "and contain only letters, digits, or underscores"
        )

    aligned = [
        str(name)
        for name, coord in da.coords.items()
        if coord.dims == ("band",) and str(name) != "band"
    ]
    ds = da.drop_vars(aligned).to_dataset(dim="band")
    for position, name in enumerate(names):
        for key in aligned:
            ds[name].attrs[key] = _scalar(da.coords[key].values[position])

    # _FillValue is a per-variable attr in CF; da carried one dataset-wide, and it's re-stamped below anyway
    ds.attrs = {key: value for key, value in ds.attrs.items() if key != "_FillValue"}

    # to_dataset drops da's nodata off every variable it split out, both branches above
    if nodata is not None:
        for name in ds.data_vars:
            ds[name] = ds[name].rio.write_nodata(nodata, inplace=True)
    return ds


def _normalize_variable_layout(ds: xr.Dataset) -> xr.Dataset:
    """Require one shared CF raster layout and put its dimensions in canonical order."""
    layouts = {frozenset(variable.dims) for variable in ds.data_vars.values()}
    if len(layouts) != 1:
        described = {str(name): tuple(variable.dims) for name, variable in ds.data_vars.items()}
        raise ValueError(f"CF bands need identical dimensions; got {described}")

    layout = {str(dim) for dim in next(iter(layouts))}
    order: tuple[str, ...]
    if layout == {"y", "x"}:
        order = ("y", "x")
    elif layout == {"time", "y", "x"}:
        order = ("time", "y", "x")
    else:
        raise ValueError(
            "CF band variables must use dimensions ('y', 'x') or ('time', 'y', 'x'); "
            f"got {sorted(layout)}"
        )

    normalized = ds.copy(deep=False)
    for name, variable in ds.data_vars.items():
        normalized[name] = variable.transpose(*order)
    return normalized


def _reject_packed_variables(ds: xr.Dataset) -> None:
    """Reject packed CF values that cannot share one canonical dtype safely."""
    packed = {
        str(name): sorted(
            key for key in _PACKING_ATTRS if key in variable.attrs or key in variable.encoding
        )
        for name, variable in ds.data_vars.items()
    }
    packed = {name: keys for name, keys in packed.items() if keys}
    if packed:
        raise ValueError(
            f"CF packed values are unsupported ({packed}); decode scale/offset with Xarray before attaching pixels"
        )


def _band_aligned_coords(ds: xr.Dataset) -> dict[str, tuple[str, list[Any]]]:
    """Collect shared per-variable attributes as coordinates along `band`."""
    names = list(ds.data_vars)
    keys = {key for var in ds.data_vars.values() for key in var.attrs} - _STRUCTURAL_VAR_ATTRS
    coords: dict[str, tuple[str, list[Any]]] = {}
    for key in sorted(keys):
        values = [ds[name].attrs.get(key) for name in names]
        if all(value is not None for value in values):
            coords[key] = ("band", values)
    return coords


def _require_shared_dtype(data_vars: Iterable[xr.DataArray]) -> None:
    """Require every CF variable to use one pixel dtype."""
    dtypes = [variable.dtype for variable in data_vars]
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise ValueError(f"CF bands disagree on dtype {dtypes}; cast them explicitly before stacking")


def _shared_nodata(data_vars: Iterable[xr.DataArray]) -> float | int | None:
    """Return the nodata value every CF variable declares identically."""
    values = [variable.rio.nodata for variable in data_vars]
    first = values[0]
    if not all(same_nodata(value, first) for value in values[1:]):
        raise ValueError(f"CF bands disagree on nodata {values}; set one explicit value before stacking")
    return first


def _scalar(value: Any) -> Any:
    """A numpy scalar as its plain Python equivalent; anything else untouched.

    Args:
        value: One element out of a coord's `.values`.

    Returns:
        `value.item()` for a numpy scalar, else `value` as given — attrs
        holding numpy types don't serialize cleanly to JSON or NetCDF.
    """
    return value.item() if isinstance(value, np.generic) else value


def cf_flag_attrs(legend: Legend | None) -> dict[str, Any]:
    """Build the CF flag_* trio out of a legend extension (CF §3.5 + community extension).

    Write-only mirror so external CF/GDAL tools can auto-style a label
    raster. Canonical arrays take this metadata off their own header.

    Args:
        legend: Extension holding `class_map`/`color_map`. None gives `{}`.

    Returns:
        {
            "flag_values": [0, 1, 2],
            "flag_meanings": "bg water veg",
            "flag_colors": ["#000000", "#0000FF", "#00FF00"],
        }
        `flag_meanings` is one space-separated string, spaces inside a
        name replaced by `_`. `flag_colors` omitted when `color_map`
        doesn't cover every class. `{}` when neither map is set.
    """
    if legend is None:
        return {}

    # class_map's keys are the class list when it's set; color_map's own keys stand in otherwise
    class_map, color_map = legend.class_map, legend.color_map
    classes = sorted(class_map if class_map is not None else color_map or {})
    if not classes:
        return {}

    names = [(class_map.get(value, str(value)) if class_map else str(value)) for value in classes]
    attrs: dict[str, Any] = {
        "flag_values": classes,
        "flag_meanings": " ".join(name.replace(" ", "_") for name in names),
    }
    if color_map is not None and all(value in color_map for value in classes):
        attrs["flag_colors"] = [_hex_color(color_map[value]) for value in classes]
    return attrs


def _hex_color(color: tuple[int, int, int] | str) -> str:
    """An RGB triple or hex string as uppercase `"#RRGGBB"`.

    Args:
        color: `(r, g, b)` 0-255 triple, or a `"#rrggbb"` string.

    Returns:
        Uppercase `"#RRGGBB"`.
    """
    if isinstance(color, str):
        return color.upper()
    red, green, blue = color
    return f"#{red:02X}{green:02X}{blue:02X}"

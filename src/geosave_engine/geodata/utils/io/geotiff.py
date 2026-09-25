"""Write a Dataset as a COG or a plain GeoTIFF, one band per variable.

`write_header` puts each attr where GDAL reads it: a band property where GDAL
keeps one, a metadata tag otherwise. rioxarray is left the pixels and the grid.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
from odc.geo.geobox import GeoBox
from rasterio.enums import ColorInterp

# rasterio ships this as a compiled module, which no type checker can resolve.
from rasterio.shutil import copy  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]

import orjson
import xarray as xr
from xarray.conventions import encode_cf_variable

from geosave_engine.geodata.attrs import (
    AttrsHeader,
    GDALVariable,
    GeoTIFFTags,
    Nodata,
    create_header,
    rebase,
)
from geosave_engine.geodata.attrs.model import attrs_equal

from geosave_engine.geodata.core.profile import TIME_COORDINATE
from geosave_engine.utils.colorize import parse_color

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from geosave_engine.geodata import Dataset

type CreationOptionValue = str | int | float | bool

_FILE_SUFFIXES = (".tif", ".tiff")
_PACKING_KEYS = ("scale_factor", "add_offset")


def _repack(ds: Dataset) -> Dataset:
    """Apply each variable's CF packing to its values, moving it into attrs.

    rioxarray's Dataset writer declares a band's scale and offset without
    applying them, so decoded pixels would be stored physical and then read
    back scaled a second time. Packing them here keeps values and tags agreed.

    Args:
        ds: Cube whose variables may carry CF packing in their encoding, as
            every raster read with ``mask_and_scale=True`` does.

    Returns:
        The same cube with stored values, each variable's packing now in its
        attrs where the header reads it. A variable carrying none is untouched.
    """
    packed = ds.copy()
    for name, variable in ds.data_vars.items():
        if not any(key in variable.encoding for key in _PACKING_KEYS):
            continue
        encoded = encode_cf_variable(variable.variable)
        # The CF encoder names the grid mapping, which rioxarray writes itself.
        attrs = {
            key: value for key, value in encoded.attrs.items() if key != "grid_mapping"
        }
        stored = xr.DataArray(
            encoded.data, dims=variable.dims, coords=variable.coords, name=name
        )
        stored.attrs = {**variable.attrs, **attrs}
        stored.encoding = {
            key: value
            for key, value in variable.encoding.items()
            if key not in (*_PACKING_KEYS, "_FillValue")
        }
        packed[name] = stored
    return packed


class GeoTIFFWriteOptions(TypedDict, total=False):
    """GDAL creation options both TIFF drivers support."""

    compress: str
    predictor: int
    zlevel: int
    num_threads: int | Literal["ALL_CPUS"]
    bigtiff: Literal["YES", "NO", "IF_NEEDED", "IF_SAFER"]
    interleave: Literal["PIXEL", "BAND"]
    sparse_ok: bool
    creation_options: dict[str, CreationOptionValue]


class COGWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options GDAL's COG driver supports."""

    blocksize: int
    overview_resampling: str
    overview_count: int


class GTiffWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options GDAL's GTiff driver supports."""

    tiled: bool
    blockxsize: int
    blockysize: int
    copy_src_overviews: bool
    photometric: str
    nbits: int


def write_cog(
    ds: Dataset,
    path: str | PathLike[str],
    *,
    map_scale: float | None = None,
    overwrite: bool = False,
    **options: Unpack[COGWriteOptions],
) -> Path:
    """Write one Dataset as a Cloud Optimized GeoTIFF, a band per variable.

    Each variable names itself in its band's metadata, and a scalar `time`
    coordinate becomes ``TIFFTAG_DATETIME``.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
        path: Output path ending in ``.tif`` or ``.tiff``.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace an existing file when true.
        **options: COG creation options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube carries a `time` dimension
            rather than a scalar coordinate, a scalar `time` carries sub-second
            precision the tag cannot hold, or the variables declare more than
            one fill value.

    Examples:
        >>> write_cog(ds, "scene.tif", compress="DEFLATE")
        PosixPath('scene.tif')
    """
    return _write(
        ds,
        path,
        "COG",
        map_scale=map_scale,
        overwrite=overwrite,
        options=options,
    )


def write_gtiff(
    ds: Dataset,
    path: str | PathLike[str],
    *,
    map_scale: float | None = None,
    overwrite: bool = False,
    **options: Unpack[GTiffWriteOptions],
) -> Path:
    """Write one Dataset as a plain GeoTIFF, a band per variable.

    Reach for `write_cog` unless a consumer needs a striped or otherwise
    non-COG file. Coordinates map to tags exactly as `write_cog` maps them.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
        path: Output path ending in ``.tif`` or ``.tiff``.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace an existing file when true.
        **options: GTiff creation options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube carries a `time` dimension
            rather than a scalar coordinate, a scalar `time` carries sub-second
            precision the tag cannot hold, or the variables declare more than
            one fill value.

    Examples:
        >>> write_gtiff(ds, "scene.tif", tiled=True, blockxsize=512)
        PosixPath('scene.tif')
    """
    return _write(
        ds,
        path,
        "GTiff",
        map_scale=map_scale,
        overwrite=overwrite,
        options=options,
    )


def _write(
    ds: Dataset,
    path: str | PathLike[str],
    driver: Literal["COG", "GTiff"],
    *,
    map_scale: float | None,
    overwrite: bool,
    options: Mapping[str, Any],
) -> Path:
    """Encode one cube through the GDAL driver `driver` names.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
        path: Output path ending in ``.tif`` or ``.tiff``.
        driver: GDAL driver creating the file.
        map_scale: Map denominator used to write pixels per centimetre.
        overwrite: Replace an existing file when true.
        options: Creation options the driver supports.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix or the `time` axis is invalid, or the variables
            declare more than one fill value.
    """
    target = Path(path)
    if target.suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {target.name!r} must end in one of {list(_FILE_SUFFIXES)}"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")

    # A GCP grid writes back as no geotransform at all, leaving the file unplaced.
    grid = ds.odc.geobox
    if grid is not None and not isinstance(grid, GeoBox):
        raise ValueError(
            f"this raster is placed by {type(grid).__name__}, which a GeoTIFF "
            f"records as ground control points GeoSave does not write; reproject "
            f"it onto a regular grid with warp.reproject first"
        )

    ds = _repack(ds)
    header = create_header(ds)

    cube = ds
    # One file holds one instant, however the axis spells it.
    if TIME_COORDINATE in cube.dims:
        if cube.sizes[TIME_COORDINATE] > 1:
            raise ValueError(
                f"one GeoTIFF holds one instant, but this cube spans "
                f"{cube.sizes[TIME_COORDINATE]}; select one with .isel(time=0) "
                f"or resample the time axis away first"
            )
        cube = cube.squeeze(TIME_COORDINATE)

    tags = GeoTIFFTags.from_xarray(cube, map_scale=map_scale)
    if TIME_COORDINATE in cube.coords:
        cube = cube.drop_vars(TIME_COORDINATE)

    names = [str(name) for name in cube.data_vars]
    bands = [header.data_vars[name] for name in names]

    # GDAL writes one fill value for the whole file, so the bands must share one.
    fill_values = [(band.get(Nodata) or Nodata()).fill_value for band in bands]
    if not all(
        attrs_equal(fill_value, fill_values[0])
        for fill_value in fill_values
    ):
        raise ValueError(
            f"one GeoTIFF holds one fill value, but these bands declare "
            f"{dict(zip(names, fill_values, strict=True))}; write one across "
            f"them with raster.gs.write_nodata(...) first"
        )

    written = rebase(cube, tags)
    # A band names itself in its own metadata; GDAL owns the interpretation.
    for name in names:
        written = rebase(written, GDALVariable(variable_name=name), target=name)

    # A header names its variables; the cube's order decides their bands.
    snapshot = create_header(written)
    header = AttrsHeader(
        root=snapshot.root,
        data_vars={name: snapshot.data_vars[name] for name in names},
    )

    # rioxarray turns any attr it is handed into a tag, so it is handed none.
    cube = cube.copy(deep=False)
    cube.attrs = {}
    for name in names:
        cube[name].attrs = {}

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as workspace:
        # A COG is copied from a source raster, which its own options create.
        if driver == "COG":
            source = Path(workspace) / target.name
            source_options: Mapping[str, Any] = {"tiled": True}
        else:
            source = target
            source_options = options

        cube.rio.to_raster(
            source,
            driver="GTiff",
            # rioxarray computes a chunked cube whole unless told to walk its windows.
            windowed=bool(cube.chunks),
            **source_options,
        )
        # GDAL honours metadata on an open dataset, not as creation options.
        with rasterio.open(source, "r+") as dst:
            write_header(dst, header)

        if driver == "COG":
            copy(source, target, driver="COG", **options)
    return target


def write_header(dst: rasterio.io.DatasetWriter, header: AttrsHeader) -> None:
    """Write every attr onto an open GDAL raster, in the slot GDAL reads it from.

    GDAL keeps the description, unit, scale, offset, fill value, interpretation
    and palette as band properties of its own, where `gdalinfo` and QGIS look;
    every other attr is a plain metadata item, and none is stated twice.

    Args:
        dst: Raster open for update, its bands in the header's data-variable
            order.
        header: Attrs to write.

    Raises:
        ValueError: The header does not describe every band of `dst`.

    Examples:
        >>> with rasterio.open(path, "r+") as dst:
        ...     write_header(dst, create_header(ds))
    """
    bands = list(header.data_vars.values())
    if len(bands) != dst.count:
        raise ValueError(
            f"{dst.name} holds {dst.count} bands but the header describes "
            f"{len(bands)}; pass one namespace per band, in band order"
        )

    dst.update_tags(**_as_text(header.root.to_attrs()))

    scales: list[float] = []
    offsets: list[float] = []
    interpretations: list[ColorInterp] = []
    for index, band in enumerate(bands, start=1):
        tags = band.to_attrs()

        # Each of these leaves the tags for a band property GDAL keeps it in.
        description = tags.pop("long_name", None)
        unit = tags.pop("units", None)
        scale = tags.pop("scale_factor", None)
        offset = tags.pop("add_offset", None)
        palette = tags.pop("color_map", None)
        interpretation = tags.pop("colorinterp", None)
        tags.pop("_FillValue", None)
        tags.pop("nodata", None)

        dst.update_tags(index, **_as_text(tags))

        if description is not None:
            dst.set_band_description(index, description)
        if unit is not None:
            dst.set_band_unit(index, unit)

        scales.append(1.0 if scale is None else scale)
        offsets.append(0.0 if offset is None else offset)
        interpretations.append(
            ColorInterp[interpretation or ColorInterp.undefined.name]
        )

        if palette is not None:
            dst.write_colormap(
                index,
                {
                    int(value): (*parse_color(colour), 255)
                    for value, colour in palette.items()
                },
            )

    if any(scale != 1.0 for scale in scales):
        dst.scales = scales
    if any(offset != 0.0 for offset in offsets):
        dst.offsets = offsets
    # Assigning none of them would clear what GDAL inferred for itself.
    if any(band is not ColorInterp.undefined for band in interpretations):
        dst.colorinterp = interpretations

    shared = (bands[0].get(Nodata) or Nodata()).fill_value
    if shared is not None:
        dst.nodata = shared


def _as_text(attrs: Mapping[str, Any]) -> dict[str, str]:
    """Spell attrs as the text a GDAL metadata tag holds.

    Args:
        attrs: Flat attrs to write as tags.

    Returns:
        {
            "<attr key>": its value as text,
        }
        A collection is spelled as JSON, which reads back as itself.
    """
    return {
        key: value if isinstance(value, str) else orjson.dumps(value).decode()
        for key, value in attrs.items()
    }

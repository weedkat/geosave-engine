from __future__ import annotations

import dataclasses
import json
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray/Dataset
from odc.geo.xr import xr_coords
import numpy as np
import xarray as xr
import torch

from pystac import Item, ItemCollection
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, cast, Literal, Mapping
from rioxarray.merge import merge_datasets
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from pyproj import CRS, Transformer

from geosave_engine.utils.geolocator import Place
from geosave_engine.utils.crs import calculate_crs, validate_coordinate
from geosave_engine.utils.datetime import DEFAULT_DATE_FORMAT, DEFAULT_DATE_PATTERN, date_from_path, parse_datetime


def _datetime_from_attrs_or_stem(
    path: Path,
    attrs: Mapping[str, Any],
    date_format: str = DEFAULT_DATE_FORMAT,
    date_pattern: str = DEFAULT_DATE_PATTERN,
) -> dt:
    """Read datetime from attrs first, then fall back to the filename stem."""
    raw_datetime = attrs.get("datetime")
    if raw_datetime is None:
        metadata = attrs.get("metadata")
        if metadata:
            try:
                tag = json.loads(metadata)
            except (TypeError, ValueError, json.JSONDecodeError):
                tag = {}
            raw_datetime = tag.get("datetime")

    if raw_datetime is not None:
        return parse_datetime(raw_datetime)

    try:
        return date_from_path(path.stem, date_format=date_format, date_pattern=date_pattern)
    except ValueError as exc:
        raise ValueError(
            f"Could not determine datetime for {path}: missing attrs and no date in stem "
            f"matching pattern {date_pattern!r}"
        ) from exc


@dataclass(frozen=True)
class GeoTile:
    """Geospatial tile with a geobox, anchor datetime, and optional pixel data.

    data is an xr.DataArray with dims (band, y, x) or (time, band, y, x).
    May be lazy or fully in memory; None means a header-only tile.
    """

    geobox: GeoBox
    datetime: dt
    data: xr.DataArray | None = field(default=None, compare=False)
    stac: list[Item] = field(default_factory=list, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    polygon: Geometry | None = field(default=None, compare=False)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def bands(self) -> tuple[str, ...]:
        """Band names from loaded data. Empty when data is None."""
        if self.data is None:
            return ()
        return tuple(str(b) for b in self.data.coords["band"].values)

    @property
    def times(self) -> tuple[dt, ...]:
        """Observation datetimes from loaded data. Empty when data has no time."""
        if self.data is None or "time" not in self.data.dims:
            return ()
        return tuple(
            dt.fromisoformat(str(t.astype("datetime64[s]")))
            for t in self.data.time.values
        )

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    @property
    def has_time(self) -> bool:
        """True if data has a time dimension."""
        return self.data is not None and "time" in self.data.dims

    @property
    def resolution(self) -> float:
        return self.geobox.affine.a

    @property
    def affine(self) -> Affine:
        return self.geobox.affine

    @property
    def crs(self) -> str | None:
        crs = self.geobox.crs
        if crs is None:
            return None
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else None

    @property
    def width(self) -> int:
        return self.geobox.width

    @property
    def height(self) -> int:
        return self.geobox.height

    @property
    def area_m2(self) -> float:
        """Polygon area if stored, else pixel-grid area. Both in native CRS units² (m² for projected CRS)."""
        if self.polygon is not None:
            return self.polygon.geom.area
        return self.width * self.height * (self.resolution ** 2)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box in native CRS: (minx, miny, maxx, maxy)."""
        return self.geobox.boundingbox.bbox

    @property
    def wgs84_bbox(self) -> tuple[float, float, float, float]:
        """Bounding box in WGS84: (minlon, minlat, maxlon, maxlat)."""
        return self.geobox.geographic_extent.boundingbox.bbox

    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid in WGS84: (lon, lat)."""
        left, bottom, right, top = self.geobox.geographic_extent.boundingbox.bbox
        return ((left + right) / 2, (bottom + top) / 2)

    @property
    def bbox_polygon(self) -> Geometry:
        return self.geobox.boundingbox.polygon

    @property
    def geojson(self) -> dict:
        """WGS84 footprint as a GeoJSON Polygon dict."""
        if self.polygon is not None:
            return self.polygon.to_crs("EPSG:4326").geojson()
        return self.bbox_polygon.geojson()

    @property
    def location(self) -> dict[str, Any]:
        """Reverse-geocode centroid. Returns {} on failure."""
        lon, lat = self.centroid
        place = Place.from_coordinate(lat, lon)
        return place.to_dict() if place is not None else {}

    # ------------------------------------------------------------------
    # Data manipulation
    # ------------------------------------------------------------------

    def with_data(self, data: xr.DataArray) -> "GeoTile":
        """Return new GeoTile with given DataArray as pixel data.

        Args:
            data: Band values shaped (band, y, x) or (time, band, y, x) with a "band" coordinate.

        Raises:
            TypeError: If data is not an xr.DataArray.
        """
        if not isinstance(data, xr.DataArray):
            raise TypeError(f"with_data expects an xr.DataArray, got {type(data).__name__}")
        return dataclasses.replace(self, data=data)

    def with_np(
        self,
        array: np.ndarray,
        names: list[str],
        times: list[dt] | None = None,
    ) -> "GeoTile":
        """Build DataArray from numpy array on this tile's geobox and attach it.

        Accepts (y, x), (band, y, x), or (time, band, y, x). Spatial coords from geobox.

        Args:
            array: Pixel array; last two axes are (y, x).
            names: Band names in order.
            times: Observation datetimes; required for 4D array.
        """
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        base_coords: dict[Any, Any] = dict(xr_coords(self.geobox))
        if arr.ndim == 3:
            if len(names) != arr.shape[0]:
                raise ValueError(f"Expected {arr.shape[0]} names, got {len(names)}")
            da = xr.DataArray(arr, dims=("band", "y", "x"), coords={**base_coords, "band": names})
        elif arr.ndim == 4:
            if len(names) != arr.shape[1]:
                raise ValueError(f"Expected {arr.shape[1]} names, got {len(names)}")
            if times is None or len(times) != arr.shape[0]:
                got = 0 if times is None else len(times)
                raise ValueError(f"Expected {arr.shape[0]} times, got {got}")
            time_coord = [np.datetime64(t, "ns") for t in times]
            da = xr.DataArray(
                arr, dims=("time", "band", "y", "x"),
                coords={**base_coords, "band": names, "time": time_coord},
            )
        else:
            raise ValueError(f"with_np expects a 2-4D array, got {arr.ndim}D")
        return self.with_data(da)

    def with_stac(self, items: list[Item]) -> "GeoTile":
        """Append pystac Items as provenance, de-duplicated by id."""
        seen = {i.id for i in self.stac}
        merged = [*self.stac, *(i for i in items if i.id not in seen)]
        return dataclasses.replace(self, stac=merged)

    def with_metadata(self, extra: Mapping[str, Any], replace: bool = False) -> "GeoTile":
        """Merge key-value pairs into metadata field.

        Args:
            extra: Key-value pairs to merge.
            replace: If True, overwrite existing keys instead of raising.

        Raises:
            ValueError: If any key in extra already exists and replace is False.
        """
        if not replace:
            clash = set(extra) & set(self.metadata)
            if clash:
                raise ValueError(
                    f"metadata keys already present: {sorted(clash)}; pass replace=True to overwrite"
                )
        return dataclasses.replace(self, metadata={**self.metadata, **extra})

    def with_geobox(self, geobox: GeoBox) -> "GeoTile":
        """Return new GeoTile rebased onto geobox, sharing data reference.

        Pure geometry — no pixels are read or copied.
        """
        return dataclasses.replace(self, geobox=geobox)

    # ------------------------------------------------------------------
    # Tensor loading
    # ------------------------------------------------------------------

    def to_tensor(self, bands: list[str] | None = None, squeeze: bool = False) -> Any:
        """Render data as a single torch.Tensor with bands stacked.

        Clips to self.bbox before reading; output shape (band, y, x) or (time, band, y, x).

        Args:
            bands: Variable names to select, in order. None uses all bands.
        """
        result = torch.from_numpy(self.to_numpy(bands=bands))
        if squeeze and result.ndim == 3 and result.shape[0] == 1:
            result = result.squeeze(0)
        return result

    def to_numpy(self, bands: list[str] | None = None) -> np.ndarray:
        """Render data as a contiguous NumPy array with bands stacked.

        Clips to self.bbox before reading; output shape (band, y, x) or (time, band, y, x).

        Args:
            bands: Band names to select, in order. None uses all bands.

        Raises:
            ValueError: If tile has no data.
        """
        if self.data is None:
            raise ValueError("GeoTile has no data — load data first")
        da = self.data.rio.clip_box(*self.bbox)
        if bands is not None:
            da = da.sel(band=bands)
        if "time" in da.dims:
            da = da.transpose("time", "band", "y", "x")
        else:
            da = da.transpose("band", "y", "x")
        return np.ascontiguousarray(da.values)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_geotiff(
        cls,
        path: str | Path,
        load_data: bool = False,
        date_format: str = DEFAULT_DATE_FORMAT,
        date_pattern: str = DEFAULT_DATE_PATTERN,
        bands: tuple[str, ...] | None = None,
    ) -> "GeoTile":
        """Create GeoTile from a single GeoTIFF or COG file.

        Opened lazily by default; pixels read only on to_tensor or load_data=True.

        Args:
            path: Path to GeoTIFF/COG file.
            load_data: Materialise all pixels into memory; default lazy.
            date_format: strftime format for parsing date from filename stem.
            date_pattern: Regex pattern for extracting date from filename stem.
            bands: Band variable names to select; None keeps all.
        """
        p = Path(path)
        opened = rioxarray.open_rasterio(p, chunks=True, band_as_variable=True)
        if not isinstance(opened, xr.Dataset):
            raise TypeError(f"Expected a Dataset from {p}, got {type(opened).__name__}")
        data: xr.Dataset = opened

        raw = data.attrs.get("metadata")
        tag: dict[str, Any] = json.loads(raw) if raw else {}
        # band_as_variable names bands band_1..band_N; restore real names from the tag
        band_names: list[str] | None = tag.pop("bands", None)
        if band_names and len(band_names) == len(data.data_vars):
            data = data.rename(dict(zip(list(data.data_vars), band_names)))

        poly_geojson = tag.pop("polygon_geojson", None)
        poly_crs = tag.pop("polygon_crs", None)
        stored_polygon: Geometry | None = None
        if poly_geojson and poly_crs:
            geojson_dict = poly_geojson if isinstance(poly_geojson, dict) else json.loads(poly_geojson)
            stored_polygon = Geometry(geojson_dict, crs=poly_crs)

        if bands:
            data = cast(xr.Dataset, data[list(bands)])
        if load_data:
            data = data.load()

        anchor_dt = _datetime_from_attrs_or_stem(
            p, data.attrs, date_format=date_format, date_pattern=date_pattern
        )
        geobox = data.odc.geobox
        da = data.to_array(dim="band").transpose("band", "y", "x")
        return cls(
            geobox=geobox,
            datetime=anchor_dt,
            data=da,
            stac=_read_stac(p),
            metadata=tag,
            polygon=stored_polygon,
        )

    @classmethod
    def from_zarr(
        cls, path: str | Path, 
        load_data: bool = False,
        date_format: str = DEFAULT_DATE_FORMAT,
        date_pattern: str = DEFAULT_DATE_PATTERN,
    ) -> "GeoTile":
        """Create GeoTile from a Zarr store written by to_zarr.

        Opened lazily by default. Geobox, datetime, and metadata restored from store attrs.

        Args:
            path: Path to Zarr store.
            load_data: Materialise all pixels into memory; default lazy.
            date_format: strftime format for parsing date from filename stem.
            date_pattern: Regex pattern for extracting date from filename stem.
        """
        path = Path(path)
        ds = xr.open_zarr(path)
        # zarr restores the CRS grid-mapping as a data variable; demote it to a coord
        grid_mappings = {
            var.attrs["grid_mapping"]
            for var in ds.data_vars.values()
            if "grid_mapping" in var.attrs
        } & set(ds.data_vars)
        if grid_mappings:
            ds = ds.set_coords(grid_mappings)
        geobox = ds.odc.geobox
        anchor_dt = _datetime_from_attrs_or_stem(
            path, ds.attrs, date_format=date_format, date_pattern=date_pattern
        )
        metadata = json.loads(ds.attrs.get("metadata", "{}"))
        poly_geojson_raw = ds.attrs.get("polygon_geojson")
        poly_crs = ds.attrs.get("polygon_crs")
        stored_polygon: Geometry | None = None
        if poly_geojson_raw and poly_crs:
            geojson_dict = json.loads(poly_geojson_raw) if isinstance(poly_geojson_raw, str) else poly_geojson_raw
            stored_polygon = Geometry(geojson_dict, crs=poly_crs)
        da = ds.to_array(dim="band")
        if "time" in da.dims:
            da = da.transpose("time", "band", "y", "x")
        else:
            da = da.transpose("band", "y", "x")
        if load_data:
            da = da.load()
        return cls(
            geobox=geobox,
            datetime=anchor_dt,
            data=da,
            stac=_read_stac(path),
            metadata=metadata,
            polygon=stored_polygon,
        )

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        datetime: dt | str,
        crs: str = "EPSG:4326",
        resolution: float = 10.0,
    ) -> "GeoTile":
        """Create a GeoTile from a bounding box and resolution.

        Args:
            bbox: (minx, miny, maxx, maxy) in the given CRS.
            crs: Coordinate reference system string (e.g. "EPSG:4326").
            resolution: Pixel size in CRS units.
            datetime: Anchor datetime for this tile.
        """
        return cls(
            geobox=GeoBox.from_bbox(bbox, crs=crs, resolution=resolution, anchor="edge"),
            datetime=parse_datetime(datetime),
        )

    @classmethod
    def from_coordinate(
        cls,
        latitude: float,
        longitude: float,
        *,
        datetime: dt | str,
        size_m: float | tuple[float, float],
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> "GeoTile":
        """Create a GeoTile centered on a WGS84 coordinate.

        Args:
            latitude: Center latitude in WGS84 degrees.
            longitude: Center longitude in WGS84 degrees.
            size_m: Tile size in meters. Single number = square; ``(w, h)`` = rectangle.
            resolution: Pixel size in meters.
            datetime: Anchor datetime for this tile.
            crs: Target projected CRS. Defaults to the local UTM/UPS zone.
        """
        validate_coordinate(latitude, longitude)
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        width_m, height_m = (
            (size_m, size_m) if isinstance(size_m, (int, float)) else size_m
        )
        if width_m <= 0 or height_m <= 0:
            raise ValueError(f"Tile size must be positive, got {size_m}")
        target_crs = (
            CRS.from_user_input(crs) if crs is not None else calculate_crs(latitude, longitude)
        )
        if target_crs.is_geographic:
            raise ValueError(
                "from_coordinate requires a projected CRS; omit crs to use local UTM/UPS"
            )
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        cx, cy = transformer.transform(longitude, latitude)
        bbox = (cx - width_m / 2, cy - height_m / 2, cx + width_m / 2, cy + height_m / 2)
        return cls(
            geobox=GeoBox.from_bbox(
                bbox, crs=target_crs.to_string(), resolution=resolution, tight=True
            ),
            datetime=parse_datetime(datetime),
        )

    @classmethod
    def from_polygon(
        cls,
        polygon: dict,
        datetime: dt | str,
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> "GeoTile":
        """Create a GeoTile from a GeoJSON polygon geometry dict.

        Coordinates must be in WGS84 lon/lat. Resolution is in meters;
        the tile is projected into local UTM/UPS unless ``crs`` is specified.

        Args:
            polygon: GeoJSON geometry dict with WGS84 lon/lat coordinates.
            resolution: Pixel size in meters.
            datetime: Anchor datetime for this tile.
            crs: Target projected CRS. Defaults to local UTM/UPS zone.
        """
        geom = Geometry(polygon, crs="EPSG:4326")
        bbox = geom.boundingbox
        if crs is None:
            lon = (bbox.left + bbox.right) / 2
            lat = (bbox.bottom + bbox.top) / 2
            target_crs = calculate_crs(lat, lon)
        else:
            target_crs = CRS.from_user_input(crs)
        if target_crs.is_geographic:
            raise ValueError(
                "from_polygon requires a projected CRS; omit crs to use local UTM/UPS"
            )
        projected_geom = geom.to_crs(target_crs)
        return cls(
            geobox=GeoBox.from_geopolygon(projected_geom, resolution=resolution, anchor="edge"),
            datetime=parse_datetime(datetime),
            polygon=projected_geom,
        )

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        datetime: dt | str,
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> "list[GeoTile]":
        """Create one GeoTile per feature/geometry in a GeoJSON file.

        Args:
            path: Path to a GeoJSON FeatureCollection, Feature, or raw geometry.
            resolution: Pixel size in meters.
            datetime: Anchor datetime applied to all features.
            crs: Target projected CRS. Defaults to local UTM/UPS per feature centroid.
        """
        parsed_dt = parse_datetime(datetime) if isinstance(datetime, str) else datetime
        with open(path) as f:
            geojson = json.load(f)
        if geojson.get("type") == "FeatureCollection":
            geometries = [feat["geometry"] for feat in geojson["features"]]
        elif geojson.get("type") == "Feature":
            geometries = [geojson["geometry"]]
        else:
            geometries = [geojson]
        return [
            cls.from_polygon(geom, resolution=resolution, datetime=parsed_dt, crs=crs)
            for geom in geometries
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_geotiff(self, path: str | Path, save_stac: bool = False) -> Path:
        """Write single-step tile to GeoTIFF (one band per variable).

        Args:
            path: Output .tif path.
            save_stac: Write STAC provenance as <stem>.stac.json sidecar.

        Returns:
            The written path.

        Raises:
            ValueError: If tile has a time dimension or no data.
        """
        return self._write(path, driver="GTiff", save_stac=save_stac)

    def to_cog(self, path: str | Path, save_stac: bool = False) -> Path:
        """Write single-step tile to Cloud-Optimized GeoTIFF.

        Args:
            path: Output .tif path.
            save_stac: Write STAC provenance as <stem>.stac.json sidecar.

        Returns:
            The written path.

        Raises:
            ValueError: If tile has a time dimension or no data.
        """
        return self._write(path, driver="COG", save_stac=save_stac)

    def to_zarr(self, path: str | Path, save_stac: bool = False) -> Path:
        """Write tile data including any time dimension to a Zarr store.

        Anchor datetime and metadata stored as store attributes.

        Args:
            path: Output Zarr store path.
            save_stac: Write STAC provenance as <stem>.stac.json sidecar.

        Returns:
            The written store path.

        Raises:
            ValueError: If tile has no data.
        """
        if self.data is None:
            raise ValueError("GeoTile has no data to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        attrs: dict[str, Any] = {
            "datetime": self.datetime.isoformat(),
            "metadata": json.dumps(self.metadata),
        }
        if self.polygon is not None:
            attrs["polygon_geojson"] = json.dumps(self.polygon.geojson())
            attrs["polygon_crs"] = str(self.polygon.crs)
        ds = self.data.to_dataset(dim="band").assign_attrs(**attrs)
        ds.to_zarr(path, mode="w")
        if save_stac:
            _write_stac(self.stac, path)
        return path

    def _write(self, path: str | Path, driver: str, save_stac: bool = False) -> Path:
        if self.data is None:
            raise ValueError("GeoTile has no data to save")
        if self.has_time:
            raise ValueError(
                "Cannot write a time-series tile to GeoTIFF; use to_zarr() instead"
            )
        path = Path(path)
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Expected .tif path, got: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tag: dict[str, Any] = {**self.metadata, "bands": [str(b) for b in self.data.coords["band"].values]}
        if self.polygon is not None:
            tag["polygon_geojson"] = self.polygon.geojson()
            tag["polygon_crs"] = str(self.polygon.crs)
        self.data.rio.to_raster(
            path,
            driver=driver,
            tags={"metadata": json.dumps(tag), "datetime": self.datetime.isoformat()},
        )
        if save_stac:
            _write_stac(self.stac, path)
        return path


# ----------------------------------------------------------------------
# Tile operations
# ----------------------------------------------------------------------

def remap(tile: GeoTile, mapping: Mapping[int, int]) -> GeoTile:
    """Return a new GeoTile with label values remapped per ``mapping``."""
    if tile.data is None:
        raise ValueError("Cannot remap a GeoTile without data")
    remapped = tile.data
    for src_val, dst_val in mapping.items():
        remapped = remapped.where(remapped != src_val, other=dst_val)
    return tile.with_data(remapped)


def align(*tiles: GeoTile) -> tuple[GeoTile, ...]:
    """Narrow each tile's geobox to their common intersection.

    Pure geometry — data is shared untouched. Tiles must share CRS, resolution, and pixel grid.

    Raises:
        ValueError: If fewer than 2 tiles, CRS/resolution mismatch, no overlap, or misaligned grid.
    """
    if len(tiles) < 2:
        raise ValueError("align() requires at least 2 tiles")
    crss = {t.crs for t in tiles}
    if len(crss) > 1:
        raise ValueError(f"align() requires one CRS, got: {crss}")
    resolutions = {round(t.resolution, 6) for t in tiles}
    if len(resolutions) > 1:
        raise ValueError(f"align() requires one resolution, got: {resolutions}")

    minx = max(t.bbox[0] for t in tiles)
    miny = max(t.bbox[1] for t in tiles)
    maxx = min(t.bbox[2] for t in tiles)
    maxy = min(t.bbox[3] for t in tiles)
    if minx >= maxx or miny >= maxy:
        raise ValueError("Tiles have no spatial overlap — cannot align")

    aligned: list[GeoTile] = []
    for t in tiles:
        res = t.resolution
        left, _, _, top = t.bbox
        col0 = (minx - left) / res
        row0 = (top - maxy) / res
        ncols = (maxx - minx) / res
        nrows = (maxy - miny) / res
        if any(abs(v - round(v)) > 1e-6 for v in (col0, row0, ncols, nrows)):
            raise ValueError("Tiles are not on a common pixel grid; reproject first")
        col0, row0, ncols, nrows = round(col0), round(row0), round(ncols), round(nrows)
        sub = t.geobox[row0:row0 + nrows, col0:col0 + ncols]
        aligned_t = t.with_geobox(sub)
        if t.polygon is not None:
            clip_box = Geometry(
                {
                    "type": "Polygon",
                    "coordinates": [[(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]],
                },
                crs=t.polygon.crs,
            )
            aligned_t = dataclasses.replace(aligned_t, polygon=t.polygon & clip_box)
        aligned.append(aligned_t)
    return tuple(aligned)

TimeRound = Literal["D", "H", "T", "S", "L", "U", "N"]  # Pandas offset aliases

def mosaic(
    tiles: list[GeoTile],
    crs: str | None = None,
    time_round_to: TimeRound = 'D',
) -> GeoTile:
    """Stitch spatially non-overlapping tiles into one larger tile.

    All tiles must have data loaded and share band names and time coordinates.

    Args:
        tiles: Tiles to merge; all must have data.
        crs: Reproject tiles to this CRS before merging. Required if tiles differ in CRS.
        time_round_to: Pandas offset alias (e.g. "D") to floor time coords before matching.

    Raises:
        ValueError: If tiles is empty, any tile has no data, CRS mismatch without crs=, or band/time mismatch.
    """
    if not tiles:
        raise ValueError("Cannot mosaic an empty tile list")
    if any(t.data is None for t in tiles):
        raise ValueError("All tiles must have data loaded before mosaicking")

    tile_crss = {t.crs for t in tiles}
    if crs is None and len(tile_crss) > 1:
        raise ValueError(
            f"Cannot mosaic: tiles have different CRS: {tile_crss}. Pass crs= to reproject."
        )

    das: list[xr.DataArray] = []
    for t in tiles:
        da = t.data
        assert da is not None
        if time_round_to is not None and "time" in da.dims:
            da = da.assign_coords(time=da.time.dt.floor(time_round_to))
        if crs is not None and t.crs != crs:
            da = da.rio.reproject(crs)
        das.append(da)

    band_sets = {tuple(str(b) for b in da.coords["band"].values) for da in das}
    if len(band_sets) > 1:
        raise ValueError(f"Cannot mosaic: tiles have different bands: {band_sets}")
    time_sets = {
        tuple(str(v) for v in da.time.values) if "time" in da.dims else ()
        for da in das
    }
    if len(time_sets) > 1:
        raise ValueError(
            "Cannot mosaic: tiles have different time steps; pass time_round_to= for tolerance"
        )

    merged_ds = merge_datasets([da.to_dataset(dim="band") for da in das])
    merged = merged_ds.to_array(dim="band")
    if "time" in merged.dims:
        merged = merged.transpose("time", "band", "y", "x")
    else:
        merged = merged.transpose("band", "y", "x")
    geobox = GeoBox.from_bbox(
        merged.rio.bounds(),
        crs=merged.rio.crs.to_string(),
        resolution=tiles[0].resolution,
    )
    mosaic_polygon: Geometry | None = None
    tile_polys = [t.polygon for t in tiles]
    if all(p is not None for p in tile_polys):
        target_crs_str: str | None = crs or tiles[0].crs
        first_poly = tile_polys[0]
        assert first_poly is not None
        merged_poly: Geometry = first_poly
        for p in tile_polys[1:]:
            if p is not None:
                if target_crs_str is not None and str(merged_poly.crs) != target_crs_str:
                    merged_poly = merged_poly.to_crs(target_crs_str)
                merged_poly = merged_poly | p
        mosaic_polygon = merged_poly
    base = GeoTile(
        geobox=geobox,
        datetime=max(t.datetime for t in tiles),
        metadata={k: v for t in tiles for k, v in t.metadata.items()},
        polygon=mosaic_polygon,
    ).with_stac([item for t in tiles for item in t.stac])
    return base.with_data(merged)


# ----------------------------------------------------------------------
# STAC provenance sidecar (<stem>.stac.json — a pystac ItemCollection)
# ----------------------------------------------------------------------

def _stac_sidecar(path: Path) -> Path:
    """Sidecar JSON path for a saved tile: ``<stem>.stac.json`` beside it."""
    return path.parent / f"{path.stem}.stac.json"


def _write_stac(items: list[Item], path: Path) -> None:
    """Write STAC items as pystac ItemCollection sidecar. No-op if empty."""
    if items:
        ItemCollection(items).save_object(str(_stac_sidecar(path)))


def _read_stac(path: Path) -> list[Item]:
    """Read a tile's STAC sidecar back into Items, or ``[]`` if none exists."""
    sidecar = _stac_sidecar(path)
    if not sidecar.exists():
        return []
    return list(ItemCollection.from_file(str(sidecar)))

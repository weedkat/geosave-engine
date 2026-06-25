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
from typing import Any, cast
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
    attrs: dict[str, Any],
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
    """Geospatial tile: a geobox, an anchor datetime, and optional pixel data.

    ``datetime`` is the anchor time — when the tile was requested or created.
    ``data`` is an ``xr.Dataset`` whose variables are bands keyed by name, each
    shaped ``(y, x)`` or ``(time, y, x)``. It may be lazy (opened from a
    GeoTIFF/COG/Zarr and read only on access) or fully in memory; ``None`` means a
    header-only tile carrying just the geobox and datetime.

    ``bands`` and ``times`` are derived from ``data`` via properties — they are
    not stored as fields to avoid manual synchronisation. Combining tiles
    (``mosaic``, ``align``) and label remapping (``remap``) are module-level free
    functions, not methods — a tile renders itself, it does not stitch others.
    """

    geobox: GeoBox
    datetime: dt
    data: xr.Dataset | None = field(default=None, compare=False)
    stac: list[Item] = field(default_factory=list, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def bands(self) -> tuple[str, ...]:
        """Band (variable) names from loaded data. Empty when data is None."""
        if self.data is None:
            return ()
        return tuple(str(name) for name in self.data.data_vars)

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
    def polygon(self) -> Geometry:
        return self.geobox.boundingbox.polygon

    @property
    def geojson(self) -> dict:
        """WGS84 footprint as a GeoJSON Polygon dict."""
        return self.polygon.geojson()

    @property
    def location(self) -> dict[str, Any]:
        """Reverse-geocode centroid. Returns {} on failure."""
        lon, lat = self.centroid
        place = Place.from_coordinate(lat, lon)
        return place.to_dict() if place is not None else {}

    # ------------------------------------------------------------------
    # Data manipulation
    # ------------------------------------------------------------------

    def with_data(self, data: xr.Dataset) -> "GeoTile":
        """Return a new GeoTile with the given Dataset as pixel data.

        ``data`` variables are bands keyed by name, each shaped ``(y, x)`` or
        ``(time, y, x)``. Callers holding a DataArray or numpy array convert it to
        a Dataset first (e.g. ``da.to_dataset(dim="band")``).
        """
        if not isinstance(data, xr.Dataset):
            raise TypeError(f"with_data expects an xr.Dataset, got {type(data).__name__}")
        return dataclasses.replace(self, data=data)

    def with_np(
        self,
        array: np.ndarray,
        names: list[str],
        times: list[dt] | None = None,
    ) -> "GeoTile":
        """Build a Dataset from a numpy array on this tile's geobox and attach it.

        The band axis (length ``len(names)``) becomes one variable per name, with
        spatial coords and CRS taken from the geobox. Accepts ``(y, x)`` (one
        band), ``(band, y, x)``, or ``(time, band, y, x)`` (pass ``times``).

        Args:
            array: Pixel array; the last two axes are ``(y, x)``.
            names: Variable name per band, in order.
            times: Observation datetimes — required for a 4D array.
        """
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        coords: dict[Any, Any] = dict(xr_coords(self.geobox))
        if arr.ndim == 3:
            if len(names) != arr.shape[0]:
                raise ValueError(f"Expected {arr.shape[0]} names, got {len(names)}")
            data_vars = {name: (("y", "x"), arr[i]) for i, name in enumerate(names)}
        elif arr.ndim == 4:
            if len(names) != arr.shape[1]:
                raise ValueError(f"Expected {arr.shape[1]} names, got {len(names)}")
            if times is None or len(times) != arr.shape[0]:
                got = 0 if times is None else len(times)
                raise ValueError(f"Expected {arr.shape[0]} times, got {got}")
            coords["time"] = [np.datetime64(t, "ns") for t in times]
            data_vars = {name: (("time", "y", "x"), arr[:, i]) for i, name in enumerate(names)}
        else:
            raise ValueError(f"with_np expects a 2-4D array, got {arr.ndim}D")
        return self.with_data(xr.Dataset(data_vars, coords=coords))

    def with_stac(self, items: list[Item]) -> "GeoTile":
        """Append pystac Items as provenance, de-duplicated by id."""
        seen = {i.id for i in self.stac}
        merged = [*self.stac, *(i for i in items if i.id not in seen)]
        return dataclasses.replace(self, stac=merged)

    def with_metadata(self, extra: dict[str, Any], replace: bool = False) -> "GeoTile":
        """Merge key-value pairs into the metadata field.

        Append-only by default: raises if any key is already present, so
        accidental clobbering is loud. Pass ``replace=True`` to overwrite.
        """
        if not replace:
            clash = set(extra) & set(self.metadata)
            if clash:
                raise ValueError(
                    f"metadata keys already present: {sorted(clash)}; pass replace=True to overwrite"
                )
        return dataclasses.replace(self, metadata={**self.metadata, **extra})

    def with_geobox(self, geobox: GeoBox) -> "GeoTile":
        """Return a new GeoTile rebased onto ``geobox``, sharing data.

        Used to derive lazy patch tiles: pass a sub-geobox (e.g. ``self.geobox[
        y0:y1, x0:x1]``) and the tile keeps its lazy ``data``, so ``to_tensor``
        reads only the patch's extent. Pure geometry — no pixels are read or copied.
        """
        return dataclasses.replace(self, geobox=geobox)

    # ------------------------------------------------------------------
    # Tensor loading
    # ------------------------------------------------------------------

    def to_tensor(self, bands: list[str] | None = None) -> Any:
        """Render the data as a single ``torch.Tensor``, bands stacked.

        Clips the (possibly lazy) data to ``self.bbox`` so windowed patch tiles
        read only their own extent, then stacks the band variables along a band
        axis: shape ``(band, y, x)`` or ``(time, band, y, x)``.

        Args:
            bands: Band (variable) names to select, in order. None uses all bands.
        """
        return torch.from_numpy(self.to_numpy(bands=bands))

    def to_numpy(self, bands: list[str] | None = None) -> np.ndarray:
        """Render the data as a contiguous NumPy array, bands stacked.

        Clips the (possibly lazy) data to ``self.bbox`` so windowed patch tiles
        read only their own extent, then stacks the band variables along a band
        axis: shape ``(band, y, x)`` or ``(time, band, y, x)``.

        Args:
            bands: Band (variable) names to select, in order. None uses all bands.
        """
        if self.data is None:
            raise ValueError("GeoTile has no data — load data first")
        da = self.data.rio.clip_box(*self.bbox).to_array(dim="band")
        if bands is not None:
            da = da.sel(band=bands)
        if "time" in da.dims:
            da = da.transpose("time", "band", "y", "x")
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
        """Create a GeoTile from a single GeoTIFF or COG file.

        The raster is opened lazily (chunked) as a Dataset with one variable per
        band: pixels are read only when accessed via ``to_tensor`` or
        ``load_data=True``. COGs and tiled GeoTIFFs serve windowed reads cheaply;
        plain striped TIFFs read whole strips per window.

        Args:
            path: Path to a GeoTIFF/COG file.
            load_data: Materialise all pixels into memory when True. Defaults to
                lazy (False).
            bands: Band (variable) names to select. None keeps all bands.
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

        if bands:
            data = cast(xr.Dataset, data[list(bands)])
        if load_data:
            data = data.load()

        anchor_dt = _datetime_from_attrs_or_stem(
            p, data.attrs, date_format=date_format, date_pattern=date_pattern
        )
        return cls(
            geobox=data.odc.geobox,
            datetime=anchor_dt,
            data=data,
            stac=_read_stac(p),
            metadata=tag,
        )

    @classmethod
    def from_zarr(cls, path: str | Path, load_data: bool = False) -> "GeoTile":
        """Create a GeoTile from a Zarr store written by ``to_zarr``.

        Opened lazily by default — pixels are read only when accessed via
        ``to_tensor`` or ``load_data=True``. Zarr carries the full data cube,
        including any time dimension. The geobox, anchor datetime, and metadata
        are restored from the store attributes.

        Args:
            path: Path to a Zarr store.
            load_data: Materialise all pixels into memory when True.
        """
        path = Path(path)
        ds = xr.open_zarr(path)
        # zarr restores the CRS grid-mapping as a data variable; demote it to a coord
        grid_mappings = {
            da.attrs["grid_mapping"]
            for da in ds.data_vars.values()
            if "grid_mapping" in da.attrs
        } & set(ds.data_vars)
        if grid_mappings:
            ds = ds.set_coords(grid_mappings)
        if load_data:
            ds = ds.load()
        return cls(
            geobox=ds.odc.geobox,
            datetime=_datetime_from_attrs_or_stem(path, ds.attrs),
            data=ds,
            stac=_read_stac(path),
            metadata=json.loads(ds.attrs.get("metadata", "{}")),
        )

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        crs: str,
        resolution: float,
        datetime: dt | str,
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
        size_m: float | tuple[float, float],
        resolution: float,
        datetime: dt | str,
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
        resolution: float,
        datetime: dt | str,
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
        return cls(
            geobox=GeoBox.from_geopolygon(
                geom.to_crs(target_crs), resolution=resolution, anchor="edge"
            ),
            datetime=parse_datetime(datetime),
        )

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        resolution: float,
        datetime: dt | str,
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
        """Write a single-step tile to a GeoTIFF file (one band per variable).

        Raises ``ValueError`` if the tile has a time dimension — use ``to_zarr``
        for time-series cubes. With ``save_stac=True``, STAC provenance is written
        to a ``<stem>.stac.json`` sidecar beside the file.

        Returns:
            The written path.
        """
        return self._write(path, driver="GTiff", save_stac=save_stac)

    def to_cog(self, path: str | Path, save_stac: bool = False) -> Path:
        """Write a single-step tile to a Cloud-Optimized GeoTIFF file.

        Raises ``ValueError`` if the tile has a time dimension — use ``to_zarr``
        for time-series cubes. With ``save_stac=True``, STAC provenance is written
        to a ``<stem>.stac.json`` sidecar beside the file.

        Returns:
            The written path.
        """
        return self._write(path, driver="COG", save_stac=save_stac)

    def to_zarr(self, path: str | Path, save_stac: bool = False) -> Path:
        """Write tile data — including any time dimension — to a Zarr store.

        Anchor datetime and metadata are stored as store attributes. With
        ``save_stac=True``, STAC provenance is written to a ``<stem>.stac.json``
        sidecar (a pystac ItemCollection) beside the store.

        Returns:
            The written store path.
        """
        if self.data is None:
            raise ValueError("GeoTile has no data to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = self.data.assign_attrs(
            datetime=self.datetime.isoformat(),
            metadata=json.dumps(self.metadata),
        )
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
        tag: dict[str, Any] = {**self.metadata, "bands": [str(b) for b in self.data.data_vars]}
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

def remap(tile: GeoTile, mapping: dict[int, int]) -> GeoTile:
    """Return a new GeoTile with label values remapped per ``mapping``."""
    if tile.data is None:
        raise ValueError("Cannot remap a GeoTile without data")
    remapped = tile.data
    for src_val, dst_val in mapping.items():
        remapped = remapped.where(remapped != src_val, other=dst_val)
    return tile.with_data(remapped)


def align(*tiles: GeoTile) -> tuple[GeoTile, ...]:
    """Narrow each tile's geobox to their common intersection — pure geometry.

    Tiles must share CRS, resolution, and a common pixel grid (as produced by
    ingesting every layer onto the same anchor geobox). Only the geobox is
    narrowed; data is shared untouched, so the intersection is read lazily on
    ``to_tensor``. Reproject mismatched grids first — not align's concern.
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
        aligned.append(t.with_geobox(sub))
    return tuple(aligned)


def mosaic(
    tiles: list[GeoTile],
    crs: str | None = None,
    round_to: str | None = None,
) -> GeoTile:
    """Stitch spatially non-overlapping tiles into one larger tile.

    Materialises a union grid via rioxarray ``merge_datasets``. All tiles must
    have data loaded and share band names + time coordinates. Pass ``crs=`` to
    reproject differing-CRS tiles first; pass ``round_to`` (a pandas offset alias,
    e.g. ``"D"``) to floor time coordinates before matching, tolerating
    sub-period acquisition jitter.
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

    datasets: list[xr.Dataset] = []
    for t in tiles:
        ds = t.data
        assert ds is not None
        if round_to is not None and "time" in ds.dims:
            ds = ds.assign_coords(time=ds.time.dt.floor(round_to))
        if crs is not None and t.crs != crs:
            ds = ds.rio.reproject(crs)
        datasets.append(ds)

    band_sets = {tuple(d.data_vars) for d in datasets}
    if len(band_sets) > 1:
        raise ValueError(f"Cannot mosaic: tiles have different bands: {band_sets}")
    time_sets = {
        tuple(str(v) for v in d.time.values) if "time" in d.dims else ()
        for d in datasets
    }
    if len(time_sets) > 1:
        raise ValueError(
            "Cannot mosaic: tiles have different time steps; pass round_to= for tolerance"
        )

    merged = merge_datasets(datasets)
    geobox = GeoBox.from_bbox(
        merged.rio.bounds(),
        crs=merged.rio.crs.to_string(),
        resolution=tiles[0].resolution,
    )
    base = GeoTile(
        geobox=geobox,
        datetime=max(t.datetime for t in tiles),
        metadata={k: v for t in tiles for k, v in t.metadata.items()},
    ).with_stac([item for t in tiles for item in t.stac])
    return base.with_data(merged)


# ----------------------------------------------------------------------
# STAC provenance sidecar (<stem>.stac.json — a pystac ItemCollection)
# ----------------------------------------------------------------------

def _stac_sidecar(path: Path) -> Path:
    """Sidecar JSON path for a saved tile: ``<stem>.stac.json`` beside it."""
    return path.parent / f"{path.stem}.stac.json"


def _write_stac(items: list[Item], path: Path) -> None:
    """Write STAC provenance as a pystac ItemCollection sidecar (no-op if empty).

    All of a tile's items — every time step included — live in this one
    ItemCollection file; an ItemCollection is itself "multiple items".
    """
    if items:
        ItemCollection(items).save_object(str(_stac_sidecar(path)))


def _read_stac(path: Path) -> list[Item]:
    """Read a tile's STAC sidecar back into Items, or ``[]`` if none exists."""
    sidecar = _stac_sidecar(path)
    if not sidecar.exists():
        return []
    return list(ItemCollection.from_file(str(sidecar)))

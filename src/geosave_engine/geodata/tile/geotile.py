from __future__ import annotations

import dataclasses
import json
import warnings
import zarr.errors
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray/Dataset
import numpy as np
import xarray as xr
import torch

from pystac import Item, ItemCollection
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path

from typing import TYPE_CHECKING, Any, cast, Literal
from rioxarray.merge import merge_datasets
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from typing_extensions import Unpack

from geosave_engine.geodata.utils.datetime import date_range_from_path
from geosave_engine.utils.colorize import Palette

from .geoanchor import AnchorDatetime, GeoAnchor

# consolidated=True on to_zarr is load-bearing (see to_zarr below), not a guess xarray
# made — zarr itself warns on every consolidated write/read under the v3 format
# regardless, since the format spec doesn't officially cover it yet. Informational,
# not actionable: nothing here reads/writes through a different, spec-strict
# implementation that this would actually bite.
warnings.filterwarnings(
    "ignore", message=r"Consolidated metadata is currently not part .* Zarr format 3", category=zarr.errors.ZarrUserWarning
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from geosave_engine.geodata.utils.geovis import PlotKwargs


@dataclass(frozen=True)
class PlotMeta:
    """Rendering hints a GeoTile carries about itself.

    Otherwise ambiguous from the tile's shape/dtype alone (e.g. which 3 of
    more than 3 bands count as RGB), or would have to be repeated at every
    `plot()` call site. `None` on any field means "not set" — `plot()`
    falls back to auto-detection or its own call-level kwarg.

    Args:
        rgb_bands: Which 3 band names count as R/G/B.
        class_map: `{pixel value: class name}` for a categorical tile.
        color_map: `{pixel value: hex or RGB}` for a categorical tile.
    """

    rgb_bands: tuple[str, str, str] | None = None
    class_map: dict[int, str] | None = None
    color_map: Palette | None = None


def _plot_meta_to_dict(meta: PlotMeta) -> dict[str, Any]:
    """JSON-safe dict for a store attr/tag — reversed by _plot_meta_from_dict."""
    return {
        "rgb_bands": list(meta.rgb_bands) if meta.rgb_bands is not None else None,
        "class_map": meta.class_map,
        "color_map": meta.color_map,
    }


def _plot_meta_from_dict(data: dict[str, Any] | None) -> PlotMeta:
    """Inverse of _plot_meta_to_dict — JSON object keys are always strings, cast back to int."""
    if not data:
        return PlotMeta()
    rgb_bands = data.get("rgb_bands")
    class_map = data.get("class_map")
    color_map = data.get("color_map")
    return PlotMeta(
        rgb_bands=tuple(rgb_bands) if rgb_bands is not None else None,
        class_map={int(k): v for k, v in class_map.items()} if class_map is not None else None,
        color_map=(
            {int(k): (tuple(v) if isinstance(v, list) else v) for k, v in color_map.items()}
            if color_map is not None
            else None
        ),
    )


@dataclass(frozen=True, kw_only=True)
class GeoTile(GeoAnchor):
    """Geospatial tile with a geobox, anchor datetime, and pixel data.

    `data` is an `xr.DataArray` with dims `(band, y, x)` or
    `(time, band, y, x)`. May be lazy or fully in memory, but always
    present — a data-less reference is a `GeoAnchor`, not a `GeoTile`.

    Examples:
        >>> tile = GeoTile.from_geotiff("sentinel_2_l1c-20240101.tif")
        >>> tile.to_zarr("data/train/13.0_52.0_20240101.geostack/sentinel_2_l1c.zarr")
    """

    data: xr.DataArray
    stac: list[Item] = field(default_factory=list, compare=False)
    plot_meta: PlotMeta = field(default_factory=PlotMeta, compare=False)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(str(b) for b in self.data.coords["band"].values)

    @property
    def times(self) -> tuple[dt, ...]:
        """Observation datetimes from loaded data. Empty when data has no time."""
        if "time" not in self.data.dims:
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
        return "time" in self.data.dims

    @property
    def nodata(self) -> float | None:
        """GDAL-standard nodata value, if set. None means no nodata declared."""
        return self.data.rio.nodata

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._repr_fields()}, bands={self.bands}, shape={self.data.shape})"

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

    def with_nodata(self, value: float | None) -> "GeoTile":
        """Return new GeoTile with the given GDAL-standard nodata value set.

        Args:
            value: Nodata value. None clears any existing nodata declaration.
        """
        return self.with_data(self.data.rio.write_nodata(value))

    def with_stac(self, items: list[Item]) -> "GeoTile":
        """Append pystac Items as provenance, de-duplicated by id."""
        seen = {i.id for i in self.stac}
        merged = [*self.stac, *(i for i in items if i.id not in seen)]
        return dataclasses.replace(self, stac=merged)

    def with_plot_meta(
        self,
        rgb_bands: tuple[str, str, str] | None = None,
        class_map: dict[int, str] | None = None,
        color_map: Palette | None = None,
    ) -> "GeoTile":
        """Return new GeoTile with given rendering hints merged into plot_meta.

        Only given (non-None) args overwrite; omitted ones keep whatever
        plot_meta already had — same merge shape as with_metadata.

        Args:
            rgb_bands: Which 3 band names count as R/G/B.
            class_map: `{pixel value: class name}` for a categorical tile.
            color_map: `{pixel value: hex or RGB}` for a categorical tile.
        """
        updates = {
            k: v
            for k, v in {"rgb_bands": rgb_bands, "class_map": class_map, "color_map": color_map}.items()
            if v is not None
        }
        return dataclasses.replace(self, plot_meta=dataclasses.replace(self.plot_meta, **updates))

    def to_anchor(self) -> GeoAnchor:
        """Strip pixel data (and STAC provenance), keeping only the anchor identity.

        The reverse of `GeoAnchor.with_data`/`with_np` — for carrying this
        tile's location/datetime through somewhere that shouldn't also drag
        the raster array along (e.g. a batch's `"anchor"` key).

        Returns:
            A bare `GeoAnchor` with this tile's `geobox`/`datetime`/`metadata`/`polygon`.
        """
        return GeoAnchor(geobox=self.geobox, datetime=self.datetime, metadata=self.metadata, polygon=self.polygon)

    def plot(self, **kwargs: Unpack[PlotKwargs]) -> tuple[Figure, np.ndarray]:
        """Plot this tile — thin wrapper, see `geosave_engine.geodata.utils.geovis.plot`.

        Matplotlib is imported lazily here, not at module load, so GeoTile
        itself stays free of a hard plotting dependency.

        Args:
            **kwargs: Forwarded to `geovis.plot` (`cmap`, `class_map`,
                `color_map`, `rgb_bands`, `cols`, `title`).

        Returns:
            `(Figure, ndarray of Axes)`.
        """
        from geosave_engine.geodata.utils.geovis import plot

        return plot(self, **kwargs)

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
        """
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
        datetime: AnchorDatetime | None = None,
        load_data: bool = False,
        bands: tuple[str, ...] | None = None,
    ) -> "GeoTile":
        """Create GeoTile from a single GeoTIFF or COG file.

        Opened lazily by default; pixels read only on to_tensor or load_data=True.

        Args:
            path: Path to GeoTIFF/COG file.
            datetime: Anchor datetime or (start, end) date range for this tile.
                None derives it from the filename's standard date suffix
                (``-YYYYMMDD`` or ``-YYYYMMDD-YYYYMMDD``) via
                ``date_range_from_path`` — pass an explicit value to bypass
                that convention entirely.
            load_data: Materialise all pixels into memory; default lazy.
            bands: Band variable names to select; None keeps all.

        Raises:
            ValueError: If datetime is None and the filename has no standard
                date suffix.
        """
        p = Path(path)
        if datetime is None:
            datetime = date_range_from_path(p)
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

        plot_meta = _plot_meta_from_dict(tag.pop("plot_meta", None))

        if bands:
            data = cast(xr.Dataset, data[list(bands)])
        if load_data:
            data = data.load()

        geobox = data.odc.geobox
        # to_array() stacks per-variable DataArrays into one — rio.nodata lives on each
        # variable individually and doesn't survive the stack, so re-attach it explicitly.
        nodata = next(iter(data.data_vars.values())).rio.nodata
        da = data.to_array(dim="band").transpose("band", "y", "x")
        if nodata is not None:
            da = da.rio.write_nodata(nodata)
        return cls(
            geobox=geobox,
            datetime=datetime,
            data=da,
            stac=_read_stac(p),
            metadata=tag,
            polygon=stored_polygon,
            plot_meta=plot_meta,
        )

    @classmethod
    def from_zarr(
        cls,
        path: str | Path,
        datetime: AnchorDatetime | None = None,
        load_data: bool = False,
    ) -> "GeoTile":
        """Create GeoTile from a Zarr store written by to_zarr.

        Opened lazily by default. Geobox and metadata restored from store
        attrs. Datetime prefers the store's own `time` coordinate when
        present (start/end = min/max observed date — the authoritative
        source for a multi-step tile); falls back to the `datetime` store
        attr only when there's no time dimension to derive it from. This
        never guesses a date from the path.

        Args:
            path: Path to Zarr store.
            datetime: Explicit override — bypasses both the `time` coordinate
                and the `datetime` attr entirely. None uses the store's own.
            load_data: Materialise all pixels into memory; default lazy.

        Raises:
            ValueError: If datetime is None, the store has no time dimension,
                and no `datetime` attr — was it written by GeoTile.to_zarr?
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
        metadata = json.loads(ds.attrs.get("metadata", "{}"))
        poly_geojson_raw = ds.attrs.get("polygon_geojson")
        poly_crs = ds.attrs.get("polygon_crs")
        stored_polygon: Geometry | None = None
        if poly_geojson_raw and poly_crs:
            geojson_dict = json.loads(poly_geojson_raw) if isinstance(poly_geojson_raw, str) else poly_geojson_raw
            stored_polygon = Geometry(geojson_dict, crs=poly_crs)
        plot_meta_raw = ds.attrs.get("plot_meta")
        plot_meta = _plot_meta_from_dict(json.loads(plot_meta_raw) if plot_meta_raw else None)
        # to_array() stacks per-variable DataArrays into one — rio.nodata lives on each
        # variable individually and doesn't survive the stack, so re-attach it explicitly.
        nodata = next(iter(ds.data_vars.values())).rio.nodata
        da = ds.to_array(dim="band")
        has_time = "time" in da.dims
        da = da.transpose("time", "band", "y", "x") if has_time else da.transpose("band", "y", "x")
        if nodata is not None:
            da = da.rio.write_nodata(nodata)

        if datetime is not None:
            resolved_datetime: AnchorDatetime = datetime
        elif has_time:
            times = da.time.values
            resolved_datetime = (
                dt.fromisoformat(str(times.min().astype("datetime64[s]"))),
                dt.fromisoformat(str(times.max().astype("datetime64[s]"))),
            )
        else:
            raw_datetime = ds.attrs.get("datetime")
            if raw_datetime is None:
                raise ValueError(
                    f"Zarr store at {path} has no time dimension and no 'datetime' attr — "
                    "was it written by GeoTile.to_zarr?"
                )
            resolved_datetime = raw_datetime
        if load_data:
            da = da.load()
        return cls(
            geobox=geobox,
            datetime=resolved_datetime,
            data=da,
            stac=_read_stac(path),
            metadata=metadata,
            polygon=stored_polygon,
            plot_meta=plot_meta,
        )

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
            ValueError: If tile has a time dimension.
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
            ValueError: If tile has a time dimension.
        """
        return self._write(path, driver="COG", save_stac=save_stac)

    def to_zarr(self, path: str | Path, save_stac: bool = False) -> Path:
        """Write tile data including any time dimension to a Zarr store.

        Metadata stored as store attributes. Anchor datetime is stored as an
        attribute too, but only when there's no time dimension — a
        multi-step tile's real dates already live natively in its `time`
        coordinate, so `from_zarr` derives start/end from that instead of a
        second, redundant copy.

        Args:
            path: Output Zarr store path.
            save_stac: Write STAC provenance as <stem>.stac.json sidecar.

        Returns:
            The written store path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        attrs: dict[str, Any] = {"metadata": json.dumps(self.metadata)}
        if not self.has_time:
            attrs["datetime"] = f"{self.start.isoformat(timespec='microseconds')}/{self.end.isoformat(timespec='microseconds')}"
        if self.polygon is not None:
            attrs["polygon_geojson"] = json.dumps(self.polygon.geojson())
            attrs["polygon_crs"] = str(self.polygon.crs)
        if self.plot_meta != PlotMeta():
            attrs["plot_meta"] = json.dumps(_plot_meta_to_dict(self.plot_meta))
        ds = self.data.to_dataset(dim="band").assign_attrs(**attrs)
        # consolidated=True (was implicit): band/variable order through the round-trip
        # (from_zarr's to_array(dim="band")) depends on consolidated metadata preserving
        # insertion order — without it, zarr falls back to a listing that doesn't
        # guarantee order at all. Explicit since this is load-bearing, not a convenience.
        ds.to_zarr(path, mode="w", consolidated=True)
        if save_stac:
            _write_stac(self.stac, path)
        return path

    def _write(self, path: str | Path, driver: str, save_stac: bool = False) -> Path:
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
        if self.plot_meta != PlotMeta():
            tag["plot_meta"] = _plot_meta_to_dict(self.plot_meta)
        dt_str = f"{self.start.isoformat(timespec='microseconds')}/{self.end.isoformat(timespec='microseconds')}"
        self.data.rio.to_raster(
            path,
            driver=driver,
            tags={"metadata": json.dumps(tag), "datetime": dt_str},
        )
        if save_stac:
            _write_stac(self.stac, path)
        return path


# ----------------------------------------------------------------------
# Tile operations
# ----------------------------------------------------------------------

def remap(tile: GeoTile, mapping: dict[int, int]) -> GeoTile:
    """Return a new GeoTile with label values remapped per ``mapping``."""
    remapped = tile.data
    for src_val, dst_val in mapping.items():
        remapped = remapped.where(remapped != src_val, other=dst_val)
    return tile.with_data(remapped)


def align(*tiles: GeoTile) -> tuple[GeoTile, ...]:
    """Narrow each tile's geobox to their common intersection.

    Pure geometry — data is shared untouched. Tiles must share CRS,
    resolution, and pixel grid; that shared CRS must also be projected
    (metric) — `resolution` (`geobox.affine.a`) only means meters under a
    projected CRS, degrees under a geographic one (e.g. EPSG:4326), and
    nothing downstream (GSD-conditioned models, area_m2, ...) means to
    handle the degrees case.

    Raises:
        ValueError: If fewer than 2 tiles, CRS/resolution mismatch, that CRS
            isn't projected, no overlap, or misaligned grid.
    """
    if len(tiles) < 2:
        raise ValueError("align() requires at least 2 tiles")
    crss = {t.crs for t in tiles}
    if len(crss) > 1:
        raise ValueError(f"align() requires one CRS, got: {crss}")
    crs = tiles[0].geobox.crs
    if crs is None or not crs.projected:
        raise ValueError(
            f"align() requires a projected CRS (resolution must mean meters), got "
            f"{tiles[0].crs} — reproject to a projected CRS (e.g. local UTM) first"
        )
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

    All tiles must share band names and time coordinates.

    Args:
        tiles: Tiles to merge.
        crs: Reproject tiles to this CRS before merging. Required if tiles differ in CRS.
        time_round_to: Pandas offset alias (e.g. "D") to floor time coords before matching.

    Raises:
        ValueError: If tiles is empty, CRS mismatch without crs=, or band/time mismatch.
    """
    if not tiles:
        raise ValueError("Cannot mosaic an empty tile list")
    if any(t.start != t.end for t in tiles):
        raise ValueError("Cannot mosaic range-datetime tiles; ingest first to resolve to single datetimes")

    tile_crss = {t.crs for t in tiles}
    if crs is None and len(tile_crss) > 1:
        raise ValueError(
            f"Cannot mosaic: tiles have different CRS: {tile_crss}. Pass crs= to reproject."
        )

    das: list[xr.DataArray] = []
    for t in tiles:
        da = t.data
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
        data=merged,
        metadata={k: v for t in tiles for k, v in t.metadata.items()},
        polygon=mosaic_polygon,
        plot_meta=tiles[0].plot_meta,
    ).with_stac([item for t in tiles for item in t.stac])
    return base


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

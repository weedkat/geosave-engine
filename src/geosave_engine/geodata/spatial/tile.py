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
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path

from typing import TYPE_CHECKING, Any, Self
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from rasterio.enums import Resampling
from typing_extensions import Unpack

from geosave_engine.geodata.utils.datetime import extract_stem_dates, format_stem_dates
from geosave_engine.geodata.utils.geodata import da_to_ds, ds_to_da, validate_da
from geosave_engine.geodata.utils.geotiff import from_geotiff, to_geotiff
from geosave_engine.geodata.utils.netcdf import from_netcdf, to_netcdf
from geosave_engine.geodata.utils.zarr import from_zarr, to_zarr
from geosave_engine.utils.colorize import Palette

from .anchor import AnchorDatetime, GeoAnchor, GeoTag

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
    from .ops import MergeMethod


@dataclass(frozen=True, kw_only=True)
class GeoTile(GeoAnchor):
    """Geospatial tile with a geobox, anchor datetime, and pixel data.

    `data` is an `xr.DataArray` with dims `(band, y, x)` or
    `(time, band, y, x)`. May be lazy or fully in memory, but always
    present — a data-less reference is a `GeoAnchor`, not a `GeoTile`.

    Examples:
        >>> tile = GeoTile.from_geotiff("sentinel_2_l1c-20240101.tif")
        >>> tile.to_zarr("data/train/sentinel_2_l1c-20240101.zarr")
    """

    data: xr.DataArray

    def __post_init__(self) -> None:
        """Sync geotag.bands to data's own band coordinate, always overwritten."""
        bands = tuple(str(b) for b in self.data.coords["band"].values) if "band" in self.data.dims else None
        
        if self.geotag.bands != bands:
            object.__setattr__(self, "geotag", self.geotag.model_copy(update={"bands": bands}))

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def bands(self) -> tuple[str, ...]:
        """Band names. Empty when data has no 'band' dim (a single unnamed band)."""
        if "band" not in self.data.dims:
            return ()
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
        """Number of bands. 1 when data has no 'band' dim (a single implicit band)."""
        return len(self.bands) if "band" in self.data.dims else 1

    @property
    def has_time(self) -> bool:
        """True if data has a time dimension."""
        return "time" in self.data.dims

    @property
    def nodata(self) -> float | None:
        """GDAL-standard nodata value, if set. None means no nodata declared."""
        return self.data.rio.nodata

    def __repr__(self) -> str:
        return f"{super().__repr__()}\n  bands:    {self.bands}\n  shape:    {self.data.shape}"

    def __and__(self, other: "GeoTile") -> "GeoTile":
        """Sugar for `GeoTile.merge(self, other)` — `&` matches align_spatial/mosaic's own intersect/union use elsewhere in this codebase."""
        return GeoTile.merge(self, other)

    # ------------------------------------------------------------------
    # Data manipulation
    # ------------------------------------------------------------------

    def rebase(
        self,
        *,
        geobox: GeoBox | None = None,
        datetime: AnchorDatetime | None = None,
        polygon: Geometry | None = None,
        rgb_bands: tuple[str, str, str] | None = None,
        class_map: dict[int, str] | None = None,
        color_map: Palette | None = None,
        data: xr.DataArray | None = None,
        nodata: float | None = None,
        stac: list[Item] | None = None,
        **metadata: Any,
    ) -> Self:
        """Extends `GeoAnchor.rebase` with this tile's own fields.

        Args:
            geobox: New geobox.
            datetime: ISO/compact string, or (start, end) pair of either.
            polygon: New footprint polygon.
            rgb_bands: New RGB band names.
            class_map: New `{pixel value: class name}` map.
            color_map: New `{pixel value: hex or RGB}` map.
            data: New pixel data — must already be a validly-shaped
                DataArray (see `validate_da`). Build one from a plain array
                via `to_geotile` first if starting from raw pixels.
            nodata: GDAL-standard nodata value to set on the (possibly also-new) data.
            stac: pystac Items to append as provenance, de-duplicated by id — not a replace.
            **metadata: Custom key-value pairs merged into existing metadata.
        """
        merged_stac = None
        if stac is not None:
            seen = {i.id for i in self.stac}
            merged_stac = [*self.stac, *(i for i in stac if i.id not in seen)]
        base = super().rebase(
            geobox=geobox, datetime=datetime, polygon=polygon,
            rgb_bands=rgb_bands, class_map=class_map, color_map=color_map, stac=merged_stac, **metadata,
        )
        changes: dict[str, Any] = {}
        if data is not None:
            changes["data"] = validate_da(data)
        if nodata is not None:
            changes["data"] = changes.get("data", self.data).rio.write_nodata(nodata)
        return dataclasses.replace(base, **changes) if changes else base

    def reproject(
        self,
        crs: str | None = None,
        resolution: float | None = None,
        resampling: Resampling = Resampling.nearest,
    ) -> Self:
        """Reproject and/or resample onto a new CRS/resolution.

        Args:
            crs: Destination CRS. None keeps the current CRS.
            resolution: Destination pixel size. None lets rioxarray pick one.
            resampling: How to resample pixels onto the new grid.

        Returns:
            New GeoTile on the requested CRS/resolution.

        Raises:
            ValueError: Both crs and resolution are None.
        """
        if crs is None and resolution is None:
            raise ValueError("reproject() needs crs and/or resolution")

        # warp pixel data onto the new grid
        reprojected = self.data.rio.reproject(crs or self.crs, resolution=resolution, resampling=resampling)

        # carry the new grid into a fresh tile
        return self.rebase(geobox=reprojected.odc.geobox, data=reprojected)

    def to_anchor(self) -> GeoAnchor:
        """Strip pixel data (and STAC provenance), keeping only the anchor identity.

        The reverse of `GeoAnchor.to_geotile` — for carrying this tile's
        location/datetime through somewhere that shouldn't also drag the
        raster array along (e.g. a batch's `"anchor"` key). `geotag`
        (datetime/polygon/rendering hints/metadata) rides along untouched.

        Returns:
            A bare `GeoAnchor` with this tile's `geobox`/`geotag`.
        """
        return GeoAnchor(geobox=self.geobox, geotag=self.geotag)

    def rgb_subset(self) -> "GeoTile | None":
        """Select this tile's own RGB bands via its rgb_bands, if usable.

        Returns:
            New 3-band GeoTile in R,G,B order, or None if rgb_bands is
            unset or isn't a subset of this tile's own band names.
        """
        rgb_bands = self.rgb_bands
        if rgb_bands is None or not set(rgb_bands).issubset(self.bands):
            return None
        return self.to_geotile(self.to_numpy(bands=list(rgb_bands)), list(rgb_bands))

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

    def to_tensor(self, bands: list[str] | None = None) -> Any:
        """Render data as a single torch.Tensor with bands stacked.

        Clips to self.bbox before reading; output shape matches to_numpy's.

        Args:
            bands: Variable names to select, in order. None uses all bands.
        
        Return:
            torch.Tensor with shape (band, y, x), (y, x), (time, band, y, x),
            or (time, y, x) — matching self.data's own dims.
        """
        result = torch.from_numpy(self.to_numpy(bands=bands))
        return result

    def to_numpy(self, bands: list[str] | None = None) -> np.ndarray:
        """Render data as a contiguous NumPy array with bands stacked.

        Clips to self.bbox before reading; output shape (band, y, x), (y, x),
        (time, band, y, x), or (time, y, x) — matching self.data's own dims.

        Args:
            bands: Band names to select, in order. None uses all bands.
                Raises if data has no 'band' dim.

        Raises:
            ValueError: `bands` given but data has no 'band' dim.
        """
        da = self.data.rio.clip_box(*self.bbox)
        if bands is not None:
            if "band" not in da.dims:
                raise ValueError("Cannot select bands — this tile's data has no 'band' dim")
            da = da.sel(band=bands)
        da = validate_da(da)
        return np.ascontiguousarray(da.values)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def merge(cls, *tiles: "GeoTile", method: "MergeMethod" = "first") -> "GeoTile":
        """Narrow tiles to their common footprint, then composite one value per pixel.

        `align_spatial` first (strict — same CRS/resolution, tiles must
        overlap), then `mosaic_spatial` picks each pixel per `method`
        across the overlap. Output is one tile, same shape as any input —
        nothing grows in time.

        Args:
            *tiles: Tiles to merge, same CRS/resolution, must overlap.
            method: Overlap-resolution rule forwarded to mosaic_spatial.

        Returns:
            One composited GeoTile.

        Raises:
            ValueError: Tiles don't overlap or disagree on CRS/resolution (see align_spatial).
        """
        from .ops import align_spatial, mosaic_spatial

        return mosaic_spatial(*align_spatial(*tiles), method=method)

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
                ``extract_stem_dates`` — pass an explicit value to bypass
                that convention entirely.
            load_data: Materialise all pixels into memory; default lazy.
            bands: Band variable names to select; None keeps all.

        Raises:
            ValueError: If datetime is None and the filename has no standard
                date suffix.
        """
        p = Path(path)
        if datetime is None:
            datetime = extract_stem_dates(p.stem)

        data = from_geotiff(p, bands=bands)

        # from_geotiff never trusts a stored datetime — caller/filename
        # convention always wins, so `datetime` (not the stored tag's own)
        # feeds the rebuilt GeoTag below.
        raw_tag = data.attrs.get("tag")
        parsed = GeoTag.model_validate_json(raw_tag) if raw_tag else None
        tag = GeoTag(
            datetime=datetime,
            polygon=parsed.polygon if parsed else None,
            rgb_bands=parsed.rgb_bands if parsed else None,
            class_map=parsed.class_map if parsed else None,
            color_map=parsed.color_map if parsed else None,
            stac=_read_stac(p),
            **(parsed.model_extra or {} if parsed else {}),
        )

        if load_data:
            data = data.load()

        geobox = data.odc.geobox
        # GDAL rasters always carry a real band axis (unlike Zarr, where a single
        # unnamed variable can genuinely mean "no band dim") — always reconstitute one,
        # regardless of whether the original GeoTile.data had a 'band' dim or not.
        # to_array() stacks per-variable DataArrays into one — rio.nodata lives on each
        # variable individually and doesn't survive the stack, so re-attach it explicitly.
        nodata = next(iter(data.data_vars.values())).rio.nodata
        da = data.to_array(dim="band").transpose("band", "y", "x")
        if nodata is not None:
            da = da.rio.write_nodata(nodata)
        da = validate_da(da)
        return cls(geobox=geobox, data=da, geotag=tag)

    @classmethod
    def from_zarr(
        cls,
        path: str | Path,
        datetime: AnchorDatetime | None = None,
        load_data: bool = False,
        group: str | None = None,
    ) -> "GeoTile":
        """Create GeoTile from a Zarr store (or one group within it) written by to_zarr.

        Opened lazily by default. Geobox and metadata restored from store
        attrs. Datetime prefers the store's own `time` coordinate when
        present (start/end = min/max observed date — the authoritative
        source for a multi-step tile); falls back to the stored `tag`'s own
        datetime only when there's no time dimension to derive it from. This
        never guesses a date from the path.

        Args:
            path: Path to Zarr store.
            datetime: Explicit override — bypasses both the `time` coordinate
                and the stored tag's datetime entirely. None uses the store's own.
            load_data: Materialise all pixels into memory; default lazy.
            group: Zarr group to read; None reads the store root. GeoStack.from_zarr
                passes this to read one layer out of a multi-group store.

        Raises:
            ValueError: If datetime is None, the store has no time dimension,
                and no stored tag — was it written by GeoTile.to_zarr?
        """
        path = Path(path)
        ds = from_zarr(path, group=group)
        return cls._from_ds(ds, path, datetime, load_data, group, written_by="GeoTile.to_zarr")

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        datetime: AnchorDatetime | None = None,
        load_data: bool = False,
        group: str | None = None,
    ) -> "GeoTile":
        """Create GeoTile from a NetCDF store (or one group within it) written by to_netcdf.

        Same restore rules as from_zarr — same on-disk Dataset shape,
        different container format.

        Args:
            path: Path to NetCDF store.
            datetime: Explicit override — bypasses both the `time` coordinate
                and the stored tag's datetime entirely. None uses the store's own.
            load_data: Materialise all pixels into memory; default lazy.
            group: NetCDF4 group to read; None reads the file root.

        Raises:
            ValueError: If datetime is None, the store has no time dimension,
                and no stored tag — was it written by GeoTile.to_netcdf?
        """
        path = Path(path)
        ds = from_netcdf(path, group=group)
        return cls._from_ds(ds, path, datetime, load_data, group, written_by="GeoTile.to_netcdf")

    @classmethod
    def _from_ds(
        cls,
        ds: xr.Dataset,
        path: Path,
        datetime: AnchorDatetime | None,
        load_data: bool,
        group: str | None,
        written_by: str,
    ) -> "GeoTile":
        """Shared from_zarr/from_netcdf restore — same Dataset shape, different container.

        Args:
            ds: Dataset already read from disk (from_zarr/from_netcdf's own job).
            path: Store path — only used for the STAC sidecar lookup.
            written_by: Name for the "was this written by X?" error, matching the caller.

        Raises:
            ValueError: datetime is None, ds has no time dimension, and no stored tag.
        """
        geobox = ds.odc.geobox
        raw_tag = ds.attrs.get("tag")
        tag = GeoTag.model_validate_json(raw_tag) if raw_tag else None

        da = ds_to_da(ds)
        da = validate_da(da)

        if datetime is not None:
            resolved_datetime: AnchorDatetime = datetime
        elif "time" in da.dims:
            times = da.time.values
            resolved_datetime = (
                dt.fromisoformat(str(times.min().astype("datetime64[s]"))),
                dt.fromisoformat(str(times.max().astype("datetime64[s]"))),
            )
        elif tag is not None:
            resolved_datetime = tag.datetime
        else:
            raise ValueError(
                f"Dataset has no time dimension and no stored 'tag' datetime — was it written by {written_by}?"
            )

        if load_data:
            da = da.load()

        final_tag = GeoTag(
            datetime=resolved_datetime,
            polygon=tag.polygon if tag else None,
            rgb_bands=tag.rgb_bands if tag else None,
            class_map=tag.class_map if tag else None,
            color_map=tag.color_map if tag else None,
            stac=_read_stac(path, group=group),
            **(tag.model_extra or {} if tag else {}),
        )
        return cls(geobox=geobox, data=da, geotag=final_tag)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_geotiff(self, path: str | Path) -> list[Path]:
        """Write tile to GeoTIFF (one band per variable).

        A time-series tile writes one file per step instead of one — each
        step's compact date appended to `path`'s stem (`format_stem_dates`),
        each carrying just that step's own `geotag`/STAC sidecar.

        STAC provenance writes alongside as a `<stem>.stac.json` sidecar
        whenever a step actually carries any (`self.stac` non-empty) —
        no separate flag needed.

        Args:
            path: Output .tif path.

        Returns:
            Written path(s) — one entry for a single-step tile.
        """
        return self._write(path, driver="GTiff")

    def to_cog(self, path: str | Path) -> list[Path]:
        """Write tile to Cloud-Optimized GeoTIFF.

        A time-series tile writes one file per step instead of one — each
        step's compact date appended to `path`'s stem (`format_stem_dates`),
        each carrying just that step's own `geotag`/STAC sidecar.

        STAC provenance writes alongside as a `<stem>.stac.json` sidecar
        whenever a step actually carries any (`self.stac` non-empty) —
        no separate flag needed.

        Args:
            path: Output .tif path.

        Returns:
            Written path(s) — one entry for a single-step tile.
        """
        return self._write(path, driver="COG")

    def to_zarr(self, path: str | Path, chunk_px: int | None = 512) -> Path:
        """Write tile data including any time dimension to a Zarr store.

        `geotag` (datetime/metadata/polygon/plot_meta) stored as one store
        attribute. A multi-step tile's real per-step dates already live
        natively in its `time` coordinate, so `from_zarr` prefers deriving
        start/end from that over the stored tag's datetime. STAC provenance
        writes alongside as a `<stem>.stac.json` sidecar whenever this tile
        actually carries any (`self.stac` non-empty) — no separate flag needed.

        Args:
            path: Output Zarr store path.
            chunk_px: Spatial (y/x) on-disk chunk side length. None skips chunking.

        Returns:
            The written store path.
        """
        path = Path(path)
        tag = self.geotag.model_dump_json(exclude_none=True)
        ds = da_to_ds(self.data).assign_attrs(tag=tag)
        path = to_zarr(path, ds, chunk_px=chunk_px)
        _write_stac(self.stac, path)
        return path

    def to_netcdf(self, path: str | Path) -> Path:
        """Write tile data including any time dimension to a NetCDF store.

        Same shape as to_zarr — `geotag` as one store attribute, STAC
        provenance as a `<stem>.stac.json` sidecar when present.

        Args:
            path: Output NetCDF path.

        Returns:
            The written store path.
        """
        path = Path(path)
        tag = self.geotag.model_dump_json(exclude_none=True)
        ds = da_to_ds(self.data).assign_attrs(tag=tag)
        path = to_netcdf(path, ds)
        _write_stac(self.stac, path)
        return path

    def _write(self, path: str | Path, driver: str) -> list[Path]:
        path = Path(path)
        if self.has_time:
            written: list[Path] = []
            for t in self.data.time.values:
                t_dt = dt.fromisoformat(str(t.astype("datetime64[s]")))
                step_path = path.with_stem(f"{path.stem}_{format_stem_dates((t_dt, t_dt))}")
                step_tile = self.rebase(data=self.data.sel(time=t), datetime=(t_dt, t_dt))
                written.extend(step_tile._write(step_path, driver))
            return written

        tags = {"tag": self.geotag.model_dump_json(exclude_none=True)}
        if "band" in self.data.dims:
            tags["bands"] = json.dumps(list(self.bands))
        written_path = to_geotiff(path, self.data, driver=driver, tags=tags)
        _write_stac(self.stac, written_path)
        return [written_path]


# ----------------------------------------------------------------------
# STAC provenance sidecar (<stem>.stac.json — a pystac ItemCollection)
# ----------------------------------------------------------------------

def _stac_sidecar(path: Path, group: str | None) -> Path:
    """Sidecar path for one tile's STAC items — `<group>.stac.json` inside a
    Zarr store when `group` is given (one store, several GeoStack layers),
    else `<stem>.stac.json` next to a standalone GeoTIFF/Zarr path."""
    return path / f"{group}.stac.json" if group else path.parent / f"{path.stem}.stac.json"


def _write_stac(items: list[Item], path: Path, group: str | None = None) -> None:
    """Write STAC items as pystac ItemCollection sidecar. No-op if empty."""
    if items:
        ItemCollection(items).save_object(str(_stac_sidecar(path, group)))


def _read_stac(path: Path, group: str | None = None) -> list[Item]:
    """Read a tile's STAC sidecar back into Items, or ``[]`` if none exists."""
    sidecar = _stac_sidecar(path, group)
    if not sidecar.exists():
        return []
    return list(ItemCollection.from_file(str(sidecar)))

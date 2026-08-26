"""GeoRaster: one big geospatial surface, windowed/composable pixel data. See GeoRaster for details."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Self, Unpack

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from rasterio.enums import Resampling

from geosave_engine.geodata.extensions import GeoExtension, StacItems, TilerMode, TilingInfo, TimeSpec, span_from_times
from geosave_engine.geodata.utils.array import (
    SELECTOR_METHODS,
    ReduceMethod,
    resample_time,
    da_to_cf,
    validate_spatial,
)
from geosave_engine.geodata.utils.array.cf import cf_flag_attrs
from geosave_engine.geodata.utils.datetime import AnchorDatetime, freq_offset
from geosave_engine.geodata.utils.io import (
    GROUPED_SUFFIXES,
    READERS,
    GeotiffOptions,
    NetcdfOptions,
    ZarrOptions,
    read_sidecar,
    to_geotiff,
    to_netcdf,
    to_zarr,
)
from geosave_engine.geodata.utils.spatial.align import validate_rasters
from geosave_engine.geodata.utils.spatial.geobox import geobox_matches
from geosave_engine.utils.fn import UNSET, Unset

from .anchor import GeoAnchor
from ._array import _SpatialArray
from .header import (
    AttrEncoding,
    GeoHeader,
    decode_attrs,
    encode_attrs,
)
from .context import ContextFn
from .tile import GeoTile
from .vector import GeoVector

if TYPE_CHECKING:
    from geosave_engine.geodata.utils.datetime import Freq

# Which raster wins where two overlap: "first" keeps self's own real pixels, "last" the last raster given.
MergeMethod = Literal["first", "last"]
# Axis `concat` joins on — the one axis the inputs are allowed to differ along.
ConcatDim = Literal["time", "band"]


def default_resampling(data: xr.DataArray) -> Resampling:
    """Pick a resampling method from an array's own dtype.

    Args:
        data: Array about to be warped.

    Returns:
        `Resampling.nearest` for an integer array — its values are classes,
        which interpolation would invent between — else `Resampling.bilinear`.
    """
    return Resampling.nearest if np.issubdtype(data.dtype, np.integer) else Resampling.bilinear


@dataclass(frozen=True, kw_only=True, eq=False)
class GeoRaster(_SpatialArray):
    """Big geospatial raster — pixel data not fully loaded into memory.

    Read windows, split into tiles, or merge several rasters. Fields are
    inherited from its private spatial-array base; `open` and every writer
    live here.
    """

    # --- Constructors ---

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        vector_path: str | Path | None = None,
        group: str | None = None,
        anchor: GeoAnchor | None = None,
        crs: str | None = None,
        timespan: AnchorDatetime | None = None,
        bands: tuple[str, ...] | None = None,
    ) -> GeoRaster:
        """Lazily open an existing raster file for windowed access — format from path's suffix.

        A `<stem>.vector.parquet` sidecar beside the file is read back as
        `vector`; missing means None, and `vector_path` overrides where it's
        looked for.

        Args:
            path: Raster file/store — any suffix in
                `geosave_engine.geodata.utils.io.READERS`.
            vector_path: Vector file to read as `vector`, for one kept off
                the sidecar path. Any suffix in
                `geosave_engine.geodata.utils.io.VECTOR_SUFFIXES`. None
                looks for the sidecar.
            group: Group to read inside the store, for the formats that
                hold several (`.zarr`, `.nc`). None reads the root.
            crs: CRS to stamp on before reading, for a file that has a
                grid but declares no CRS (common in NetCDF). Ignored when
                `anchor` is given.
            timespan: Time range to use instead of the file's own. Must still
                cover every time label the file carries.
            bands: Band names to keep, in this order. None keeps all.

        Returns:
            Open GeoRaster, pixels still on disk, `attrs` read off the
            file's own `.attrs`. A `bands` selection that drops a band the
            stored `render.rgb_bands` names clears those references, since
            the header could not otherwise load.

        Raises:
            ValueError: no reader is registered for `path`'s suffix,
                `group` was given for a format that has none, the file has
                no CRS and neither `anchor` nor `crs` was given,
                `anchor`'s grid doesn't match the file's shape, or `timespan`
                is narrower than the file's own time labels.
            FileNotFoundError: `vector_path` was given but isn't a file.
            KeyError: a name in `bands` isn't one of the file's bands.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        reader = READERS.get(suffix)
        if reader is None:
            raise ValueError(f"No reader for {suffix!r} — known suffixes: {sorted(READERS)}")
        if group is not None and suffix not in GROUPED_SUFFIXES:
            raise ValueError(f"group= only applies to {sorted(GROUPED_SUFFIXES)}, not {suffix!r}")
        raw = reader(path, group=group) if group is not None else reader(path)

        if anchor is not None:
            grid = (raw.sizes.get("y"), raw.sizes.get("x"))
            if grid != anchor.shape:
                raise ValueError(f"anchor's grid {anchor.shape} doesn't match the file's own {grid}")
            raw = raw.assign_coords(xr_coords(anchor.geobox, always_yx=True))
        elif crs is not None:
            raw = raw.odc.assign_crs(crs)

        data = validate_spatial(raw)
        if bands is not None:
            data = data.sel(band=list(bands))

        # a vector file named outright has to exist
        if vector_path is not None:
            if not Path(vector_path).exists():
                raise FileNotFoundError(f"no vector file at {vector_path}")
            features: GeoVector | None = GeoVector.open(vector_path)
        else:
            gdf = read_sidecar(path, group)
            features = None if gdf is None else GeoVector(gdf=gdf)
        if features is None and anchor is not None:
            features = anchor.vector

        # readers hand back whatever sits on disk, so the header is decoded here, once
        foreign, header = decode_attrs(data.attrs)
        data.attrs = foreign
        resolved_anchor = GeoAnchor(geobox=data.odc.geobox, vector=features, header=header)
        if timespan is not None:
            resolved_anchor = resolved_anchor.rebase(timespan=timespan)
        return cls(data=data, anchor=resolved_anchor)

    # --- Transforms ---

    def rebase(
        self,
        *,
        data: xr.DataArray | Unset = UNSET,
        timespan: AnchorDatetime | None | Unset = UNSET,
        vector: GeoVector | None | Unset = UNSET,
        nodata: float | None | Unset = UNSET,
        **extensions: GeoExtension | Mapping[str, Any] | None,
    ) -> Self:
        """Return a raster with replaced pixels or metadata.

        `timespec` is deliberately absent: it records how an axis was
        actually bucketed, so only `resample_time` creates one and `concat`
        carries one forward.

        Args:
            data: New pixel array. Omitted keeps the current pixels.
            timespan: New recorded time span, or None to clear it.
            vector: Replacement features, or None to clear them.
            nodata: Replacement nodata value, or None to clear it.
            **extensions: Registered extension namespace updates.

        Returns:
            New GeoRaster with validated data and anchor metadata.
        """
        return self._rebase(
            data=data,
            timespan=timespan,
            vector=vector,
            nodata=nodata,
            extensions=extensions,
        )

    def reproject(
        self,
        target: GeoBox | str | None = None,
        resolution: float | None = None,
        resampling: Resampling | None = None,
    ) -> Self:
        """Put this raster on a different grid — another CRS, another resolution, or an exact grid.

        For a CRS target odc.geo picks the grid, snapping pixel edges to
        X=0/Y=0, so one crs/resolution always gives the same grid and
        rasters reprojected apart still line up. Lazy in, lazy out.

        Args:
            target: Grid to land on — a GeoBox to match another raster
                exactly (see `reproject_like`), a CRS string, or None to
                keep this raster's own CRS and change only `resolution`.
            resolution: Target pixel size in the target CRS's units. None
                lets odc.geo pick one keeping roughly the same pixel
                count. Ignored when `target` is a GeoBox.
            resampling: How pixels are interpolated onto the new grid. None
                picks by dtype — nearest for an integer raster (classes must
                stay whole), bilinear for a float one.

        Returns:
            New GeoRaster on the new grid, same bands, same time, `vector`
            moved onto the new CRS. Pixels the source doesn't reach come
            back as nodata, and `tiling` is cleared — it described the old
            grid. Returns self untouched for a GeoBox target this raster is
            already on.

        Raises:
            ValueError: `target` and `resolution` are both None, the
                resolved target has no CRS, or this raster's pixels are an
                integer type with no nodata declared, leaving no value to
                fill the new grid's uncovered corners with.

        Examples:
            >>> raster.reproject("EPSG:32633", resolution=10)
        """
        if target is None:
            if resolution is None:
                raise ValueError("reproject() needs a target grid/CRS and/or a resolution")
            target = self.crs
        if isinstance(target, GeoBox):
            if geobox_matches(self.data.odc.geobox, target):
                return self
            target_crs = None if target.crs is None else str(target.crs)
        else:
            target_crs = target

        if target_crs is None:
            raise ValueError("reproject() needs the target grid to have a CRS")

        if not isinstance(target, GeoBox) and resolution is not None:
            footprint = self.anchor.geobox.extent
            if footprint.crs != target_crs:
                footprint = footprint.to_crs(target_crs)
            target = GeoBox.from_geopolygon(footprint, resolution=resolution, anchor="edge")
            if geobox_matches(self.data.odc.geobox, target):
                return self

        if np.issubdtype(self.data.dtype, np.integer) and self.nodata is None:
            raise ValueError(
                "warping fills the parts the source doesn't reach with this raster's nodata, "
                "which isn't set — rebase(nodata=...) first"
            )

        warped = self.data.odc.reproject(
            target,
            resampling=resampling if resampling is not None else default_resampling(self.data),
            resolution="auto" if resolution is None else resolution,
        )
        # odc names a geographic grid's dims latitude/longitude; canonical Spatial data is y/x
        if "latitude" in warped.dims:
            warped = warped.rename({"latitude": "y", "longitude": "x"})
        # Floating rasters use NaN for uncovered pixels when no sentinel was declared.
        nodata = self.nodata
        if nodata is None and np.issubdtype(self.data.dtype, np.floating):
            nodata = np.nan
        return self._rebase(
            data=warped.rio.write_nodata(nodata),
            vector=self.vector.to_crs(target_crs) if self.vector is not None else None,
            extensions={"tiling": None},
        )

    def reproject_like(self, other: GeoRaster, resampling: Resampling | None = None) -> Self:
        """Put this raster on another raster's exact grid, ready to stack with it.

        Args:
            other: Raster whose geobox to land on.
            resampling: How pixels are interpolated onto the new grid. None
                picks by dtype — nearest for an integer raster, bilinear
                for a float one.

        Returns:
            New GeoRaster on `other`'s grid. Ground `other` covers but this
            raster doesn't comes back as nodata; `tiling` is cleared. Self
            untouched when already on that grid.

        Raises:
            ValueError: `other`'s grid has no CRS, or this raster's pixels
                are an integer type declaring no nodata to fill uncovered
                ground with.

        Examples:
            >>> GeoStack(image=s2, dem=dem.reproject_like(s2))
        """
        return self.reproject(other.anchor.geobox, resampling=resampling)

    def resample_time(
        self,
        freq: Freq,
        method: ReduceMethod | Callable[..., xr.DataArray] = "last",
        *,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | dt | Unset = UNSET,
        offset: str | timedelta | None = None,
        restore_coord_dims: bool = False,
    ) -> Self:
        """Bucket this raster's time steps onto a coarser, fixed cadence.

        Only buckets something landed in become steps: an unobserved month is
        absent, never a fabricated step of solid nodata. Steps therefore carry
        real pixels but need not sit one `freq` apart.

        Args:
            freq: Target cadence — any pandas offset alias, e.g. `"5D"`,
                `"ME"` (month end), `"MS"` (month start).
            method: How each bucket's steps reduce to one — a named
                reducer ("last", "mean", "median", ...) or a callable
                taking the bucket's own DataArray. The default picks an
                observed value rather than deriving one, so a categorical
                layer keeps real class codes.
            closed: Which bucket edge is inclusive. None uses pandas' default for `freq`.
            label: Which bucket edge labels the result. None uses pandas' default for `freq`.
            origin: Bucket grid origin, forwarded to xarray's resample.
                Unset uses the declared span's start, else `"start_day"`.
            offset: Shift applied to the bucket grid, forwarded to xarray's resample.
            restore_coord_dims: Forwarded to xarray's resample.

        Returns:
            New instance holding one step per observed bucket, same geobox,
            same bands, same dtype. Its `timespec` records the resolved bucket
            grid, and its `timespan` runs from the first bucket's start to the
            last one's end, not the labels.

        Raises:
            ValueError: this raster has no time dim, `freq` isn't a pandas
                offset alias, or `method` is a string outside `ReduceMethod`.

        Examples:
            >>> raster.timespan                      # a year, 9 months of it cloud-free
            (datetime(2024, 1, 1, 0, 0), datetime(2024, 12, 31, 23, 59, 59, 999999))
            >>> raster.resample_time("MS", "median").data.sizes["time"]
            9
        """
        if not self.has_time:
            raise ValueError("resample_time() needs a raster with a time dim")
        declared_span = self.timespan
        if isinstance(origin, Unset):
            origin = "start_day" if declared_span is None else declared_span[0]

        data = resample_time(
            self.data,
            freq,
            method,
            closed=closed,
            label=label,
            origin=origin,
            offset=offset,
            restore_coord_dims=restore_coord_dims,
        )
        # record the resolved bucket grid, read off the times as they were before bucketing
        spec = TimeSpec.from_resample(
            self.data.time.values,
            freq,
            method=method if isinstance(method, str) else None,
            closed=closed,
            label=label,
            origin=origin,
            offset=offset,
        )
        # count observations per bucket on a 1-D indicator, binned exactly as the pixels were
        occupied = (
            xr.DataArray(
                np.ones(self.data.sizes["time"], dtype="int8"),
                dims="time",
                coords={"time": self.data.time.values},
            )
            .resample(time=freq_offset(freq), closed=closed, label=label, origin=origin, offset=offset)
            .count()
        )
        # resample emits every bucket between the first and last observation, unobserved ones as solid nodata
        data = data.sel(time=occupied.time.values[occupied.values > 0])
        # a selector picks real values, so the promoting NaN left with its step; only undeclared nodata reaches here
        if data.dtype != self.dtype and isinstance(method, str) and method in SELECTOR_METHODS:
            data = data.rio.write_nodata(None).astype(self.dtype)

        # take the span from the bucket edges, since the labels only name each bucket
        spans = spec.bounds(data.time.values)
        return self._rebase(data=data, timespan=(spans[0][0], spans[-1][1]), timespec=spec)

    def time_windows(self, length: int, stride: int | None = None) -> Iterator[Self]:
        """Cut consecutive time steps into fixed-length windows.

        Time-axis counterpart of `tiles`. Each window's `time` span comes
        from its own labels' bucket bounds, and its `stac` provenance
        narrows to that span.

        Args:
            length: Time steps per window.
            stride: Steps between consecutive window starts. None uses
                `length`, so windows don't overlap.

        Yields:
            One raster per window, in time order, each holding `length`
            steps. A trailing remainder shorter than `length` is dropped.

        Raises:
            ValueError: This raster has no time dim, `length` or `stride`
                isn't positive, or `length` is longer than the time axis.

        Examples:
            >>> monthly = raster.resample_time("MS", "median")
            >>> [window.time for window in monthly.time_windows(4, stride=1)][0]
            (datetime(2024, 1, 1, 0, 0), datetime(2024, 4, 30, 23, 59, 59, 999999))
        """
        if not self.has_time:
            raise ValueError("time_windows() needs a raster with a time dim")
        if length < 1:
            raise ValueError(f"length must be positive, got {length}")
        step = length if stride is None else stride
        if step < 1:
            raise ValueError(f"stride must be positive, got {stride}")

        steps = self.data.sizes["time"]
        if length > steps:
            raise ValueError(f"length={length} needs more than this raster's own {steps} time step(s)")

        provenance = self.header.extensions.get(StacItems.NAMESPACE)
        for start in range(0, steps - length + 1, step):
            window = self.data.isel(time=slice(start, start + length))
            span = span_from_times(window.time.values, self.header.timespec)
            narrowed: dict[str, Any] = {}
            if isinstance(provenance, StacItems):
                narrowed["stac"] = {"items": provenance.between(*span).items}
            yield self._rebase(data=window, timespan=span, extensions=narrowed)

    def rename_bands(self, mapping: dict[str, str]) -> Self:
        """Rename bands without changing their order or pixels.

        Args:
            mapping: Existing name to replacement name. Names not present
                in the mapping stay unchanged.

        Returns:
            New raster carrying the renamed band coordinate. Render RGB
            band references are renamed alongside it.

        Raises:
            ValueError: A replacement is empty or the result would
                contain duplicate names.
            KeyError: A source name is not present on this raster.

        Examples:
            >>> raster.rename_bands({"1": "red", "2": "nir"})
        """
        if not mapping:
            return self
        if not all(isinstance(source, str) and isinstance(target, str) for source, target in mapping.items()):
            raise TypeError("rename_bands() mapping keys and values must be strings")

        unknown = sorted(set(mapping) - set(self.bands))
        if unknown:
            raise KeyError(f"bands {unknown} aren't in this raster's {list(self.bands)}")

        replacements = {source: target.strip() for source, target in mapping.items()}
        empty = sorted(source for source, target in replacements.items() if not target)
        if empty:
            raise ValueError(f"replacement band names must not be empty, got empty values for {empty}")

        names = [replacements.get(name, name) for name in self.bands]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Band names must be unique, repeated after rename: {repeated}")

        data = self.data.assign_coords(band=names)
        if self.render is not None and self.render.rgb_bands is not None:
            rgb_bands = tuple(replacements.get(name, name) for name in self.render.rgb_bands)
            return self._rebase(data=data, extensions={"render": {"rgb_bands": rgb_bands}})
        return self._rebase(data=data)

    def merge_spatial(
        self,
        *others: GeoRaster,
        method: MergeMethod = "first",
    ) -> GeoRaster:
        """Combine grid-compatible rasters covering different footprints.

        Thin wrapper over `GeoMosaic`, which takes rasters one at a time
        when a whole run shouldn't be held at once. Lazy end to end for
        dask-backed inputs. Touches `y`/`x` only, never `time`.

        Args:
            *others: Rasters to merge in, in priority order (see `method`).
            method: Overlap rule — "first" keeps self's own real data,
                "last" prefers the last raster given.

        Returns:
            One GeoRaster covering the union footprint. Pixels no input
            covers come back as nodata. Tags, extensions and `timespec`
            are self's; `vector` is every input's combined; `tiling` is
            cleared.

        Raises:
            ValueError: This raster declares no nodata, or inputs disagree
                on dtype, nodata, bands, time, CRS, resolution,
                orientation, or pixel-grid origin.
            TypeError: An input is not a GeoRaster.
        """
        from .mosaic import GeoMosaic

        mosaic = GeoMosaic(method=method)
        mosaic.add(self, *others)
        return mosaic.result()

    def concat(
        self,
        *others: GeoRaster,
        dim: ConcatDim,
    ) -> GeoRaster:
        """Combine rasters that split one footprint across `time` or `band` into a whole.

        Every axis but `dim` has to already agree. A coordinate landing in two
        inputs is rejected — which pixels win would come down to argument order.
        For different *footprints* use `merge_spatial`, which resolves overlap.

        Args:
            *others: Rasters to combine in. For `dim="time"` each one's own
                timestamps decide placement, not argument order; for
                `dim="band"` bands land in the order given.
            dim: Axis to join on — `"time"` puts separate acquisitions on
                one axis, `"band"` puts a per-band GeoTIFF split back together.
        Returns:
            One GeoRaster carrying every input's steps or bands. Tags and
            extensions are self's; `vector` is every input's combined;
            `tiling` is cleared. For `dim="time"` the result is
            chronological, its span re-read off the merged labels, and
            `timespec` survives only if every input agreed on it. `self`
            when no other was given.

        Raises:
            ValueError: Inputs disagree on their grid, dtype, nodata, or
                untouched axis; an input lacks the requested axis; or a
                timestamp/band name occurs in more than one input.
            TypeError: An input is not a GeoRaster.

        Examples:
            >>> january.concat(february, march, dim="time")
            >>> red.concat(green, blue, dim="band")
        """
        if dim not in ("time", "band"):
            raise ValueError(f"dim must be 'time' or 'band', got {dim!r}")
        for position, raster in enumerate(others):
            if not isinstance(raster, GeoRaster):
                raise TypeError(
                    f"concat() expects GeoRaster inputs; argument {position} is {type(raster).__name__}"
                )
        if not others:
            return self
        if dim == "time":
            return self._concat_time(*others)
        return self._concat_band(*others)

    def _concat_time(
        self,
        *others: GeoRaster,
    ) -> GeoRaster:
        """Join rasters along `time` — see `concat`.

        Args:
            *others: Rasters to join, at least one.
        Returns:
            GeoRaster spanning every input's timestamps, chronological.

        Raises:
            ValueError: an input has no `time` dimension, a timestamp
                occurs more than once, or an input disagrees on grid,
                bands, dtype, or nodata.
        """
        steps = (self, *others)
        timeless = [raster.stem for raster in steps if not raster.has_time]
        if timeless:
            raise ValueError(
                f"concat(dim='time') requires every raster to have a time dimension; missing on {timeless}. "
                "Add explicit time coordinates before concatenating"
            )
        validate_rasters(*steps, grid=True, bands=True, operation="concat(dim='time')")

        labels, counts = np.unique(np.concatenate([raster.data.time.values for raster in steps]), return_counts=True)
        if (counts > 1).any():
            raise ValueError(f"timestamps land in more than one raster: {list(labels[counts > 1])}")

        merged = xr.concat([raster.data for raster in steps], dim="time", join="exact").sortby("time")
        return steps[0]._rebase(
            data=merged,
            header=GeoHeader.combine(*(raster.header for raster in steps)),
            vector=GeoVector.concat(*(raster.vector for raster in steps)),
        )

    def _concat_band(
        self,
        *others: GeoRaster,
    ) -> GeoRaster:
        """Join rasters along `band` — see `concat`.

        Args:
            *others: Rasters to join, at least one.
        Returns:
            GeoRaster carrying every input's bands, in the order given.

        Raises:
            ValueError: an input has no `band` dim, a band name lands in
                more than one input, or an input disagrees on grid,
                timestamps, dtype, or nodata.
        """
        layers = (self, *others)
        validate_rasters(*layers, grid=True, times=True, operation="concat(dim='band')")

        names = [band for raster in layers for band in raster.bands]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"band names land in more than one raster: {repeated}")

        merged = xr.concat([raster.data for raster in layers], dim="band", join="exact")
        return layers[0]._rebase(
            data=merged,
            header=GeoHeader.combine(*(raster.header for raster in layers)),
            vector=GeoVector.concat(*(raster.vector for raster in layers)),
        )

    # --- Windowing ---

    def crop(self, geobox: GeoBox) -> GeoRaster:
        """Cut this raster down to a window already on its own pixel grid.

        No resampling, so `geobox` has to share this raster's CRS,
        resolution and pixel-grid phase — reproject onto it first
        otherwise. Lazy in, lazy out.

        Args:
            geobox: Window to keep. Must sit fully inside this raster's
                own geobox and land on its pixel grid.

        Returns:
            New GeoRaster on `geobox`, this raster's header with `tiling`
            cleared — it described the whole surface — and its `vector`
            filtered to the window (None if nothing intersects). Features
            are kept whole, so one may overhang the window — see
            `GeoVector.filter`.

        Raises:
            ValueError: `geobox` isn't on this raster's pixel grid, or
                isn't fully inside its extent.

        Examples:
            >>> raster.crop(anchor.geobox)
            >>> raster.reproject(other.anchor.geobox).crop(other.anchor.geobox)
        """
        try:
            rows, cols = self.data.odc.geobox.overlap_roi(geobox)
        except ValueError as e:
            raise ValueError(
                f"crop() needs a window on this raster's own pixel grid ({e}) — "
                "call reproject(geobox) first"
            ) from e
        if rows.stop - rows.start != geobox.height or cols.stop - cols.start != geobox.width:
            raise ValueError(
                f"crop()'s {(geobox.height, geobox.width)} window isn't fully inside this "
                f"raster's {self.shape}"
            )

        features = self.vector.filter(geobox) if self.vector is not None else None
        window = self.data.isel(y=rows, x=cols)
        return self._rebase(data=window, vector=features, extensions={"tiling": None})

    def to_tile(self) -> GeoTile:
        """Read this surface as one window — same pixels, same header, same vector.

        The inverse of `GeoTile.to_raster`. Both are pure relabelings: the
        disk boundary lives on GeoRaster, windowing on GeoTile. Nothing
        here checks that the pixels fit in memory.

        Returns:
            GeoTile over the same data and anchor.
        """
        return GeoTile(data=self.data, anchor=self.anchor)

    def tiles(
        self,
        tile_size_px: int | None = None,
        stride_px: int | None = None,
        overlap: int | float | tuple[int, int] | None = None,
        mode: TilerMode = "reflect",
        vector: bool = True,
        *,
        name: str | None = None,
        context_fn: ContextFn | None = None,
    ) -> Iterator[GeoTile]:
        """Window this raster's own pixels into square tiles.

        Every tile carries a `tiling` stamp so `from_tiles` can put them
        back. Its `group_id` derives from the grid, tiling config and time
        span, so the same cut on another worker produces the same group.

        Args:
            tile_size_px: Window side length in pixels (square), before
                edge handling. None uses the shorter of the two axes.
            stride_px: Distance between consecutive window origins. None = tile_size_px.
            overlap: Forwarded to tiler.Tiler's own overlap kwarg (int px,
                float [0,1) fraction, or (row, col) tuple). Wins over
                stride_px when both are given.
            mode: How a trailing window's overhang is filled — "reflect"
                mirrors, "edge" repeats the last pixel, "constant" fills
                with this raster's own nodata.
            vector: True (default) gives each tile this raster's `vector`
                filtered to its window, features kept whole. False yields
                tiles with no vector.
            name: Extra text folded into the derived `group_id`, separating
                two otherwise identical cuts — e.g. two models over one
                surface. None derives from the cut alone.
            context_fn: Called once per window with that window's own anchor,
                whose header carries the window's bands and steps; its result
                becomes that tile's `model_context`. None leaves it unset, and
                an encoder derives its own at forward time.

        Yields:
            One GeoTile per window position, in row-major order. Lazy in,
            lazy out — no pixel is read here.

        Raises:
            ValueError: `tile_size_px` isn't positive, `stride_px` isn't
                positive or is wider than the tile, or `mode` is invalid.
        """
        height, width = self.shape
        side = tile_size_px if tile_size_px is not None else min(height, width)
        if side <= 0:
            raise ValueError(f"tile_size_px must be positive, got {side}")
        if mode not in ("reflect", "edge", "constant"):
            raise ValueError(f"mode must be 'reflect', 'edge', or 'constant', got {mode!r}")

        if overlap is None:
            if stride_px is not None and not 0 < stride_px <= side:
                raise ValueError(f"stride_px must be in 1..{side} (the tile side), got {stride_px}")
            overlap = 0 if stride_px is None else side - stride_px

        if mode == "constant" and self.nodata is None:
            raise ValueError("mode='constant' fills with this raster's nodata, which isn't set — rebase(nodata=...) first")

        # one stamp describes the whole cut; each tile takes a copy carrying its own position
        cut = TilingInfo.from_grid(
            self.data.odc.geobox,
            tile_shape=(side, side),
            overlap=overlap,
            mode=mode,
            time=self.timespan,
            name=name,
        )
        tiler = cut.tiler()

        # grow the far edge so a trailing window still has real pixels to read
        data, geobox = self.data, self.data.odc.geobox
        last_row, last_col = (int(axis) for axis in tiler.get_tile_bbox(len(tiler) - 1)[0])
        pad_y = max(0, last_row + side - height)
        pad_x = max(0, last_col + side - width)
        if pad_y or pad_x:
            geobox = GeoBox(shape=(height + pad_y, width + pad_x), affine=geobox.affine, crs=geobox.crs)
            grown = (
                data.pad(y=(0, pad_y), x=(0, pad_x), mode="constant", constant_values=self.nodata)
                if mode == "constant"
                else data.pad(y=(0, pad_y), x=(0, pad_x), mode=mode)
            )
            # restamp y/x off the grown grid — pad mirrors coord labels along with pixels
            data = grown.assign_coords(xr_coords(geobox, always_yx=True))

        for tile_id in range(len(tiler)):
            row, col = (int(axis) for axis in tiler.get_tile_bbox(tile_id)[0])
            window = data.isel(y=slice(row, row + side), x=slice(col, col + side))
            features = self.vector.filter(window.odc.geobox) if vector and self.vector is not None else None
            anchor = self.anchor.rebase(
                geobox=window.odc.geobox,
                vector=features,
                tiling=cut.model_copy(update={"tile_id": tile_id}),
            )
            tile = GeoTile(data=window, anchor=anchor)
            if context_fn is None:
                yield tile
                continue
            # the built tile's anchor, not the pre-build one — its header carries the window's own steps
            context = context_fn(tile.anchor)
            yield tile if context is None else replace(tile, model_context=dict(context))

    # --- Persistence ---

    def to_zarr(
        self,
        path: str | Path,
        chunk_px: int | None = 512,
        progress: bool = True,
        **options: Unpack[ZarrOptions],
    ) -> Path:
        """Write a CF-compliant Zarr store — one variable per band.

        A `vector` is written beside the store as `<stem>.vector.parquet`,
        or `<stem>.<group>.vector.parquet` when a group is named.

        Args:
            path: Output `.zarr` store path.
            chunk_px: Spatial (y/x) chunk side length. `time` is never split.
            progress: Show a dask progress bar while pixels compute.
            **options: Passed to `xarray.Dataset.to_zarr` — group, spec
                version, encoding. See `ZarrOptions`.

        Returns:
            The written store path.

        Raises:
            ValueError: An option is one the writer sets itself or cannot
                forward — see `ZarrOptions`.

        Examples:
            >>> raster.to_zarr("scene.zarr", encoding={"B04": {"compressors": None}})
        """
        # compute would bind to the adapter's own parameter and return a Delayed, not a Path
        if "compute" in options:
            raise ValueError(
                "to_zarr() always writes — GeoStack.to_zarr batches its layers into one pass"
            )
        return to_zarr(
            path,
            self._cf_encoded("json"),
            vector=None if self.vector is None else self.vector.gdf,
            chunk_px=chunk_px,
            progress=progress,
            **options,
        )

    def to_netcdf(
        self,
        path: str | Path,
        chunk_px: int | None = 512,
        progress: bool = True,
        **options: Unpack[NetcdfOptions],
    ) -> Path:
        """Write a CF-compliant NetCDF file — one variable per band.

        A `vector` is written beside it as `<stem>.vector.parquet`, or as
        `<stem>.<group>.vector.parquet` when a group is named.

        Args:
            path: Output `.nc` path.
            chunk_px: Spatial (y/x) chunk side length. `time` is never split.
            progress: Show a dask progress bar while pixels compute.
            **options: Passed to `xarray.Dataset.to_netcdf` — group, format
                version, encoding. See `NetcdfOptions`.

        Returns:
            The written path.

        Raises:
            ValueError: An option is one the writer sets itself or cannot
                forward — see `NetcdfOptions`.
            FileExistsError: The named group already exists — NetCDF groups
                cannot be replaced without rebuilding the whole file.
        """
        return to_netcdf(
            path,
            self._cf_encoded("text"),
            vector=None if self.vector is None else self.vector.gdf,
            chunk_px=chunk_px,
            progress=progress,
            **options,
        )

    def _cf_encoded(self, encoding: AttrEncoding) -> xr.Dataset:
        """This raster in CF form, its header encoded for one store's attrs.

        Args:
            encoding: What the target store's attrs may hold — `"json"`
                for zarr, `"text"` for netcdf.

        Returns:
            Dataset in CF form, one variable per band, `.attrs` carrying
            this raster's header written the way `encoding` names.
        """
        dataset = da_to_cf(self.data)
        flag_attrs = cf_flag_attrs(self.legend)
        if flag_attrs:
            dataset = dataset.assign_attrs(flag_attrs)
        dataset.attrs = encode_attrs(dataset.attrs, self.header, encoding)
        return dataset

    def to_geotiff(
        self,
        path: str | Path,
        time: dt | np.datetime64 | str | None = None,
        chunk_px: int | None = 512,
        progress: bool = True,
        **options: Unpack[GeotiffOptions],
    ) -> Path:
        """Write this raster as one GeoTIFF.

        Attrs become flat GDAL tags; CF-only `flag_*` attrs are dropped. A
        `vector` is written beside it as `<stem>.vector.parquet`.

        Args:
            path: Output `.tif` path.
            time: Which step to write, for a raster with a `time` dim — one of
                its own timestamps, as a datetime or a string xarray can match
                (`"2024-01-15"`). Must name exactly one step. None is only
                valid for a raster with no `time` dim.
            chunk_px: On-disk block side length. None leaves GTiff untiled.
            progress: Show a dask progress bar while pixels compute.
            **options: Passed to `rioxarray.rio.to_raster` — compression,
                threads, tags, any GDAL creation option. See `GeotiffOptions`.

        Returns:
            The written path.

        Raises:
            ValueError: this raster has a `time` dim and `time` wasn't given,
                `time` was given for a raster with no `time` dim, `time` names
                no step or more than one, or an option is one the writer sets.

        Examples:
            >>> raster.to_geotiff("scene.tif", time="2024-01-15", compress="ZSTD")
        """
        return self._write_tiff(path, time, chunk_px, progress, options)

    def to_cog(
        self,
        path: str | Path,
        time: dt | np.datetime64 | str | None = None,
        chunk_px: int | None = 512,
        progress: bool = True,
        **options: Unpack[GeotiffOptions],
    ) -> Path:
        """Write this raster as one cloud-optimized GeoTIFF — same contract as `to_geotiff`.

        Through GDAL's COG driver, which self-tiles and adds overviews, so the
        file is range-readable off object storage.

        Args:
            path: Output `.tif` path.
            time: Which step to write, for a raster with a `time` dim. Must
                name exactly one. None is only valid for a timeless raster.
            chunk_px: On-disk block side length. None leaves the COG driver's
                own default alone.
            progress: Show a dask progress bar while pixels compute.
            **options: Passed to `rioxarray.rio.to_raster`. `driver` is set to
                `"COG"` here. See `GeotiffOptions`.

        Returns:
            The written path.

        Raises:
            ValueError: this raster has a `time` dim and `time` wasn't given,
                `time` was given for a raster with no `time` dim, `time` names
                no step or more than one, or an option is one the writer sets.
        """
        return self._write_tiff(path, time, chunk_px, progress, {**options, "driver": "COG"})

    def _write_tiff(
        self,
        path: str | Path,
        time: dt | np.datetime64 | str | None,
        chunk_px: int | None,
        progress: bool,
        options: GeotiffOptions,
    ) -> Path:
        """Write one GDAL-driver TIFF plus its vector — shared by `to_geotiff` and `to_cog`.

        Args:
            path: Output `.tif` path.
            time: Step to write, for a raster with a `time` dim. Must name
                exactly one. None is only valid for a timeless raster.
            chunk_px: On-disk block side length, or None for the driver's own.
            progress: Show a dask progress bar while pixels compute.
            options: Writer options, already carrying the resolved driver.

        Returns:
            The written path.

        Raises:
            ValueError: `time` is missing, given for a timeless raster, or
                names no step or more than one.
        """
        path = Path(path)
        data, anchor = self.data, self.anchor

        # GeoTIFF carries no time axis, so select the one step named by the caller
        if self.has_time:
            if time is None:
                steps = self.data.time.values
                raise ValueError(
                    f"GeoTIFF holds no time axis — name one of this raster's {len(steps)} steps with time=, "
                    f"e.g. time={str(steps[0].astype('datetime64[us]').item())!r}"
                )
            try:
                selected = data.sel(time=time)
            except KeyError as e:
                raise ValueError(f"time={time!r} isn't one of this raster's steps: {list(self.data.time.values)}") from e
            if "time" in selected.dims:
                raise ValueError(f"time={time!r} names {selected.sizes['time']} of this raster's steps, needs exactly one")
            instant = selected.time.values.astype("datetime64[us]").item()
            data = selected.drop_vars("time")
            anchor = anchor.rebase(timespan=(instant, instant))
        elif time is not None:
            raise ValueError("time= names a step to write, but this raster has no time dim")

        # GDAL tags hold strings only, and a GeoTIFF carries no CF legend to mirror the header into
        stamped = data.assign_attrs(encode_attrs(data.attrs, anchor.header, "text"))

        return to_geotiff(
            path,
            stamped,
            vector=None if self.vector is None else self.vector.gdf,
            chunk_px=chunk_px,
            progress=progress,
            **options,
        )


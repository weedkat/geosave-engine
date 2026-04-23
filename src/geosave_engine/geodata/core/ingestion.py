"""Base ingestor for STAC-backed raster data sources."""
from __future__ import annotations

import abc
import logging
import time
import rioxarray  # noqa: F401
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pystac
import xarray as xr
from dask.diagnostics import ProgressBar
from odc.geo.geobox import GeoBox
from odc.stac import load as odc_load
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from geosave_engine.geodata.core.client import BaseStacClient
from geosave_engine.geodata.core.query import BaseStacQuery
from geosave_engine.geodata.core.types import GeoTiffOptions, LoadRequest
from geosave_engine.utils.geodata.datetime import datetime_range_buffer
from geosave_engine.utils.geodata.tiff import TiffMetadata, read_tiff_metadata


log = logging.getLogger(__name__)

_UNRECOVERABLE = ("Read failed", "opj_get_decoded_tile")
_MAX_ATTEMPTS  = 3


def _utm_crs_from_bbox(bbox: tuple[float, float, float, float]) -> CRS:
    west, south, east, north = bbox
    lon = (west + east) / 2.0
    lat = (south + north) / 2.0
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    if not utm_crs_list:
        raise ValueError(f"could not resolve a UTM CRS for bbox={bbox!r}")
    return CRS.from_epsg(int(utm_crs_list[0].code))


class BaseIngestor(abc.ABC):
    """Generic base for any STAC-backed raster ingestor (Sentinel-2, Landsat, DEM, ...)."""

    COLLECTION: ClassVar[str]

    def __init__(self, client: BaseStacClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        query: BaseStacQuery,
        *,
        bands: list[str],
        crs: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        resolution: int = 10,
        as_reflectance: bool = True,
    ) -> xr.DataArray | None:
        """Search and load a (band, y, x) DataArray for a single STAC query.

        All items are mosaicked so samples at tile boundaries receive data from
        both tiles. Retry logic is handled internally (up to 3 attempts).

        ``crs`` and ``bounds`` pin the output to an exact pixel grid. When
        omitted, a UTM CRS is derived from ``query.bbox``.

        Returns ``None`` when no items are found or an unrecoverable error occurs.

        The returned DataArray carries provenance in ``.attrs``:
          - ``item_ids``         — list of STAC item IDs
          - ``sensing_datetime`` — ISO-8601 datetime string
          - ``sun_azimuth``      — solar azimuth angle (degrees)
        """
        items = self._client.search(query)
        if not items:
            log.info("query=%s -- no items found", query)
            return None
        if query.bbox is None:
            raise ValueError("query.bbox is required for loading")
        request = LoadRequest(
            bbox=tuple(query.bbox),
            bands=bands,
            crs=crs,
            bounds=bounds,
            resolution=resolution,
            as_reflectance=as_reflectance,
        )
        return self._load_with_retry(items, request)

    def from_tiff(
        self,
        tiff_path: Path,
        *,
        bands: list[str],
        date_window: timedelta = timedelta(days=1),
        resolution: int = 10,
        as_reflectance: bool = True,
    ) -> tuple[xr.DataArray, TiffMetadata] | None:
        """Load data aligned to a TIFF's spatial grid, returning (DataArray, TiffMetadata).

        Datetime is parsed from the TIFF filename (expects ``…-YYYYMMDD.tif``).
        CRS and bounds are read from the raster. The STAC query is built internally
        using ``self.COLLECTION`` — no caller-supplied query needed.

        Returns ``None`` when no STAC items are found.
        """
        meta     = read_tiff_metadata(tiff_path)
        dt_range = datetime_range_buffer(
            meta.datetime,
            delta_before=date_window,
            delta_after=date_window,
        )
        query = BaseStacQuery(
            collections=[self.COLLECTION],
            bbox=meta.geometry.bounds,
            datetime=dt_range,
        )
        da = self.load(
            query,
            crs=meta.crs,
            bounds=meta.bounds,
            bands=bands,
            resolution=resolution,
            as_reflectance=as_reflectance,
        )
        return None if da is None else (da, meta)

    def save(
        self,
        da: xr.DataArray,
        path: Path,
        options: GeoTiffOptions = GeoTiffOptions(),
    ) -> None:
        """Write a DataArray to a compressed tiled GeoTIFF."""
        predictor = options.predictor
        if predictor is None:
            predictor = 3 if np.issubdtype(da.dtype, np.floating) else 2
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        da.rio.to_raster(
            path,
            compress=options.compress,
            predictor=predictor,
            tiled=options.tiled,
        )

    # ------------------------------------------------------------------
    # Radiometry hook — override per product/provider
    # ------------------------------------------------------------------

    def _apply_radiometry(
        self,
        da: xr.DataArray,
        items: list[pystac.Item],
    ) -> xr.DataArray:
        """Convert DN to physical values. Default is a passthrough.

        Subclasses override this to apply product-specific DN→reflectance (or
        DN→backscatter, DN→elevation, etc.) using scale/offset read from STAC
        item metadata — no hardcoded values.
        """
        return da

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_with_retry(
        self,
        items: list[pystac.Item],
        request: LoadRequest,
    ) -> xr.DataArray | None:
        lazy = self._build_lazy(items, request)
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                with ProgressBar():
                    return lazy.compute()
            except Exception as exc:
                msg = str(exc)
                if any(tag in msg for tag in _UNRECOVERABLE):
                    log.warning("Unrecoverable JP2 error (skip): %s", exc)
                    return None
                if attempt == _MAX_ATTEMPTS:
                    log.warning("Load failed after %d attempts (skip): %s", _MAX_ATTEMPTS, exc)
                    return None
                log.debug("Retry %d/%d: %s", attempt, _MAX_ATTEMPTS, exc)
                time.sleep(2 ** attempt)

    def _build_lazy(
        self,
        items: list[pystac.Item],
        request: LoadRequest,
    ) -> xr.DataArray:
        resolved_crs = request.crs or _utm_crs_from_bbox(request.bbox)
        if request.bounds:
            geobox = GeoBox.from_bbox(request.bounds, crs=resolved_crs, resolution=request.resolution)
            ds = odc_load(
                items, bands=request.bands, geobox=geobox,
                resampling="bilinear", chunks={"x": 1024, "y": 1024},
            )
        else:
            ds = odc_load(
                items, bands=request.bands, bbox=request.bbox, crs=resolved_crs,
                resolution=request.resolution,
                resampling="bilinear", chunks={"x": 1024, "y": 1024},
            )
        da = ds.to_array(dim="band")

        time_dims = [d for d in da.dims if d not in ("band", "y", "x")]
        if time_dims:
            da = da.isel({d: 0 for d in time_dims}, drop=True)

        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        da.attrs.update({
            "item_ids":         [i.id for i in items],
            "sensing_datetime": items[0].properties.get("datetime"),
            "sun_azimuth":      float(items[0].properties.get("view:sun_azimuth", 0.0)),
        })

        if request.as_reflectance:
            da = self._apply_radiometry(da, items)

        return da

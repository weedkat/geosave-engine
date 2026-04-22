"""Sentinel-2 L1C data service."""
from __future__ import annotations

import rioxarray  # noqa: F401
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
import pyproj
import pystac
import shapely
import shapely.geometry
import shapely.ops
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.stac import load as odc_load

from geosave_engine.geodata.stac_client.base_client import BaseStacClient
from geosave_engine.geodata.stac_query.base_query import BaseStacQuery
from geosave_engine.geodata.ingestion.utils import aoi_from_query, is_wgs84, to_utm_bounds, radiometry_from_item

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SentinelL1CService:
    """Sentinel-2 L1C data service: search, lazy load, TOA reflectance, save.

    Output from ``load`` and ``load_item`` is raw TOA reflectance
    (``DN * scale + offset``).  Normalisation is intentionally omitted —
    compute dataset statistics after ingestion and apply them in the training
    pipeline.
    """

    ALL_BANDS: ClassVar[list[str]] = [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
    ]

    def __init__(self, client: BaseStacClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        queries: BaseStacQuery | list[BaseStacQuery],
        *,
        bands: list[str] | None = None,
        resolution: int = 10,
        as_reflectance: bool = True,
    ) -> Iterator[xr.DataArray]:
        """Yield a lazy (band, y, x) DataArray for every unique item matching the queries.

        Deduplicates across queries by item.id. Each yielded DataArray is
        clipped to the query AOI and in the item's native UTM CRS.
        """
        query_list = [queries] if isinstance(queries, BaseStacQuery) else queries

        bands = bands or self.ALL_BANDS

        seen: dict[str, tuple[pystac.Item, shapely.Geometry | None]] = {}
        for query in query_list:
            aoi_geom = aoi_from_query(query)
            for item in self._client.search(query):
                if item.id not in seen:
                    seen[item.id] = (item, aoi_geom)

        for item, aoi_geom in seen.values():
            yield self._make_lazy_da([item], aoi_geom, bands, resolution, as_reflectance=as_reflectance)

    def search_items(self, query: BaseStacQuery) -> list[pystac.Item]:
        """Return all items matching ``query`` via a live STAC API call."""
        return self._client.search(query)

    def load_item(
        self,
        items: list[pystac.Item],
        query: BaseStacQuery,
        *,
        bands: list[str] | None = None,
        resolution: int = 10,
        as_reflectance: bool = True,
        utm_bounds: tuple[float, float, float, float] | None = None,
        utm_crs: str | int | None = None,
    ) -> xr.DataArray:
        """Build a lazy DataArray mosaicked from one or more items.

        Multiple items (e.g. two adjacent MGRS tiles) are merged by odc-stac
        into the same GeoBox, covering boundary AOIs without tile-edge NaN gaps.

        utm_bounds + utm_crs bypass the WGS-84 reprojection round-trip.
        Provide native UTM bounds so the GeoBox matches the source pixels exactly.
        Radiometry is derived from items[0]; all items in one query share
        the same scale/offset from the same provider.
        """
        aoi_geom = None if utm_bounds is not None else aoi_from_query(query)
        return self._make_lazy_da(
            items,
            aoi_geom,
            bands or self.ALL_BANDS,
            resolution,
            as_reflectance=as_reflectance,
            utm_bounds=utm_bounds,
            utm_crs=utm_crs,
        )

    def save(
        self,
        da: xr.DataArray,
        path: Path,
        rio_kwargs: dict | None = None,
    ) -> None:
        """Write a computed DataArray to an LZW-compressed tiled GeoTIFF."""
        rio_kwargs = rio_kwargs or {}
        predictor  = 3 if np.issubdtype(da.dtype, np.floating) else 2
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        defaults = dict(compress="lzw", predictor=predictor, tiled=True)
        defaults.update(rio_kwargs)
        da.rio.to_raster(path, **defaults)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_lazy_da(
        self,
        items: list[pystac.Item],
        aoi_geom: shapely.Geometry | None,
        bands: list[str],
        resolution: int,
        *,
        as_reflectance: bool = True,
        utm_bounds: tuple[float, float, float, float] | None = None,
        utm_crs: str | int | None = None,
    ) -> xr.DataArray:
        geobox, crs_str = self._resolve_geobox(items, aoi_geom, resolution, utm_bounds, utm_crs)
        da = self._load_datacube(items, bands, geobox, crs_str)
        if as_reflectance:
            scale, offset = radiometry_from_item(items[0])
            da = da.astype(np.float32) * np.float32(scale) + np.float32(offset)
        return da

    def _resolve_geobox(
        self,
        items: list[pystac.Item],
        aoi_geom: shapely.Geometry | None,
        resolution: int,
        utm_bounds: tuple[float, float, float, float] | None,
        utm_crs: str | int | None,
    ) -> tuple[GeoBox, str]:
        """Determine bounds + CRS and build a GeoBox.

        Priority:
          1. utm_bounds + utm_crs  — use directly (no reprojection)
          2. utm_bounds only       — use first item's native CRS
          3. aoi_geom (WGS-84)    — reproject to first item's native CRS
          4. no constraint        — full tile extent in native CRS
        """
        if utm_bounds is not None:
            bounds  = utm_bounds
            crs_str = _normalize_epsg(utm_crs) if utm_crs is not None else f"EPSG:{self._item_epsg(items[0])}"
        elif aoi_geom is not None:
            epsg    = self._item_epsg(items[0])
            bounds  = to_utm_bounds(aoi_geom, epsg) if is_wgs84(aoi_geom) else aoi_geom.bounds
            crs_str = f"EPSG:{epsg}"
        else:
            epsg       = self._item_epsg(items[0])
            scene_geom = shapely.geometry.shape(items[0].geometry)
            bounds     = to_utm_bounds(scene_geom, epsg)
            crs_str    = f"EPSG:{epsg}"
        return GeoBox.from_bbox(bounds, crs=crs_str, resolution=resolution), crs_str

    def _load_datacube(
        self,
        items: list[pystac.Item],
        bands: list[str],
        geobox: GeoBox,
        crs_str: str,
    ) -> xr.DataArray:
        """Load items via odc-stac and return a (band, y, x) lazy DataArray."""
        ds = odc_load(
            items,
            bands=bands,
            geobox=geobox,
            resampling="bilinear",
            chunks={"x": 1024, "y": 1024},
        )
        da = ds.to_array(dim="band")
        time_dims = [d for d in da.dims if d not in ("band", "y", "x")]
        if time_dims:
            da = da.isel({d: 0 for d in time_dims}, drop=True)
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        da = da.rio.write_crs(crs_str, inplace=False)
        da = da.rio.write_transform(geobox.transform, inplace=False)
        return da

    def _item_epsg(self, item: pystac.Item) -> int:
        """Resolve an item's EPSG code from item properties or asset metadata."""
        def _parse(value: object) -> int | None:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            text = str(value)
            if text.upper().startswith("EPSG:"):
                return int(text.split(":", 1)[1])
            if text.isdigit():
                return int(text)
            return None

        for attr in ("proj:epsg", "proj:code"):
            if (epsg := _parse(item.properties.get(attr))) is not None:
                return epsg

        for asset in item.assets.values():
            for attr in ("proj:epsg", "proj:code"):
                if (epsg := _parse(asset.extra_fields.get(attr))) is not None:
                    return epsg

        raise ValueError(f"item {item.id!r} has no proj:epsg/proj:code metadata")

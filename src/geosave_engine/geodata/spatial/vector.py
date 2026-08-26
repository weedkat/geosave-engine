"""GeoVector: vector features over one area. See GeoVector for details."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import pandas as pd
from odc.geo.geom import Geometry

from geosave_engine.geodata.utils.io.vector import from_vector, to_geoparquet
from geosave_engine.geodata.utils.spatial.crs import calculate_crs, linear_unit_factors
from geosave_engine.geodata.utils.spatial.geobox import geobox_crs
from geosave_engine.geodata.utils.spatial.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox
    from pyproj import CRS


@dataclass(frozen=True, eq=False)
class GeoVector:
    """Vector features over one area — geometries plus their properties.

    Rides along a GeoAnchor as its `vector`, stored as a GeoParquet
    sidecar. Filtering, joins, and dissolve go through `gdf` directly.
    Compares by identity, caches nothing — compare rows with `gdf.equals`.

    Args:
        gdf: One row per geometry, any property columns, one CRS.

    Examples:
        >>> gv = GeoVector.open("plantations.geojson")
        >>> gv.gdf[gv.gdf["crop"] == "palm"]
    """

    gdf: gpd.GeoDataFrame

    def __post_init__(self) -> None:
        """Require a CRS and at least one row — "no features" is spelled `None`, not an empty vector.

        Raises:
            ValueError: `gdf` has no CRS set, holds no rows, or contains
                null, empty, or invalid geometries.
        """
        if self.gdf.crs is None:
            raise ValueError("GeoVector needs a CRS — call gdf.set_crs(...) first")
        if len(self.gdf) == 0:
            raise ValueError("GeoVector needs at least one row — spell 'no features' as None")
        if self.gdf.geometry.isna().any():
            raise ValueError("GeoVector geometries must not be null")
        if self.gdf.geometry.is_empty.any():
            raise ValueError("GeoVector geometries must not be empty")
        if not self.gdf.geometry.is_valid.all():
            raise ValueError("GeoVector geometries must be valid")

    def __repr__(self) -> str:
        columns = [str(name) for name in self.gdf.columns if name != self.gdf.geometry.name]
        return f"{type(self).__name__}({len(self.gdf)} features, {self.crs.to_string()}, columns={columns})"

    def __len__(self) -> int:
        """Row count. Always at least 1."""
        return len(self.gdf)

    @property
    def crs(self) -> CRS:
        """This vector's CRS.

        Raises:
            ValueError: `gdf` lost its CRS after this vector was built —
                only reachable by mutating `gdf` in place.
        """
        crs = self.gdf.crs
        if crs is None:
            raise ValueError("GeoVector's gdf lost its CRS — it was cleared after construction")
        return crs

    @property
    def footprint(self) -> Geometry:
        """Union of every geometry, in this vector's own CRS. Recomputed per call."""
        return Geometry(self.gdf.geometry.union_all(), crs=self.crs)

    @property
    def area_m2(self) -> float:
        """Ground area the features cover, in m² — what `GeoAnchor.area_m2` is to the pixel grid.

        Every feature counts in full, so two that overlap count their
        shared ground twice; read `footprint_area_m2` for the merged
        outline instead. Lines and points contribute nothing.

        Returns:
            Summed feature area in m².
        """
        projected = self._projected()
        x_to_m, y_to_m = linear_unit_factors(projected.crs)
        return float(projected.gdf.area.sum()) * x_to_m * y_to_m

    @property
    def footprint_area_m2(self) -> float:
        """Ground area the features cover once merged, in m² — overlaps counted a single time.

        Returns:
            Area of `footprint` in m².
        """
        projected = self._projected()
        x_to_m, y_to_m = linear_unit_factors(projected.crs)
        return float(projected.gdf.geometry.union_all().area) * x_to_m * y_to_m

    def _projected(self) -> GeoVector:
        """This vector on a projected CRS, ready for planar area calculations.

        Returns:
            `self` when already projected, else a copy on the UTM/UPS
            CRS local to its own centroid.
        """
        if self.crs.is_projected:
            return self
        lon, lat = self.to_crs("EPSG:4326").footprint.centroid.coords[0]
        return self.to_crs(calculate_crs(float(lat), float(lon)))

    def to_crs(self, crs: str | CRS) -> GeoVector:
        """Reproject every geometry.

        Args:
            crs: Target CRS, e.g. `"EPSG:32748"` or a pyproj CRS.

        Returns:
            New GeoVector on `crs`. `self` when it's already on it.
        """
        if self.crs == crs:
            return self
        return GeoVector(self.gdf.to_crs(crs))

    def filter(self, geobox: GeoBox) -> GeoVector | None:
        """Keep whole features that touch geobox's extent, on geobox's CRS.

        Geometry is never modified — a feature crossing the edge is kept
        entire and overhangs past geobox. One spanning several tiles shows
        up in each, whole, with its original index.

        Args:
            geobox: Pixel grid to test against. Needs a CRS.

        Returns:
            New GeoVector on geobox's CRS holding every intersecting row
            unchanged, or None if nothing intersects. Property columns
            (`area`, `length`) stay valid, unlike after `clip`.

        Raises:
            ValueError: `geobox` has no CRS, so there's nothing to reproject onto.
        """
        target = self.to_crs(geobox_crs(geobox))
        # sindex.query runs the exact predicate but only over bbox candidates, so it scales with matches
        matches = target.gdf.sindex.query(geobox.boundingbox.polygon.geom, predicate="intersects")
        if len(matches) == 0:
            return None
        return GeoVector(gpd.GeoDataFrame(target.gdf.iloc[matches]))

    def clip(self, geobox: GeoBox) -> GeoVector | None:
        """Cut features down to geobox's extent exactly, on geobox's CRS.

        A geometry crossing the edge is cut, not dropped, so an area/length
        column computed before the cut no longer matches its geometry. For
        tiling use `filter` — cheaper and reversible.

        Args:
            geobox: Pixel grid to clip against. Needs a CRS.

        Returns:
            New GeoVector on geobox's CRS, or None if nothing intersects.

        Raises:
            ValueError: `geobox` has no CRS, so there's nothing to reproject onto.
        """
        target = self.to_crs(geobox_crs(geobox))
        clipped = target.gdf.clip(geobox.boundingbox.bbox)
        if len(clipped) == 0:
            return None
        return GeoVector(gpd.GeoDataFrame(clipped))

    def add(self, geometry: SomeGeometry, **properties: Any) -> GeoVector:
        """Add one new feature.

        Args:
            geometry: GeoJSON geometry dict, WKT string, or shapely
                geometry — coordinates already on this vector's CRS.
            **properties: Column values for the new row, e.g.
                `crop="palm", year=2024`. A column this vector already has
                but that isn't given here comes out null; a new column
                comes out null on every existing row.

        Returns:
            New GeoVector one row longer, the row indexed past the current maximum.

        Raises:
            ValueError: `geometry` can't be parsed or is empty, or this
                vector's index isn't integer so there's no next index to
                hand out — `gdf.reset_index()` first.

        Examples:
            >>> gv.add("POLYGON ((13 52, 13.1 52, 13.1 52.1, 13 52.1, 13 52))", crop="palm")
            >>> gv.add({"type": "Point", "coordinates": [13.05, 52.05]}, crop="rice")
            >>> from shapely.geometry import box
            >>> gv.add(box(13.0, 52.0, 13.1, 52.1), crop="palm", year=2024)
            >>> gv.add(anchor.extent.geom, note="whole tile")  # odc Geometry -> shapely via .geom
        """
        if not pd.api.types.is_integer_dtype(self.gdf.index.dtype):
            raise ValueError(f"add() needs an integer index, got {self.gdf.index.dtype} — call gdf.reset_index() first")

        next_index = cast(int, self.gdf.index.max()) + 1
        row = gpd.GeoDataFrame(
            [properties], geometry=[to_shapely(geometry)], crs=self.crs, index=pd.Index([next_index])
        )
        return GeoVector(gpd.GeoDataFrame(pd.concat([self.gdf, row]), crs=self.crs))

    @classmethod
    def concat(cls, *vectors: GeoVector | None) -> GeoVector | None:
        """Combine vectors, keeping one row per distinct feature.

        Row labels are ignored: two rows are the same feature when their
        geometry and every property match. A feature `filter` fanned out
        across tiles collapses back to one row, separate ones all survive.

        Args:
            *vectors: Vectors to combine, in any order. None entries are skipped.

        Returns:
            New GeoVector on the first non-None input's CRS, indexed
            0..n-1. None if every input was None (or none were given).
        """
        present = [vector for vector in vectors if vector is not None]
        if not present:
            return None

        target_crs = present[0].crs
        geometry_name = str(present[0].gdf.geometry.name)
        frames: list[gpd.GeoDataFrame] = []
        for vector in present:
            frame = vector.to_crs(target_crs).gdf
            source_name = str(frame.geometry.name)
            if source_name != geometry_name:
                if geometry_name in frame.columns:
                    raise ValueError(
                        f"cannot combine geometry column {source_name!r} with property column {geometry_name!r}"
                    )
                frame.rename_geometry(geometry_name, inplace=True)
            frames.append(frame)

        combined = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry=geometry_name,
            crs=target_crs,
        )

        # geometry objects don't compare as values, so match on their WKB bytes instead
        keys = pd.DataFrame(combined).assign(**{geometry_name: combined.geometry.to_wkb()})
        unique = combined[~keys.duplicated()].reset_index(drop=True)
        return cls(gpd.GeoDataFrame(unique, crs=target_crs))

    @classmethod
    def from_geometry(cls, geometry: SomeGeometry, crs: str = "EPSG:4326") -> GeoVector:
        """One feature, no properties, from a geometry in any of the usual spellings.

        Args:
            geometry: GeoJSON geometry dict, WKT string (e.g.
                `"POINT (13.0 52.0)"`), or a shapely geometry.
            crs: CRS those coordinates are in.

        Returns:
            GeoVector holding exactly that one geometry.

        Raises:
            ValueError: `geometry` is WKT that can't be parsed, or is empty.
        """
        return cls(gpd.GeoDataFrame(geometry=[to_shapely(geometry)], crs=crs))

    @classmethod
    def open(cls, path: str | Path) -> GeoVector:
        """Read features from a vector file — format from path's suffix.

        Args:
            path: `.parquet`, `.geojson`, `.gpkg`, or `.shp`.

        Returns:
            GeoVector on the file's own CRS.

        Raises:
            ValueError: `path`'s suffix isn't one of `VECTOR_SUFFIXES`.
        """
        return cls(from_vector(path))

    def to_parquet(self, path: str | Path) -> Path:
        """Write features as GeoParquet, in this vector's own CRS.

        Args:
            path: Output `.parquet` path.

        Returns:
            The written path.

        Raises:
            ValueError: `path` doesn't end in `.parquet`.
        """
        return to_geoparquet(self.gdf, path)

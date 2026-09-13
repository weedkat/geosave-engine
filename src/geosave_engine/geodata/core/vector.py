"""Validated GeoDataFrame-backed vector data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
from odc.geo import SomeCRS
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave_engine.geodata.utils.geo.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    from typing_extensions import Unpack

    from geosave_engine.geodata.utils.io.geojson import GeoJSONWriteOptions
    from geosave_engine.geodata.utils.io.geopackage import GeoPackageWriteOptions
    from geosave_engine.geodata.utils.io.geoparquet import GeoParquetWriteOptions


@dataclass(frozen=True, eq=False)
class GeoVector:
    """Store vector geometries and properties in one CRS.

    Containment against a raster geobox belongs to the operation combining
    vector and raster data.

    Args:
        gdf: Non-empty GeoDataFrame with one CRS and valid geometries.

    Raises:
        TypeError: `gdf` is not a GeoDataFrame.
        ValueError: The CRS or active geometry column is missing, or a
            geometry is null, empty, or invalid.

    Examples:
        >>> vector = GeoVector.from_geometry("POINT (112.15 -8.05)")
        >>> vector.crs.to_epsg()
        4326
    """

    gdf: gpd.GeoDataFrame

    def __post_init__(self) -> None:
        """Validate the vector contract.

        Raises:
            TypeError: `gdf` is not a GeoDataFrame.
            ValueError: The CRS or active geometry column is missing, or a
                geometry is null, empty, or invalid.
        """
        if not isinstance(self.gdf, gpd.GeoDataFrame):
            raise TypeError(
                f"gdf must be a GeoDataFrame, got {type(self.gdf).__name__}"
            )
        if self.gdf.active_geometry_name is None:
            raise ValueError("GeoVector needs an active geometry column")
        if self.gdf.crs is None:
            raise ValueError("GeoVector needs a CRS; call gdf.set_crs(...) first")
        if self.gdf.empty:
            raise ValueError("GeoVector needs at least one feature")
        if self.gdf.geometry.isna().any():
            raise ValueError("GeoVector geometries must not be null")
        if self.gdf.geometry.is_empty.any():
            raise ValueError("GeoVector geometries must not be empty")
        if not self.gdf.geometry.is_valid.all():
            raise ValueError("GeoVector geometries must be valid")

    def __len__(self) -> int:
        """Return the number of features.

        Returns:
            Number of GeoDataFrame rows.
        """
        return len(self.gdf)

    @property
    def crs(self) -> OdcCRS:
        """Return the vector coordinate reference system.

        Returns:
            CRS shared by every geometry.

        Raises:
            ValueError: The GeoDataFrame CRS was cleared after construction.
        """
        if self.gdf.crs is None:
            raise ValueError("GeoVector's CRS was cleared after construction")
        return OdcCRS(self.gdf.crs)

    @property
    def footprint(self) -> Geometry:
        """Return the union of all geometries in the vector CRS.

        Returns:
            ODC geometry carrying the vector CRS.
        """
        return Geometry(self.gdf.geometry.union_all(), crs=self.crs)

    def to_geojson(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
        **options: Unpack[GeoJSONWriteOptions],
    ) -> Path:
        """Write this vector as one GeoJSON file.

        GeoJSON states coordinates in WGS84, so a projected vector is
        reprojected on the way out. Reach for `to_geopackage` or
        `to_geoparquet` to keep the vector's own CRS.

        Args:
            path: Output path ending in `.geojson` or `.json`.
            overwrite: Replace an existing file when true.
            **options: GeoJSON write options.

        Returns:
            The written path.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.to_geojson("plots.geojson")
            PosixPath('plots.geojson')
        """
        from geosave_engine.geodata.utils.io import geojson

        return geojson.write(self.gdf, path, overwrite=overwrite, **options)

    def to_geopackage(
        self,
        path: str | PathLike[str],
        *,
        layer: str | None = None,
        overwrite: bool = False,
        **options: Unpack[GeoPackageWriteOptions],
    ) -> Path:
        """Write this vector as one layer in a GeoPackage.

        Args:
            path: Output path ending in `.gpkg`.
            layer: Layer name. None names the layer after the file stem.
            overwrite: Replace an existing file when true.
            **options: GeoPackage write options.

        Returns:
            The written path.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.to_geopackage("survey.gpkg", layer="plots")
            PosixPath('survey.gpkg')
        """
        from geosave_engine.geodata.utils.io import geopackage

        return geopackage.write(
            self.gdf, path, layer=layer, overwrite=overwrite, **options
        )

    def to_geoparquet(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
        **options: Unpack[GeoParquetWriteOptions],
    ) -> Path:
        """Write this vector as one GeoParquet file.

        Args:
            path: Output path ending in `.parquet` or `.geoparquet`.
            overwrite: Replace an existing file when true.
            **options: GeoParquet write options.

        Returns:
            The written path.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.to_geoparquet("plots.parquet", write_covering_bbox=True)
            PosixPath('plots.parquet')
        """
        from geosave_engine.geodata.utils.io import geoparquet

        return geoparquet.write(self.gdf, path, overwrite=overwrite, **options)

    def to_crs(self, crs: SomeCRS) -> GeoVector:
        """Reproject every geometry.

        Args:
            crs: Target coordinate reference system.

        Returns:
            New vector on `crs`, or this vector when already on it.
        """
        if self.crs == crs:
            return self
        return type(self)(self.gdf.to_crs(crs))

    @classmethod
    def from_geometry(
        cls,
        geometry: SomeGeometry,
        *,
        crs: SomeCRS | None = None,
    ) -> GeoVector:
        """Build a one-feature vector from a common geometry representation.

        Args:
            geometry: GeoJSON mapping, WKT string, Shapely geometry, or ODC
                geometry.
            crs: CRS of the supplied coordinates. None uses an ODC geometry's
                own CRS, or WGS84 for other representations.

        Returns:
            One-feature vector with no property columns.

        Raises:
            ValueError: The geometry cannot be parsed, is empty, is invalid,
                or carries a CRS different from `crs`.
        """
        geometry_crs = geometry.crs if isinstance(geometry, Geometry) else None
        if geometry_crs is not None and crs is not None and OdcCRS(crs) != geometry_crs:
            raise ValueError(
                f"geometry is in {geometry_crs} but crs= names {crs}; "
                "transform the geometry or provide its actual CRS"
            )
        resolved_crs = geometry_crs or crs or "EPSG:4326"
        return cls(
            gpd.GeoDataFrame(
                geometry=[to_shapely(geometry)],
                crs=resolved_crs,
            )
        )

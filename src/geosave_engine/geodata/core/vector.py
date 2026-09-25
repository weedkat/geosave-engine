"""Validated GeoDataFrame-backed vector data."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import UTC
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo import SomeCRS
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave_engine.geodata.utils.datetime import naive_utc
from geosave_engine.geodata.utils.geo.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from typing_extensions import Unpack

    from geosave_engine.geodata.utils.io.geojson import GeoJSONWriteOptions
    from geosave_engine.geodata.utils.io.geopackage import GeoPackageWriteOptions
    from geosave_engine.geodata.utils.io.geoparquet import GeoParquetWriteOptions

    from .anchor import GeoAnchor

type AnchorField = Literal["time", "grid"]
type XarrayField = AnchorField | Literal["variables"]
type SpatialPredicate = Literal[
    "intersects", "within", "contains", "covers", "covered_by"
]


@dataclass(frozen=True, eq=False)
class GeoVector:
    """Store vector geometries and properties in one CRS.

    Containment against a raster geobox belongs to the operation combining
    vector and raster data.

    Args:
        gdf: GeoDataFrame with one CRS and valid present geometries.

    Raises:
        TypeError: `gdf` is not a GeoDataFrame.
        ValueError: The CRS or active geometry column is missing, or a
            geometry is null, empty, or invalid.

    Examples:
        >>> vector = GeoVector.from_geometry("POINT (112.15 -8.05)")
        >>> vector.crs.to_epsg()
        4326
        >>> record = GeoVector.from_xarray(
        ...     prediction, path="rasters/prediction.zarr"
        ... )
        >>> catalog = GeoVector.concat([catalog, record])
        >>> matches = labels.query(prediction)
        >>> catalog.to_geoparquet("dataset/catalog.parquet")
        PosixPath('dataset/catalog.parquet')
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
        if self.gdf.empty:
            raise ValueError("an empty GeoVector has no footprint")
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
    ) -> Path | str:
        """Write this vector as one GeoParquet file.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            overwrite: Replace an existing file when true.
            **options: GeoParquet write options.

        Returns:
            Local writes return a path; URL writes return the supplied URL.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.to_geoparquet("plots.parquet")
            PosixPath('plots.parquet')
            >>> catalog.to_geoparquet(
            ...     "hf://buckets/fatmur/test/catalog.parquet",
            ...     storage_options={"token": token},
            ... )
            'hf://buckets/fatmur/test/catalog.parquet'
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
    def concat(cls, vectors: Iterable[GeoVector]) -> GeoVector:
        """Concatenate spatial records without interpreting their identity.

        Args:
            vectors: Collections already expressed on one CRS plane.

        Returns:
            New vector with unioned property columns and a fresh RangeIndex.

        Raises:
            ValueError: No vectors are supplied or their CRSs differ.
        """
        items = list(vectors)
        if not items:
            raise ValueError("concat needs at least one GeoVector")
        crs = items[0].crs
        mismatches = [str(item.crs) for item in items if item.crs != crs]
        if mismatches:
            raise ValueError(
                f"GeoVector CRSs differ from {crs}: {mismatches}; "
                "call to_crs explicitly"
            )
        frames: list[gpd.GeoDataFrame] = []
        for item in items:
            frame = item.gdf.copy()
            if frame.active_geometry_name != "geometry":
                if "geometry" in frame.columns:
                    raise ValueError(
                        "cannot concatenate an active geometry with another "
                        "column named 'geometry'"
                    )
                frame = frame.rename_geometry("geometry")
            frames.append(frame)
        frame = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs
        )
        return cls(frame)

    def upsert(self, records: GeoVector, *, on: str) -> GeoVector:
        """Replace matching keyed rows and append new keys.

        Args:
            records: Incoming rows in the same CRS.
            on: Property column containing explicit record identity.

        Returns:
            Updated collection.

        Raises:
            KeyError: Either collection lacks the key column.
            ValueError: CRSs differ or incoming keys are null or duplicated.
        """
        if self.crs != records.crs:
            raise ValueError(
                f"GeoVector CRSs differ ({self.crs} and {records.crs}); "
                "call to_crs explicitly"
            )
        for label, frame in (("existing", self.gdf), ("incoming", records.gdf)):
            if on not in frame:
                raise KeyError(f"{label} GeoVector has no {on!r} column")
        incoming = cast("pd.Series", records.gdf[on])
        if incoming.isna().any():
            raise ValueError(f"incoming {on!r} keys must not be null")
        duplicates = incoming[incoming.duplicated(keep=False)].tolist()
        if duplicates:
            raise ValueError(
                f"incoming {on!r} keys must be unique, got {duplicates}"
            )
        retained = type(self)(self.gdf.loc[~self.gdf[on].isin(incoming)].copy())
        combined = type(self).concat([retained, records])
        return type(self)(combined.gdf)

    def query(
        self,
        target: SomeGeometry
        | GeoAnchor
        | GeoVector
        | xr.DataArray
        | xr.Dataset
        | xr.DataTree,
        *,
        predicate: SpatialPredicate = "intersects",
    ) -> GeoVector:
        """Select rows having one exact spatial relation to a target.

        The predicate reads from each collection row to the target: `within`
        selects rows within the target, while `contains` selects rows that
        contain it.

        Args:
            target: Geometry, anchor, vector, or geolocated xarray object.
            predicate: Exact row-relative spatial predicate.

        Returns:
            Matching original rows in source order.

        Raises:
            ValueError: The predicate is unsupported or target is unlocatable.
        """
        predicates = ("intersects", "within", "contains", "covers", "covered_by")
        if predicate not in predicates:
            raise ValueError(
                f"predicate must be one of {sorted(predicates)}, got {predicate!r}"
            )
        if self.gdf.empty:
            return type(self)(self.gdf.copy())

        from .anchor import GeoAnchor

        if isinstance(target, GeoVector):
            geometry = target.footprint
        elif isinstance(target, GeoAnchor):
            geometry = target.geobox.extent
        elif isinstance(target, (xr.DataArray, xr.Dataset, xr.DataTree)):
            geometry = target.gs.anchor.geobox.extent
        else:
            geometry = type(self).from_geometry(target).footprint
        target_geometry = geometry.to_crs(self.crs).geom
        positions = sorted(set(self.gdf.sindex.query(target_geometry)))
        candidates = self.gdf.iloc[positions]
        exact = getattr(candidates.geometry, predicate)(target_geometry)
        return type(self)(candidates.loc[exact].copy())

    @classmethod
    def vectorize(
        cls,
        flags: xr.DataArray,
        *,
        value_name: str = "value",
        mask: xr.DataArray | np.ndarray | None = None,
        connectivity: Literal[4, 8] = 4,
    ) -> GeoVector:
        """Polygonize contiguous values from one geolocated flag plane.

        This operation computes lazy flags because geometry depends on their
        values.

        Args:
            flags: Two-dimensional categorical or flag array.
            value_name: Property column receiving each region's value.
            mask: Optional exact-grid mask selecting additional valid pixels.
            connectivity: Four- or eight-neighbour region connectivity.

        Returns:
            One row per contiguous flag region, in the raster CRS.

        Raises:
            ValueError: The array, mask, name, connectivity, or dtype is
                unsuitable for polygonization.
        """
        from geosave_engine.geodata.transform.vector import vectorize

        return vectorize(
            flags,
            value_name=value_name,
            mask=mask,
            connectivity=connectivity,
        )

    def rasterize(
        self,
        like: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        column: str | None = None,
        fill: int | float | bool = 0,
        dtype: DTypeLike | None = None,
        all_touched: bool = False,
    ) -> xr.DataArray:
        """Burn geometries or one property onto an existing exact grid.

        Args:
            like: Anchor or geolocated xarray object supplying the target grid.
            column: Property to burn. None returns a boolean presence mask.
            fill: Value outside geometries.
            dtype: Output dtype for property values.
            all_touched: Burn every touched pixel instead of pixel centres.

        Returns:
            Eager geolocated array on the exact target grid.

        Raises:
            KeyError: The selected property is absent.
            ValueError: Values or fill cannot be represented by the dtype.
        """
        from geosave_engine.geodata.transform.vector import rasterize

        return rasterize(
            self,
            like,
            column=column,
            fill=fill,
            dtype=dtype,
            all_touched=all_touched,
        )

    @classmethod
    def empty(cls, crs: SomeCRS) -> GeoVector:
        """Build an empty spatial collection on one CRS.

        Args:
            crs: Coordinate reference system for future geometries.

        Returns:
            Empty vector with an active geometry column.
        """
        return cls(gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs)))

    @classmethod
    def from_geometry(
        cls,
        geometry: SomeGeometry,
        *,
        crs: SomeCRS | None = None,
        **properties: object,
    ) -> GeoVector:
        """Build a one-feature vector from a common geometry representation.

        Args:
            geometry: GeoJSON mapping, WKT string, Shapely geometry, or ODC
                geometry.
            crs: CRS of the supplied coordinates. None uses an ODC geometry's
                own CRS, or WGS84 for other representations.
            **properties: Scalar properties stored beside the geometry.

        Returns:
            One-feature vector carrying the supplied property columns.

        Raises:
            ValueError: The geometry cannot be parsed, is empty, is invalid,
                or carries a CRS different from `crs`.
        """
        if "geometry" in properties:
            raise ValueError(
                "geometry is owned by GeoVector; do not pass it as a property"
            )
        geometry_crs = geometry.crs if isinstance(geometry, Geometry) else None
        if geometry_crs is not None and crs is not None and OdcCRS(crs) != geometry_crs:
            raise ValueError(
                f"geometry is in {geometry_crs} but crs= names {crs}; "
                "transform the geometry or provide its actual CRS"
            )
        resolved_crs = geometry_crs or crs or "EPSG:4326"
        return cls(
            gpd.GeoDataFrame(
                {name: [value] for name, value in properties.items()},
                geometry=[to_shapely(geometry)],
                crs=resolved_crs,
            )
        )

    @classmethod
    def from_anchor(
        cls,
        anchor: GeoAnchor,
        *,
        crs: SomeCRS | None = None,
        fields: Collection[AnchorField] = ("time", "grid"),
        **properties: object,
    ) -> GeoVector:
        """Build one spatial record from an exact grid and timespan.

        Args:
            anchor: Exact spatial and temporal coverage to register.
            crs: Optional CRS for the record geometry.
            fields: Lightweight metadata groups to store as columns.
            **properties: Caller-owned scalar properties.

        Returns:
            One-row vector describing the anchor without pixels.

        Raises:
            ValueError: A field is unsupported or a property is derived.
        """
        selected = set(fields)
        unsupported = sorted(selected - {"time", "grid"})
        if unsupported:
            raise ValueError(
                f"unsupported vector fields {unsupported}; "
                "choose from ['grid', 'time']"
            )

        derived: dict[str, object] = {}
        if "time" in selected:
            span = anchor.timespan
            derived.update(
                start_datetime=None
                if span is None
                else naive_utc(span[0]).replace(tzinfo=UTC),
                end_datetime=None
                if span is None
                else naive_utc(span[1]).replace(tzinfo=UTC),
            )
        if "grid" in selected:
            derived.update(
                grid_crs=str(anchor.crs),
                grid_transform=tuple(anchor.geobox.transform)[:6],
                grid_height=anchor.geobox.height,
                grid_width=anchor.geobox.width,
            )

        collisions = sorted(set(properties) & ({"geometry"} | set(derived)))
        if collisions:
            raise ValueError(f"properties collide with derived columns {collisions}")

        vector = cls.from_geometry(anchor.geobox.extent)
        frame = vector.gdf.copy()
        for name, value in {**properties, **derived}.items():
            frame[name] = pd.Series([value], index=frame.index)
        vector = cls(frame)
        return vector if crs is None else vector.to_crs(crs)

    @classmethod
    def from_xarray(
        cls,
        data: xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        geometry: SomeGeometry | None = None,
        crs: SomeCRS | None = None,
        path: str | PathLike[str] | None = None,
        fields: Collection[XarrayField] = ("time", "grid"),
        **properties: object,
    ) -> GeoVector:
        """Register xarray coverage and metadata without reading pixels.

        Args:
            data: Geolocated array, raster, or single-grid stack.
            geometry: Semantic geometry to retain. None uses the grid extent.
            crs: Optional CRS for the record geometry.
            path: Optional persisted asset reference.
            fields: Lightweight metadata groups to store as columns.
            **properties: Caller-owned scalar properties.

        Returns:
            One-row vector describing the xarray object without its pixels.

        Raises:
            ValueError: Data has no shared locatable grid, a field is
                unsupported, or a property collides with derived metadata.
        """
        selected = set(fields)
        unsupported = sorted(selected - {"time", "grid", "variables"})
        if unsupported:
            raise ValueError(
                f"unsupported vector fields {unsupported}; "
                "choose from ['grid', 'time', 'variables']"
            )
        anchor = data.gs.anchor
        anchor_fields: list[AnchorField] = []
        if "time" in selected:
            anchor_fields.append("time")
        if "grid" in selected:
            anchor_fields.append("grid")
        base = cls.from_anchor(anchor, fields=anchor_fields)
        row = base.gdf.drop(columns="geometry").iloc[0].to_dict()
        derived = set(row)
        if "variables" in selected:
            derived.add("variables")
        if path is not None:
            derived.add("path")
        collisions = sorted(set(properties) & derived)
        if collisions:
            raise ValueError(f"properties collide with derived columns {collisions}")

        row.update(properties)
        if "variables" in selected:
            row["variables"] = data.gs.variables
        if path is not None:
            row["path"] = path

        footprint = anchor.geobox.extent if geometry is None else geometry
        vector = cls.from_geometry(footprint)
        frame = vector.gdf.copy()
        for name, value in row.items():
            frame[name] = [value]
        vector = cls(frame)
        return vector if crs is None else vector.to_crs(crs)

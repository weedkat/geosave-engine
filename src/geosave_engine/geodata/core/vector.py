"""The `gs` GeoDataFrame accessor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo import SomeCRS
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave_engine.geodata.utils.datetime import naive_utc
from geosave_engine.geodata.utils.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from typing_extensions import Unpack

    from geosave_engine.geodata import GeoDataFrame
    from geosave_engine.geodata.io.vector.geojson import GeoJSONWriteOptions
    from geosave_engine.geodata.io.vector.geopackage import GeoPackageWriteOptions
    from geosave_engine.geodata.io.vector.geoparquet import GeoParquetWriteOptions

    from .anchor import GeoAnchor

type SpatialPredicate = Literal[
    "intersects", "within", "contains", "covers", "covered_by"
]

_PREDICATES = get_args(SpatialPredicate.__value__)


@pd.api.extensions.register_dataframe_accessor("gs")
class GeoVector:
    """GeoSave operations on a GeoDataFrame, read through `gs`.

    A vector is geometry with information attached: an active geometry column,
    a CRS, and any other columns. Nothing here requires or adds a column. A
    time filter applies in `query` only where the frame happens to carry
    `datetime`, `start_datetime` or `end_datetime`.

    Args:
        data: GeoDataFrame with an active geometry column.

    Raises:
        AttributeError: `data` is not a GeoDataFrame, or has no active
            geometry column.

    Examples:
        >>> plots = read_vector("plots.geojson")
        >>> plots.gs.crs.to_epsg()
        4326
        >>> mask = plots.gs.rasterize(scene, column="class")
        >>> plots.gs.to_geoparquet("plots.parquet")
        PosixPath('plots.parquet')
    """

    def __init__(self, data: gpd.GeoDataFrame) -> None:
        """Bind the accessor to one GeoDataFrame."""
        # pandas reads AttributeError as "this object has no such accessor".
        if not isinstance(data, gpd.GeoDataFrame):
            raise AttributeError(
                f"gs reads a GeoDataFrame, got {type(data).__name__}; build one "
                f"with gpd.GeoDataFrame(...)"
            )
        if data.active_geometry_name is None:
            raise AttributeError("gs needs an active geometry column")
        self._data = data

    def _time_bounds(self) -> tuple[pd.Series, pd.Series] | None:
        """Return each row's first and last instant in UTC, or None if undated.

        A frame is dated when it has `datetime`, `start_datetime` or
        `end_datetime`. A row missing an edge leaves it NaT, so a time query
        treats that edge as open.
        """
        frame = self._data
        times = {
            name: pd.to_datetime(frame[name], utc=True, format="mixed")
            for name in ("datetime", "start_datetime", "end_datetime")
            if name in frame
        }
        if not times:
            return None
        instant = times.get(
            "datetime",
            pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"),
        )
        first = times.get("start_datetime", instant).fillna(instant)
        last = times.get("end_datetime", instant).fillna(instant)
        return first, last

    @property
    def crs(self) -> OdcCRS:
        """Return the coordinate reference system every geometry shares."""
        if self._data.crs is None:
            raise ValueError("this vector has no CRS")
        return OdcCRS(self._data.crs)

    @property
    def footprint(self) -> Geometry:
        """Return the union of all geometries in the vector CRS.

        Returns:
            ODC geometry carrying the vector CRS.

        Raises:
            ValueError: The frame holds no row.
        """
        if self._data.empty:
            raise ValueError("an empty vector has no footprint")
        return Geometry(self._data.geometry.union_all(), crs=self.crs)

    def to_geojson(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
        **options: Unpack[GeoJSONWriteOptions],
    ) -> Path | str:
        """Write this vector as one GeoJSON file.

        GeoJSON states coordinates in WGS84, so a projected vector is
        reprojected on the way out. Reach for `to_geopackage` or
        `to_geoparquet` to keep the vector's own CRS.

        Args:
            path: Output path or URL ending in `.geojson` or `.json`.
            overwrite: Replace an existing file when true.
            **options: GeoJSON write options.

        Returns:
            Local writes return a path; URL writes return the supplied URL.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.gs.to_geojson("plots.geojson")
            PosixPath('plots.geojson')
        """
        from geosave_engine.geodata.io import geojson

        return geojson.write(self._data, path, overwrite=overwrite, **options)

    def to_geopackage(
        self,
        path: str | PathLike[str],
        *,
        layer: str | None = None,
        overwrite: bool = False,
        **options: Unpack[GeoPackageWriteOptions],
    ) -> Path | str:
        """Write this vector as one layer in a GeoPackage.

        Args:
            path: Output path or URL ending in `.gpkg`.
            layer: Layer name. None names the layer after the file stem.
            overwrite: Replace an existing file when true.
            **options: GeoPackage write options.

        Returns:
            Local writes return a path; URL writes return the supplied URL.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.gs.to_geopackage("survey.gpkg", layer="plots")
            PosixPath('survey.gpkg')
        """
        from geosave_engine.geodata.io import geopackage

        return geopackage.write(
            self._data, path, layer=layer, overwrite=overwrite, **options
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
            >>> plots.gs.to_geoparquet("plots.parquet")
            PosixPath('plots.parquet')
            >>> plots.gs.to_geoparquet(
            ...     "hf://buckets/fatmur/test/plots.parquet",
            ...     storage_options={"token": token},
            ... )
            'hf://buckets/fatmur/test/plots.parquet'
        """
        from geosave_engine.geodata.io import geoparquet

        return geoparquet.write(self._data, path, overwrite=overwrite, **options)

    @classmethod
    def concat(cls, vectors: Iterable[gpd.GeoDataFrame]) -> GeoDataFrame:
        """Concatenate spatial records without interpreting their identity.

        Args:
            vectors: Frames already expressed on one CRS plane.

        Returns:
            New frame with unioned property columns, a fresh RangeIndex, and
            its geometry column named `geometry`.

        Raises:
            ValueError: No frames are supplied, their CRSs differ, or a frame
                holds a second column named `geometry` beside its active one.
        """
        items = list(vectors)
        if not items:
            raise ValueError("concat needs at least one GeoDataFrame")
        crs = OdcCRS(items[0].crs)
        mismatches = [str(item.crs) for item in items if OdcCRS(item.crs) != crs]
        if mismatches:
            raise ValueError(
                f"the vectors' CRSs differ from {crs}: {mismatches}; "
                "call to_crs explicitly"
            )
        frames: list[gpd.GeoDataFrame] = []
        for item in items:
            frame = item.copy()
            if frame.active_geometry_name != "geometry":
                if "geometry" in frame.columns:
                    raise ValueError(
                        "cannot concatenate an active geometry with another "
                        "column named 'geometry'"
                    )
                frame = frame.rename_geometry("geometry")
            frames.append(frame)
        return cast(
            "GeoDataFrame",
            gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs
            ),
        )

    def query(
        self,
        target: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        predicate: SpatialPredicate = "intersects",
    ) -> GeoDataFrame:
        """Select the rows related to an anchor in space and in time.

        The predicate reads from each row to the anchor's ground. Where the
        anchor has a timespan and the table a `datetime`, a row must also
        overlap it, by its `start_datetime` and `end_datetime` or its `datetime`.

        Args:
            target: Anchor, or a geolocated xarray object whose anchor is used.
            predicate: Exact row-relative spatial predicate.

        Returns:
            Matching original rows in source order.

        Raises:
            ValueError: The predicate is unsupported, or the target carries no
                locatable grid.

        Examples:
            >>> manifest.gs.query(scene)["id"].tolist()
            ['s1', 's4']
        """
        if predicate not in _PREDICATES:
            raise ValueError(
                f"predicate must be one of {sorted(_PREDICATES)}, got {predicate!r}"
            )

        from .anchor import GeoAnchor

        anchor = target if isinstance(target, GeoAnchor) else target.gs.anchor
        frame = self._data
        if frame.empty:
            return cast("GeoDataFrame", frame.copy())

        ground = anchor.geobox.extent.to_crs(self.crs).geom
        candidates = frame.iloc[sorted(set(frame.sindex.query(ground)))]
        matched = candidates.loc[getattr(candidates.geometry, predicate)(ground)]

        span = anchor.timespan
        bounds = matched.gs._time_bounds()
        if span is not None and bounds is not None:
            start, end = (pd.Timestamp(naive_utc(edge), tz="UTC") for edge in span)
            first, last = bounds
            matched = matched.loc[
                (first.isna() | (first <= end)) & (last.isna() | (last >= start))
            ]
        return cast("GeoDataFrame", matched.copy())

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
            ValueError: A geometry is null, empty, or invalid, or values or
                fill cannot be represented by the dtype.
        """
        from geosave_engine.geodata.transform.vector import rasterize

        return rasterize(
            self._data,
            like,
            column=column,
            fill=fill,
            dtype=dtype,
            all_touched=all_touched,
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: SomeGeometry,
        *,
        crs: SomeCRS | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> GeoDataFrame:
        """Build a one-feature frame from a common geometry representation.

        Args:
            geometry: GeoJSON mapping, WKT string, Shapely geometry, or ODC
                geometry.
            crs: CRS of the supplied coordinates. None uses an ODC geometry's
                own CRS, or WGS84 for other representations.
            properties: Scalar columns stored beside the geometry.

        Returns:
            One-feature frame carrying the supplied property columns.

        Raises:
            ValueError: The geometry cannot be parsed, carries a CRS different
                from `crs`, or a property is named `geometry`.
        """
        properties = properties or {}
        if "geometry" in properties:
            raise ValueError(
                "geometry is the frame's own column; do not pass it as a property"
            )
        geometry_crs = geometry.crs if isinstance(geometry, Geometry) else None
        if geometry_crs is not None and crs is not None and OdcCRS(crs) != geometry_crs:
            raise ValueError(
                f"geometry is in {geometry_crs} but crs= names {crs}; "
                "transform the geometry or provide its actual CRS"
            )
        frame = gpd.GeoDataFrame(
            {name: [value] for name, value in properties.items()},
            geometry=[to_shapely(geometry)],
            crs=geometry_crs or crs or "EPSG:4326",
        )
        return cast("GeoDataFrame", frame)

"""The `gs` GeoDataFrame accessor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import pystac
from odc.geo import SomeCRS
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave_engine.geodata.utils.datetime import naive_utc
from geosave_engine.geodata.utils.geo.geometry import SomeGeometry, to_shapely

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from typing_extensions import Unpack

    from geosave_engine.geodata import GeoDataFrame
    from geosave_engine.geodata.io.geojson import GeoJSONWriteOptions
    from geosave_engine.geodata.io.geopackage import GeoPackageWriteOptions
    from geosave_engine.geodata.io.geoparquet import GeoParquetWriteOptions

    from .anchor import GeoAnchor

type SpatialPredicate = Literal[
    "intersects", "within", "contains", "covers", "covered_by"
]

_PREDICATES = get_args(SpatialPredicate.__value__)


@pd.api.extensions.register_dataframe_accessor("gs")
class GeoVector:
    """GeoSave operations on a GeoDataFrame, read through `gs`.

    The frame stays a native GeoDataFrame, so every GeoPandas operation still
    applies to it.

    Args:
        data: GeoDataFrame with an active geometry column. Pixel-only catalogs
            may have no CRS; geographic operations require one.

    Raises:
        AttributeError: `data` is not a GeoDataFrame, or has no active
            geometry column.

    Examples:
        >>> plots = read_vector("plots.geojson")
        >>> plots.gs.crs.to_epsg()
        4326
        >>> catalog = GeoVector.from_items(ds.gs.to_items("dataset/forest"))
        >>> catalog = catalog.gs.upsert(another_item, on="id")
        >>> matches = labels.gs.query(prediction)
        >>> catalog.gs.to_geoparquet("dataset/catalog.parquet")
        PosixPath('dataset/catalog.parquet')
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

    def to_raster(self, **options: Any) -> xr.Dataset:
        """Read the data assets of every row as one lazy raster.

        Select rows first, with pandas or `query`. Variables of split-band
        scenes come back in asset-key order; select by name where order matters.

        Args:
            **options: Forwarded to `read_raster`. `chunks` defaults to `{}`.

        Returns:
            Dataset joining the rows' scenes along `time`.

        Raises:
            ValueError: The frame holds no row, or its assets sit on different
                grids or repeat an instant.

        Examples:
            >>> catalog.gs.query(scene).gs.to_raster().sizes["time"]
            2
        """
        from geosave_engine.geodata.io import read_raster

        hrefs = [
            href for _, row in self._data.iterrows() for href in row.gs.hrefs.values()
        ]
        return read_raster(hrefs, **{"chunks": {}, **options})

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

        A table of STAC Items is written as STAC GeoParquet, whose schema and
        file metadata stac-geoparquet owns. Geometry options such as
        `write_covering_bbox` apply only to ordinary tables.

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
            >>> catalog.gs.to_geoparquet(
            ...     "hf://buckets/fatmur/test/catalog.parquet",
            ...     storage_options={"token": token},
            ... )
            'hf://buckets/fatmur/test/catalog.parquet'
        """
        from geosave_engine.geodata.io import geoparquet

        return geoparquet.write(self._data, path, overwrite=overwrite, **options)

    @classmethod
    def from_items(cls, items: Iterable[pystac.Item]) -> GeoDataFrame:
        """Build a native GeoDataFrame from STAC Items without mutating them.

        Args:
            items: Items with self hrefs for resolving any relative assets.

        Returns:
            Frame with library-converted STAC properties and absolute assets.

        Raises:
            ValueError: No Items are supplied.
            pystac.STACError: A relative asset lacks an Item self href.
        """
        from stac_geoparquet.arrow import parse_stac_items_to_arrow

        clones = []
        for item in items:
            clone = item.clone()
            clone.make_asset_hrefs_absolute()
            clones.append(clone)
        if not clones:
            raise ValueError("from_items needs at least one STAC Item")
        table = parse_stac_items_to_arrow(
            clones, drop_invalid_properties=False
        ).read_all()
        return cast("GeoDataFrame", gpd.GeoDataFrame.from_arrow(table))

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

    def upsert(
        self, records: gpd.GeoDataFrame | pystac.Item, *, on: str
    ) -> GeoDataFrame:
        """Replace matching keyed rows and append new keys.

        Args:
            records: Incoming frame in the same CRS, or one native STAC Item.
            on: Property column containing explicit record identity.

        Returns:
            Updated collection. Replaced rows move to the end, so row order
            is not preserved.

        Raises:
            KeyError: Either collection lacks the key column.
            ValueError: CRSs differ or incoming keys are null or duplicated.
        """
        if isinstance(records, pystac.Item):
            records = type(self).from_items([records])
        if self.crs != records.gs.crs:
            raise ValueError(
                f"the vectors' CRSs differ ({self.crs} and {records.gs.crs}); "
                "call to_crs explicitly"
            )
        for label, frame in (("existing", self._data), ("incoming", records)):
            if on not in frame:
                raise KeyError(f"the {label} vector has no {on!r} column")
        keys = cast("pd.Series", records[on])
        if keys.isna().any():
            raise ValueError(f"incoming {on!r} keys must not be null")
        duplicates = keys[keys.duplicated(keep=False)].tolist()
        if duplicates:
            raise ValueError(f"incoming {on!r} keys must be unique, got {duplicates}")
        others = self._data.loc[~self._data[on].isin(keys)]
        return type(self).concat([others, records])

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
        if span is not None and "datetime" in matched:
            start, end = (pd.Timestamp(naive_utc(edge), tz="UTC") for edge in span)
            instant = pd.to_datetime(matched["datetime"], utc=True)
            first, last = (
                pd.to_datetime(matched.get(name, instant), utc=True).fillna(instant)
                for name in ("start_datetime", "end_datetime")
            )
            matched = matched.loc[(first <= end) & (last >= start)]
        return cast("GeoDataFrame", matched.copy())

    @classmethod
    def vectorize(
        cls,
        flags: xr.DataArray,
        *,
        value_name: str = "value",
        mask: xr.DataArray | np.ndarray | None = None,
        connectivity: Literal[4, 8] = 4,
    ) -> GeoDataFrame:
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
    def empty(cls, crs: SomeCRS) -> GeoDataFrame:
        """Build an empty spatial collection on one CRS.

        Args:
            crs: Coordinate reference system for future geometries.

        Returns:
            Empty frame with an active geometry column.
        """
        return cast(
            "GeoDataFrame", gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs))
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
        return cast(
            "GeoDataFrame",
            gpd.GeoDataFrame(
                {name: [value] for name, value in properties.items()},
                geometry=[to_shapely(geometry)],
                crs=geometry_crs or crs or "EPSG:4326",
            ),
        )

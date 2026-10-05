"""The `gs` GeoDataFrame accessor."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

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
    from tiler import Tiler
    from numpy.typing import DTypeLike
    from typing_extensions import Unpack

    from geosave_engine.geodata import GeoDataFrame
    from geosave_engine.geodata.stac.item import Asset
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
        >>> record = GeoVector.from_assets({"prediction": "rasters/prediction.zarr"})
        >>> catalog = GeoVector.concat([catalog, record])
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

        A table with an `assets` column is written as a STAC table: asset
        hrefs below the file's folder are stored relative to it, and the
        bounding box and the `stac-geoparquet` file key are added.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            overwrite: Replace an existing file when true.
            **options: GeoParquet write options.

        Returns:
            Local writes return a path; URL writes return the supplied URL.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong, or a STAC table has a null or
                repeated `id` or is not in longitude/latitude.

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

    def upsert(self, records: gpd.GeoDataFrame, *, on: str) -> GeoDataFrame:
        """Replace matching keyed rows and append new keys.

        Args:
            records: Incoming rows in the same CRS.
            on: Property column containing explicit record identity.

        Returns:
            Updated collection. Replaced rows move to the end, so row order
            is not preserved.

        Raises:
            KeyError: Either collection lacks the key column.
            ValueError: CRSs differ or incoming keys are null or duplicated.
        """
        if self.crs != records.gs.crs:
            raise ValueError(
                f"the vectors' CRSs differ ({self.crs} and {records.gs.crs}); "
                "call to_crs explicitly"
            )
        for label, frame in (("existing", self._data), ("incoming", records)):
            if on not in frame:
                raise KeyError(f"the {label} vector has no {on!r} column")
        incoming = cast("pd.Series", records[on])
        if incoming.isna().any():
            raise ValueError(f"incoming {on!r} keys must not be null")
        duplicates = incoming[incoming.duplicated(keep=False)].tolist()
        if duplicates:
            raise ValueError(f"incoming {on!r} keys must be unique, got {duplicates}")
        retained = self._data.loc[~self._data[on].isin(incoming)]
        return type(self).concat([retained, records])

    def query(
        self,
        target: GeoAnchor | xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        predicate: SpatialPredicate = "intersects",
    ) -> GeoDataFrame:
        """Select the rows related to an anchor in space and in time.

        The predicate reads from each row to the anchor's ground: `within`
        selects rows within it, while `contains` selects rows that contain it.
        Time is compared where both sides state one: the anchor has a timespan
        and the table has `datetime`, `start_datetime`, and `end_datetime`. A
        row is then read by its span, or by its `datetime` where its span is
        null, and matches when that overlaps the anchor's timespan.

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
        times = ("datetime", "start_datetime", "end_datetime")
        if span is not None and all(name in matched for name in times):
            start, end = (pd.Timestamp(naive_utc(edge), tz="UTC") for edge in span)
            instant, first, last = (
                pd.to_datetime(matched[name], utc=True) for name in times
            )
            first, last = first.fillna(instant), last.fillna(instant)
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

    @classmethod
    def from_layouts(
        cls,
        parents: Mapping[str, xr.Dataset | xr.DataArray | xr.DataTree],
        layouts: Mapping[str, Tiler],
        *,
        padding: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    ) -> gpd.GeoDataFrame:
        """Associate native tile IDs with pixel windows and optional exact grids.

        Args:
            parents: Original prepared scenes or frames keyed by persistent ID.
            layouts: Native spatial Tiler for each parent.
            padding: Halo widths used to extend each layout's data shape.
                None means no halo; native fringe padding needs no entry.

        Returns:
            Metadata-only reference with an `id` column. Footprints use WGS84;
            unreferenced parents have null geometry and projection fields.

        Raises:
            ValueError: Parents are empty, IDs are empty, keys differ, or a
                layout does not describe two spatial dimensions.

        Examples:
            >>> reference = GeoVector.from_layouts(parents, layouts)
            >>> reference.set_index("id").loc["scene-a/tile-0", "tile_id"]
            0
        """
        from geosave_engine.geodata.transform.vector import from_layouts

        return from_layouts(parents, layouts, padding=padding)

    @classmethod
    def from_assets(
        cls,
        assets: Mapping[str, Asset] | str | PathLike[str],
        *,
        id: str | None = None,
        datetime: dt.datetime | None = None,
        geometry: SomeGeometry | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> GeoDataFrame:
        """Register stored rasters as one STAC item, read from the files.

        The row is read off what is on disk, so it cannot describe pixels that
        were never written. No pixel is read.

        Args:
            assets: Where each raster is stored, by layer name. A value is a
                path or URL, or a STAC asset mapping stating `href` and an
                optional Zarr/NetCDF `group`. One bare path uses its file stem.
            id: Item identifier. None uses the anchor stem.
            datetime: Item instant. None leaves it null, which the timespan in
                `start_datetime` and `end_datetime` then stands in for.
            geometry: Semantic geometry to retain. None uses the grid extent.
            properties: Caller-owned scalar columns.

        Returns:
            One-row frame in longitude/latitude describing the rasters, its
            `sources` listing the provider items they were loaded from.

        Raises:
            OSError: An asset names no stored raster.
            ValueError: The rasters share no grid, an asset states no href,
                the item has no time, or a property takes the name of an item
                column.

        Examples:
            >>> record = GeoVector.from_assets(
            ...     {
            ...         "label": "samples/s1/label.tif",
            ...         "sentinel_2_l2a": "samples/s1/sentinel_2_l2a.tif",
            ...     },
            ...     properties={"land_cover": "forest"},
            ... )
            >>> record.loc[0, "id"]
            '2.7415W_5.6550N_5.1kmx5.1km_20181226_10m'
        """
        from geosave_engine.geodata.io.assets import normalize, read
        from geosave_engine.geodata.stac.item import record

        absolute = normalize(assets)
        with read(absolute) as stored:
            return record(
                stored,
                assets=absolute,
                id=id,
                datetime=datetime,
                geometry=geometry,
                properties=properties,
            )

    @classmethod
    def from_xarray(
        cls,
        data: xr.Dataset | xr.DataTree,
        *,
        id: str | None = None,
        datetime: dt.datetime | None = None,
        geometry: SomeGeometry | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> GeoDataFrame:
        """Register an object by the file it was read from.

        The path is found on the object and the row is read from that file,
        as `from_assets` reads it. An object GeoSave changed since it was read,
        or one selected, renamed, or joined so that it no longer matches its
        file, must be written first. Plain xarray arithmetic keeps both the
        path and the shape, so the row then describes the file, not the
        changed values.

        Args:
            data: Raster read from a file, or a stack whose groups each were;
                a group's name becomes its asset key.
            id: Item identifier. None uses the anchor stem.
            datetime: Item instant, as `from_assets` takes it.
            geometry: Semantic geometry to retain. None uses the grid extent.
            properties: Caller-owned scalar columns.

        Returns:
            One-row frame in longitude/latitude describing the stored rasters.

        Raises:
            ValueError: A raster was not read from a file, its grid, variables
                or shape no longer match, or the input is a single band.

        Examples:
            >>> GeoVector.from_xarray(read_raster("samples/s1/scene.tif"))
        """
        unsaved = "write it, then register the path with GeoVector.from_assets"
        if isinstance(data, xr.DataTree):
            rasters = data.gs.rasters
        elif isinstance(data, xr.Dataset):
            rasters = {"": data}
        else:
            raise ValueError(
                f"a row is read from a raster's file, and a {type(data).__name__} "
                f"is one band of it; {unsaved}"
            )

        from geosave_engine.geodata.io.assets import read
        from geosave_engine.geodata.stac.item import file_assets, record

        assets = file_assets(data)

        # Compare metadata with the selected stored group, without reading pixels.
        stored = read(assets)
        try:
            for name, raster in zip(assets, rasters.values(), strict=True):
                original = stored.gs.rasters[name]
                if (
                    original.gs.geobox != raster.gs.geobox
                    or tuple(original.data_vars) != tuple(raster.data_vars)
                    or dict(original.sizes) != dict(raster.sizes)
                ):
                    raise ValueError(
                        f"{name} no longer matches its stored raster; {unsaved}"
                    )
            return record(
                stored,
                assets=assets,
                id=id,
                datetime=datetime,
                geometry=geometry,
                properties=properties,
            )
        finally:
            stored.close()

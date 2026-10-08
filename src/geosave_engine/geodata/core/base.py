"""Properties and pixel operations shared by every `gs` raster accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import pandas as pd
import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.utils.datetime import parse_daterange

from geosave_engine.geodata.conventions import TIME_COORDINATE, require_yx

if TYPE_CHECKING:
    import geopandas as gpd
    from collections.abc import Mapping, Sequence

    from odc.geo import CRS, BoundingBox, Resolution
    from odc.geo.gcp import GCPGeoBox
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import Geometry

    from geosave_engine.geodata.attrs import AttrsHeader, AttrsModel, AttrsNamespace
    from odc.geo import SomeResolution

    from geosave_engine.geodata.transform.warp import Resampling, Target
    from geosave_engine.geodata.utils.datetime import DateRange

    from .anchor import GeoAnchor


class GeoRasterAccessor[DataT: xr.Dataset | xr.DataArray | xr.DataTree]:
    """Spatial, attrs, and pixel members every `gs` raster accessor shares.

    Concrete accessors bind their xarray object to `_data`; this class is
    never built directly. Each pixel operation delegates to `transform`,
    which treats a band, a raster, and a stack alike.
    """

    _data: DataT

    def __init__(self, data: DataT) -> None:
        """Bind the xarray object.

        Args:
            data: Band, raster, or stack to read through this accessor.

        Raises:
            ValueError: Its grid, or a group's, spans dimensions other than
                `y` and `x`.
        """
        require_yx(data)
        self._data = data

    @property
    def _grid_source(self) -> xr.Dataset | xr.DataArray:
        """Return the object odc-geo reads a geobox off directly.

        A DataTree carries no `.odc` accessor of its own, so its root
        Dataset stands in for it.

        Returns:
            `_data` itself, or its root Dataset for a DataTree.
        """
        return self._data.dataset if isinstance(self._data, xr.DataTree) else self._data

    @property
    def geobox(self) -> GeoBox | GCPGeoBox | None:
        """Read the pixel grid, if this object carries one.

        Same lookup as `.odc.geobox`, since `gs` is built on odc-geo: None is
        a normal answer, not a refused one. A caller that needs ground
        position checks for None and a regular `GeoBox` itself.

        Returns:
            Grid including CRS, transform, bounds, and shape; a `GCPGeoBox`
            if placed by ground control points rather than a transform; or
            None if this object carries no CRS or spatial dims.
        """
        return self._grid_source.odc.geobox

    @property
    def is_georeferenced(self) -> bool:
        """Whether this object carries a locatable grid.

        Returns:
            True when `geobox` resolves to one.
        """
        return self.geobox is not None

    @property
    def crs(self) -> CRS | None:
        """Read the coordinate reference system, if this object carries one.

        Returns:
            CRS the grid mapping coordinate names, or None if this object
            carries no locatable grid or the grid itself carries no CRS.
        """
        geobox = self.geobox
        return None if geobox is None else geobox.crs

    @property
    def crs_name(self) -> str | None:
        """Name the coordinate reference system briefly, for a reader.

        A CRS read back off `spatial_ref` prints its whole WKT, and odc's own
        `crs_str` prints whatever the CRS was built from, so neither is short
        enough to name one in a message.

        Returns:
            Its EPSG code as `"EPSG:32633"`, the projection's own name where
            it carries no code, or None if this object carries no CRS.

        Examples:
            >>> scene.gs.crs_name
            'EPSG:32633'
        """
        crs = self.crs
        if crs is None:
            return None
        return f"EPSG:{crs.epsg}" if crs.epsg else crs.proj.name

    @property
    def bounds(self) -> BoundingBox | None:
        """Read the grid's extent in its own CRS, if this object carries one.

        Returns:
            Bounding box covering every pixel, or None if this object
            carries no locatable grid.
        """
        geobox = self.geobox
        return None if geobox is None else geobox.boundingbox

    @property
    def resolution(self) -> Resolution | None:
        """Read the pixel size in CRS units, if this object carries one.

        Returns:
            Signed resolution along x and y, or None if this object carries
            no locatable grid.
        """
        geobox = self.geobox
        return None if geobox is None else geobox.resolution

    @property
    def attrs(self) -> AttrsHeader:
        """Read the typed attrs this object carries.

        Returns:
            Detached header, reread on every access because xarray attrs are
            mutable in place.
        """
        return attrs.create_header(self._data)

    @overload
    def rebase(
        self,
        *models: AttrsModel | AttrsHeader | AttrsNamespace,
        target: str | Sequence[str] | None = None,
        inplace: Literal[False] = False,
        **model_kwargs: Mapping[str, Any] | None,
    ) -> DataT: ...

    @overload
    def rebase(
        self,
        *models: AttrsModel | AttrsHeader | AttrsNamespace,
        target: str | Sequence[str] | None = None,
        inplace: Literal[True],
        **model_kwargs: Mapping[str, Any] | None,
    ) -> None: ...

    def rebase(
        self,
        *models: AttrsModel | AttrsHeader | AttrsNamespace,
        target: str | Sequence[str] | None = None,
        inplace: bool = False,
        **model_kwargs: Mapping[str, Any] | None,
    ) -> DataT | None:
        """Return a copy of this object carrying the supplied attrs, as `attrs.rebase`.

        A DataTree writes only its root; groups are rebased through
        `stack["<group>"].gs.rebase`.

        Args:
            *models: Model instances to apply to `target`, one `AttrsNamespace`
                to patch onto it, or one `AttrsHeader` to restore.
            target: Variable or coordinate name the models describe, or
                several of them. None writes to the object's own attrs.
            inplace: Write into this object rather than returning a new one.
            **model_kwargs: Model name mapped to its field values, or to None
                to drop that model.

        Returns:
            New object carrying the attrs without copying pixel data, or None
            when `inplace` is set.

        Raises:
            KeyError: A keyword names no GeoSave attrs model.
            ValueError: `target` names neither a variable nor a coordinate, or
                an attrs value belongs to another scope than its target.
            ValidationError: A supplied value does not satisfy its field.

        Examples:
            >>> ds.gs.rebase(ACDD(title="Sentinel-2 Level-2A"))
            >>> ds.gs.rebase(cf_variable={"units": "1"}, target="B04")
            >>> joined.gs.rebase(attrs.merge(rasters))
        """
        if inplace:
            attrs.rebase(
                self._data, *models, target=target, inplace=True, **model_kwargs
            )
            return None
        return attrs.rebase(
            self._data, *models, target=target, inplace=False, **model_kwargs
        )

    @property
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage.

        A resampled axis carries a `TimeSpec`, so a label standing for a month
        covers that month. An axis carrying none covers what its labels spell:
        `2018-12-26` is a whole date, `2018-12-26T10:30:31` one second.

        Returns:
            First and last covered instant, or None for timeless data.
        """
        coords = getattr(self._data, "coords", None)
        if coords is None:
            return None
        if TIME_COORDINATE not in coords:
            return None
        labels = coords[TIME_COORDINATE].values
        # A merge that dropped a disagreeing cadence leaves the model behind.
        spec = attrs.TimeSpec.from_attrs(coords[TIME_COORDINATE].attrs)
        if spec is not None and spec.time_freq is not None:
            return spec.timespan(labels)

        covered = [
            parse_daterange(str(spelled))
            for spelled in np.datetime_as_string(np.atleast_1d(labels), unit="auto")
        ]
        return min(start for start, _ in covered), max(end for _, end in covered)

    @property
    def times(self) -> pd.DatetimeIndex | None:
        """Read the time coordinate labels.

        Returns:
            Labels in axis order, or None for timeless data.

        Examples:
            >>> ds.gs.times.strftime("%Y%m%d").tolist()
            ['20250601', '20250611']
        """
        # ty misbinds the coords property on the bounded type parameter.
        data = cast(xr.Dataset | xr.DataArray | xr.DataTree, self._data)
        coords = data.coords
        if TIME_COORDINATE not in coords:
            return None
        return pd.DatetimeIndex(np.atleast_1d(coords[TIME_COORDINATE].values))

    @property
    def anchor(self) -> GeoAnchor:
        """Read exact spatial and temporal coverage.

        Returns:
            Anchor over this object's grid and time span, which names its
            centroid and place and formats both into a filename.

        Raises:
            ValueError: This object carries no locatable grid.

        Examples:
            >>> ds["ndvi"].gs.anchor.format("{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}")
            '45.00N_13.00E_20250601'
        """
        from odc.geo.geobox import GeoBox

        from .anchor import GeoAnchor

        geobox = self.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(f"{type(self._data).__name__} carries no locatable grid")
        return GeoAnchor(geobox, timespan=self.timespan)

    def unpack(self) -> DataT:
        """Read physical values out of stored digital numbers.

        Returns:
            New object of physical values, a variable unchanged where it
            carries neither `scale_factor` nor `add_offset`.

        Examples:
            >>> ds.gs.unpack().B04.max().item()
            0.09
        """
        from geosave_engine.geodata.transform import packing

        return packing.unpack(self._data)

    def mask_and_scale(self) -> DataT:
        """Decode stored nodata, scale, and offset into physical values.

        Nodata is masked before scaling so its stored value cannot become
        an ordinary reading. Already decoded variables pass through.

        Returns:
            New object of the same kind, preserving coordinates, descriptive
            metadata, and laziness. Decoded storage metadata leaves attrs.

        Examples:
            >>> reflectance = ds.gs.mask_and_scale()
            >>> reflectance.red.dtype
            dtype('float32')
        """
        from geosave_engine.geodata.transform import nodata, packing

        return packing.unpack(nodata.to_nan(self._data))

    def mask(
        self, valid: xr.DataArray | np.ndarray, *, fill: float | int | None = None
    ) -> DataT:
        """Make nodata every pixel `valid` does not keep.

        Args:
            valid: Boolean array, True where a pixel is real data. A DataArray
                names its own axes and may span fewer, broadcasting over the
                rest; a bare numpy array is read as the grid alone.
            fill: Value the blanked pixels take, also written where a variable
                carries no fill value yet. None reads what each one carries.

        Returns:
            New object, its unkept pixels holding their variable's fill value.

        Raises:
            ValueError: `valid` is not boolean, does not span the grid, spans an
                axis this object does not, `fill` does not fit a dtype, or a
                variable carries no fill value and none is given.

        Examples:
            >>> clear = ds.gs.mask(ds.scl.isin([4, 5, 6, 7]))
        """
        from geosave_engine.geodata.transform import nodata

        return nodata.mask(self._data, valid, fill=fill)

    def to_nan(self) -> DataT:
        """Replace each variable's fill value with NaN.

        Returns:
            New object holding NaN where the pixels were nodata, a variable
            unchanged where it carries no fill value.

        Examples:
            >>> ds.gs.to_nan().red.dtype
            dtype('float32')
        """
        from geosave_engine.geodata.transform import nodata

        return nodata.to_nan(self._data)

    def reproject(
        self,
        target: Target,
        *,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | None = None,
    ) -> DataT:
        """Warp pixels onto the grid a target names.

        A target grid is adopted whole — CRS, resolution, and extent — while a
        bare CRS only decides the projection, sizing the grid from this object.

        Args:
            target: Grid to land on, a raster already on one, or a CRS.
            resampling: One GDAL kernel for every variable, or a mapping naming
                each variable's own, which `"*"` answers the rest of.
            resolution: Output pixel size, taken only for a CRS target. None
                keeps the ground sampling as closely as the new CRS allows.

        Returns:
            New object on the target grid, carrying its own `spatial_ref` and
            the CF semantics its axes earn.

        Raises:
            ValueError: This object or `target` sits on no locatable grid,
                `resolution` contradicts a target grid, a variable does not
                span the grid, or `resampling` would blend class codes.

        Examples:
            >>> ds.gs.reproject("EPSG:3857").gs.crs.epsg
            3857
            >>> srtm.gs.reproject(scene, resampling="bilinear")
        """
        from geosave_engine.geodata.transform import warp

        return warp.reproject(
            self._data, target, resampling=resampling, resolution=resolution
        )

    def crop(
        self, region: Geometry | gpd.GeoSeries | gpd.GeoDataFrame, *, mask: bool = True
    ) -> DataT:
        """Cut this object down to the extent of a region.

        Args:
            region: Geometry, or geometries, to cut against, reprojected onto
                this object's CRS where they sit in another. Disconnected
                geometries share one cut.
            mask: Also make nodata the pixels outside the geometries, which
                then take their own variable's fill value.

        Returns:
            New object covering the region's extent, its dtype unchanged.

        Raises:
            TypeError: `region` carries no CRS, as a bare shapely geometry.
            ValueError: This object carries no CRS, `region` is empty or does
                not overlap it, or `mask` is set while a variable carries no
                fill value.

        Examples:
            >>> ds.gs.crop(field_boundaries)
            >>> ds.gs.crop(box(10, 10, 30, 30, "EPSG:32633"))
        """
        from geosave_engine.geodata.transform import vector as vectors

        return vectors.crop(self._data, region, mask=mask)

"""Spatial and attrs properties shared by every `gs` accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.utils.datetime import parse_daterange

from .profile import TIME_COORDINATE

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import torch
    from numpy.typing import DTypeLike
    from odc.geo import CRS, BoundingBox, Resolution
    from odc.geo.gcp import GCPGeoBox
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.attrs import AttrsHeader, AttrsModel
    from geosave_engine.geodata.utils.datetime import DateRange

    from .anchor import GeoAnchor


def tensor(
    pixels: Callable[[DTypeLike], np.ndarray], dtype: torch.dtype | None
) -> torch.Tensor:
    """Build a model-input tensor from pixels read in a torch-compatible dtype.

    Args:
        pixels: Reads the pixels in the numpy dtype it is given.
        dtype: Tensor dtype, which the pixels are also read in. None casts to
            `torch.float32`, which keeps unsigned imagery off `torch.uint16` —
            a dtype torch accepts and carries no arithmetic kernels for. A
            dtype numpy cannot hold, such as `torch.bfloat16`, is read as
            float32 and narrowed on the way out.

    Returns:
        Tensor over contiguous pixels, in `dtype`.
    """
    import torch

    target = dtype or torch.float32
    try:
        reading_dtype = torch.empty(0, dtype=target).numpy().dtype
    except TypeError:
        reading_dtype = np.dtype(np.float32)
    return torch.as_tensor(np.ascontiguousarray(pixels(reading_dtype)), dtype=target)


class GeoAccessor[DataT: xr.Dataset | xr.DataArray | xr.DataTree]:
    """Spatial and attrs properties every `gs` accessor reads the same way.

    Concrete accessors bind their xarray object to `_data`; this class is
    never built directly.
    """

    _data: DataT

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
        *models: AttrsModel,
        target: str | Sequence[str] | None = None,
        inplace: Literal[False] = False,
        **model_kwargs: Mapping[str, Any] | None,
    ) -> DataT: ...

    @overload
    def rebase(
        self,
        *models: AttrsModel,
        target: str | Sequence[str] | None = None,
        inplace: Literal[True],
        **model_kwargs: Mapping[str, Any] | None,
    ) -> None: ...

    def rebase(
        self,
        *models: AttrsModel,
        target: str | Sequence[str] | None = None,
        inplace: bool = False,
        **model_kwargs: Mapping[str, Any] | None,
    ) -> DataT | None:
        """Return a copy of this object carrying the supplied attrs.

        A DataTree writes only its root; groups are rebased through
        `stack["<group>"].gs.rebase`.

        Args:
            *models: Model instances to apply to `target`.
            target: Variable or coordinate name the models describe, or
                several of them. None writes to the object's own attrs.
            inplace: Write into this object rather than returning a new one.
            **model_kwargs: Model name mapped to its field values, or to None
                to drop that model.

        Returns:
            New object carrying the attrs without copying pixel data, or None
            when `inplace` is set.

        Raises:
            KeyError: A keyword names no registered model.
            ValueError: `target` names neither a variable nor a coordinate.
            ValidationError: A supplied value does not satisfy its field.

        Examples:
            >>> ds.gs.rebase(ACDD(title="Sentinel-2 Level-2A"))
            >>> ds.gs.rebase(cf={"units": "1"}, target="B04")
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
        spec = self.attrs.coords[TIME_COORDINATE].get(attrs.TimeSpec)
        if spec is not None and spec.time_freq is not None:
            return spec.timespan(labels)

        covered = [
            parse_daterange(str(spelled))
            for spelled in np.datetime_as_string(np.atleast_1d(labels), unit="auto")
        ]
        return min(start for start, _ in covered), max(end for _, end in covered)

    @property
    def anchor(self) -> GeoAnchor:
        """Read exact spatial and temporal coverage.

        Returns:
            Anchor over this object's grid and time span, which names its
            centroid, filename stem, and place.

        Raises:
            ValueError: This object carries no locatable grid.

        Examples:
            >>> ds["ndvi"].gs.anchor.stem
            '13.0016E_45.0011N_5.12kmx5.12km_10m'
        """
        from odc.geo.geobox import GeoBox

        from .anchor import GeoAnchor

        geobox = self.geobox
        if not isinstance(geobox, GeoBox):
            raise ValueError(f"{type(self._data).__name__} carries no locatable grid")
        return GeoAnchor(geobox, timespan=self.timespan)

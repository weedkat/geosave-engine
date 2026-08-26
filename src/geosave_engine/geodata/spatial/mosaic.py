"""GeoMosaic: lay rasters covering different footprints onto one surface. See GeoMosaic."""
from __future__ import annotations

from functools import reduce
from typing import Literal

import numpy as np
from affine import Affine
from odc.geo.geobox import GeoBox, pixel_translation
from odc.geo.xr import xr_coords

from geosave_engine.geodata.utils.array import mask_nodata
from geosave_engine.geodata.utils.spatial.align import validate_rasters

from .header import GeoHeader
from .raster import GeoRaster
from .vector import GeoVector

# Which raster wins where two overlap: "first" keeps the earliest added, "last" the latest.
MosaicMethod = Literal["first", "last"]
_GRID_TOL_PX = 1e-6


class GeoMosaic:
    """Lay rasters covering different footprints onto one surface — the union of what they cover.

    The destination is discovered from the inputs, so every raster has to be
    added before `result`. Adding stays lazy for dask-backed rasters. Touches
    `y`/`x` only. For tiles cut from one known surface use `GeoStitcher`.

    Args:
        method: Overlap rule — "first" keeps the earliest added raster's
            real data, "last" prefers the latest added.

    Examples:
        >>> mosaic = GeoMosaic()
        >>> for scene in scenes:
        ...     mosaic.add(scene)
        >>> mosaic.result().to_cog("region.tif")
    """

    def __init__(
        self,
        *,
        method: MosaicMethod = "first",
    ) -> None:
        if method not in ("first", "last"):
            raise ValueError(f"method must be 'first' or 'last', got {method!r}")
        self._method = method
        self._rasters: list[GeoRaster] = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}(method={self._method!r}, rasters={len(self._rasters)})"

    def __len__(self) -> int:
        """How many rasters have been added."""
        return len(self._rasters)

    def add(self, *rasters: GeoRaster) -> None:
        """Take rasters into the mosaic, in priority order (see `method`).

        Args:
            *rasters: Rasters to lay in. Held lazily — no pixel is read here.

        Raises:
            TypeError: An input is not a GeoRaster.
        """
        for position, raster in enumerate(rasters):
            if not isinstance(raster, GeoRaster):
                raise TypeError(
                    f"GeoMosaic.add() expects GeoRaster inputs; argument {position} is "
                    f"{type(raster).__name__}"
                )
        self._rasters.extend(rasters)

    def result(self) -> GeoRaster:
        """Combine everything added into one surface.

        Returns:
            GeoRaster covering the union footprint. Pixels no input
            covers come back as nodata.
            Tags, extensions and `timespec` are the first raster's;
            `vector` is every input's combined; `tiling` is cleared.

        Raises:
            ValueError: Nothing was added; the first raster has no nodata;
                or inputs disagree on dtype, nodata, bands, time, CRS,
                pixel size, orientation, or pixel-grid origin.
        """
        if not self._rasters:
            raise ValueError("GeoMosaic.result() needs at least one raster — add() some first")

        # checked first, so "you set none at all" doesn't read as "yours differs from mine"
        nodata = self._rasters[0].nodata
        if nodata is None:
            raise ValueError(
                "GeoMosaic fills the gaps between footprints with the first raster's nodata, "
                "which isn't set — rebase(nodata=...) it first"
            )
        if len(self._rasters) == 1:
            return self._rasters[0].rebase(tiling=None)

        pieces = tuple(self._rasters)
        validate_rasters(
            *pieces,
            bands=True,
            times=True,
            operation="GeoMosaic",
        )
        reference = pieces[0]
        aligned = [reference]
        for raster in pieces[1:]:
            aligned.append(_match_grid(reference, raster))
        union = reduce(lambda left, right: left | right, (raster.anchor.geobox for raster in aligned))
        union_coords = xr_coords(union, always_yx=True)

        # combine_first takes the next array's pixel wherever this one is null, so nodata has to read as NaN first
        masked = [
            mask_nodata(raster.data).reindex(
                y=union_coords["y"].values,
                x=union_coords["x"].values,
            )
            for raster in aligned
        ]
        ordered = masked if self._method == "first" else masked[::-1]
        merged = reduce(lambda winner, filler: winner.combine_first(filler), ordered)

        restored = merged if np.isnan(nodata) else merged.fillna(nodata)
        return aligned[0]._rebase(
            data=restored.astype(aligned[0].data.dtype).rio.write_nodata(nodata),
            header=GeoHeader.combine(*(raster.header for raster in aligned)),
            vector=GeoVector.concat(*(raster.vector for raster in aligned)),
        )


def _match_grid(reference: GeoRaster, raster: GeoRaster) -> GeoRaster:
    """Validate and snap one raster onto a mosaic's pixel-grid phase."""
    reference_grid = reference.anchor.geobox
    grid = raster.anchor.geobox
    if grid.crs != reference_grid.crs:
        raise ValueError(
            f"GeoMosaic needs CRS {reference.crs}; raster at {raster.stem} uses {raster.crs}. "
            "Reproject inputs onto one CRS and resolution explicitly before add()"
        )

    try:
        offset = pixel_translation(grid, reference_grid)
    except ValueError as error:
        reference_basis = tuple(reference_grid.affine[index] for index in (0, 1, 3, 4))
        basis = tuple(grid.affine[index] for index in (0, 1, 3, 4))
        raise ValueError(
            f"GeoMosaic needs pixel basis {reference_basis}; raster at {raster.stem} uses {basis}. "
            "Reproject inputs onto one resolution and orientation explicitly before add()"
        ) from error

    tx, ty = offset.xy
    if any(abs(value - round(value)) > _GRID_TOL_PX for value in (tx, ty)):
        raise ValueError(
            f"GeoMosaic needs origins on one pixel grid; raster at {raster.stem} starts "
            f"{tx:.6f}, {ty:.6f} pixels from the first raster's origin. "
            "Reproject inputs onto one shared pixel grid explicitly before add()"
        )

    snapped = GeoBox(
        grid.shape,
        reference_grid.affine * Affine.translation(round(tx), round(ty)),
        reference_grid.crs,
    )
    if grid == snapped:
        return raster
    return raster.rebase(data=raster.data.assign_coords(xr_coords(snapped, always_yx=True)))

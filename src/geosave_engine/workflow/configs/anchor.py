"""Serializable inputs that open native geospatial anchors."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, PositiveInt, model_validator

from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata import io

from .base import ConfigModel


class CoordinateAnchorConfig(ConfigModel):
    """Build a grid around one WGS84 coordinate."""

    kind: Literal["coordinates"]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    shape: PositiveInt | tuple[PositiveInt, PositiveInt]
    resolution: Annotated[float, Field(gt=0)]
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None

    def open(self) -> GeoAnchor:
        """Build the configured native anchor."""
        return GeoAnchor.from_coordinates(
            **self.model_dump(exclude={"kind"}, exclude_none=True)
        )


class GeoJSONAnchorConfig(ConfigModel):
    """Build a grid covering one local GeoJSON file."""

    kind: Literal["geojson"]
    path: Path
    shape: PositiveInt | tuple[PositiveInt, PositiveInt] | None = None
    resolution: Annotated[float, Field(gt=0)] | None = None
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None
    pad: Annotated[float, Field(ge=0)] = 0

    @model_validator(mode="after")
    def require_grid_size(self) -> Self:
        """Require one grid sizing method."""
        if (self.shape is None) == (self.resolution is None):
            raise ValueError("GeoJSON anchor needs shape or resolution")
        return self

    def open(self) -> GeoAnchor:
        """Read the geometry and build its native anchor."""
        return GeoAnchor.from_geometry(
            io.read_vector(self.path).gs.footprint,
            **self.model_dump(exclude={"kind", "path"}, exclude_none=True),
        )


class RasterAnchorConfig(ConfigModel):
    """Read the exact grid and time from one raster."""

    kind: Literal["raster"]
    path: str | Path

    def open(self) -> GeoAnchor:
        """Read the raster's native anchor."""
        with io.read_raster(self.path) as raster:
            return raster.gs.anchor


AnchorConfig = Annotated[
    CoordinateAnchorConfig | GeoJSONAnchorConfig | RasterAnchorConfig,
    Field(discriminator="kind"),
]

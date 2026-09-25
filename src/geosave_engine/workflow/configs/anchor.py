"""Primitive spatial parameters converted to native anchors on demand."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, PositiveInt, field_validator, model_validator

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.utils import io

from .base import ConfigModel


class CoordinateAnchorConfig(ConfigModel):
    """Grid centred on one WGS84 coordinate."""

    kind: Literal["coordinates"]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    shape: PositiveInt | tuple[PositiveInt, PositiveInt]
    resolution: Annotated[float, Field(gt=0)]
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None

    def open(self) -> GeoAnchor:
        """Build the native anchor without mutating this config."""
        return GeoAnchor.from_coordinates(
            **self.model_dump(exclude={"kind"}, exclude_none=True)
        )


class GeoJSONAnchorConfig(ConfigModel):
    """Grid covering a geometry read from one local GeoJSON file."""

    kind: Literal["geojson"]
    path: Path
    shape: PositiveInt | tuple[PositiveInt, PositiveInt] | None = None
    resolution: Annotated[float, Field(gt=0)] | None = None
    crs: str | None = None
    timespan: str | tuple[str, str] | None = None
    pad: Annotated[float, Field(ge=0)] = 0

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: object) -> Path:
        """Require local GeoJSON syntax without opening the file."""
        if "://" in str(value):
            raise ValueError("GeoJSON anchor requires a local path")
        path = Path(value)
        if path.suffix.lower() not in (".json", ".geojson"):
            raise ValueError("GeoJSON anchor path must end in .json or .geojson")
        return path

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        """Require exactly one native grid sizing method."""
        if (self.shape is None) == (self.resolution is None):
            raise ValueError(
                "GeoJSON anchor requires exactly one of shape or resolution"
            )
        return self

    def open(self) -> GeoAnchor:
        """Read the geometry and build its native anchor."""
        geometry = io.read_vector(self.path).footprint
        return GeoAnchor.from_geometry(
            geometry,
            **self.model_dump(exclude={"kind", "path"}, exclude_none=True),
        )


AnchorConfig = Annotated[
    CoordinateAnchorConfig | GeoJSONAnchorConfig,
    Field(discriminator="kind"),
]

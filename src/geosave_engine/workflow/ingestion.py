"""Bind model requirements to native STAC sources for one spatial sample."""

from collections.abc import Mapping
from copy import copy

import xarray as xr

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.stac.source import StacSource, StacSourceConfig

from .spec import RasterRequirement


def stac_config(
    requirement: RasterRequirement,
    *,
    defaults: StacSourceConfig | None = None,
) -> StacSourceConfig:
    """Select required bands while preserving the caller's STAC load settings.

    Args:
        requirement: Raw raster requirements from `spec.sources[name]`.
        defaults: Existing settings, including chunking and metadata capture.

    Returns:
        A new native configuration with bands in the required order.

    Examples:
        >>> config = stac_config(spec.sources["optical"])
        >>> config.bands
        ('nir', 'red')
    """
    requirement = RasterRequirement.model_validate(requirement)
    settings = defaults if defaults is not None else StacSourceConfig()
    return StacSourceConfig.model_validate(
        {**settings.model_dump(), "bands": requirement.variables}
    )


def acquire(
    sources: Mapping[str, StacSource],
    anchor: GeoAnchor,
    *,
    requirements: Mapping[str, RasterRequirement] | None = None,
) -> xr.DataTree:
    """Load named STAC rasters into a native stack.

    Requirements select sources and bands and validate acquired metadata. Without
    requirements, every source uses its own configuration. Caller-owned sources
    and their settings remain unchanged.

    Args:
        sources: Configured STAC sources keyed by raw source name.
        anchor: Explicit spatial grid and optional time range to acquire.
        requirements: Optional raster requirements keyed by source name. Extra
            source bindings are unused.

    Returns:
        Native DataTree with one acquired raster per source group.

    Raises:
        ValueError: Bindings are missing or an acquired raster is incompatible.
        AnchorFetchError: A source search matches no items.

    Examples:
        >>> raw = acquire(
        ...     {"optical": optical_source}, anchor, requirements=spec.sources
        ... )
        >>> raw.gs.rasters["optical"]
        <xarray.Dataset> ...
    """
    if requirements is not None:
        if not requirements:
            raise ValueError(
                "At least one source requirement is required when requirements "
                "are provided"
            )
        if missing := requirements.keys() - sources.keys():
            raise ValueError(f"Source bindings are missing: {sorted(missing)}")
    if not sources:
        raise ValueError("At least one source is required")

    rasters = {}
    for name in sources if requirements is None else requirements:
        source = sources[name]
        requirement = (
            None
            if requirements is None
            else RasterRequirement.model_validate(requirements[name])
        )
        config = (
            StacSourceConfig.model_validate(source.config.model_dump())
            if requirement is None
            else stac_config(requirement, defaults=source.config)
        )
        raster = _acquire_raster(source, anchor, config=config)
        if requirement is not None:
            try:
                requirement.validate_raster(raster)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Source raster {name!r}: {error}") from error
            raster = raster[list(requirement.variables)]
        rasters[name] = raster
    return stack(rasters)


def _acquire_raster(
    source: StacSource,
    anchor: GeoAnchor,
    *,
    config: StacSourceConfig,
) -> xr.Dataset:
    """Load one source using resolved settings without mutating the caller's source."""
    bound = copy(source)
    bound.config = config
    return bound.load(anchor)

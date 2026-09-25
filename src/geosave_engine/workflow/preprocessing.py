"""Validate source rasters and execute independent, lazy Dataset recipes."""

from collections.abc import Mapping

from odc.geo.geobox import GeoBox
import xarray as xr

from geosave_engine.geodata.core.profile import CRS_COORDINATE
from geosave_engine.geodata.core.stack import stack

from .spec import ModelSpec, RasterRequirement


def preprocess(
    raw: Mapping[str, xr.Dataset] | xr.DataTree, *, spec: ModelSpec
) -> xr.DataTree:
    """Preserve selected sources and add recipe outputs to a native raster stack.

    Args:
        raw: Source Datasets by name, or a flat DataTree.
        spec: Model requirements and importable Dataset recipes.

    Returns:
        DataTree containing selected sources and each declared recipe output.

    Raises:
        ValueError: Sources, references, or callable arguments are invalid.
        TypeError: An operation does not return a Dataset.
    """
    resolved = spec.validated_copy()
    order = resolved.preparation_order()
    for recipe in resolved.preprocessing.values():
        for operation in recipe.operations:
            operation.resolve(xr.Dataset())
    if isinstance(raw, xr.DataTree):
        raw = raw.gs.rasters
    prepared = _select_rasters(resolved.sources, raw, scope="Source")
    for name, recipe in resolved.preprocessing.items():
        if recipe.raster in prepared:
            _select_variables(prepared[recipe.raster], recipe.variables, name=name)
    reference = next(iter(prepared.values())).gs.geobox
    if isinstance(reference, GeoBox) and all(
        raster.gs.geobox == reference for raster in prepared.values()
    ):
        # Native stacks promote a common source grid to their root. Reserve
        # these coordinate names before any recipe can execute.
        root_coordinates = {*reference.dimensions, CRS_COORDINATE}
        collision = (prepared.keys() | resolved.preprocessing.keys()) & root_coordinates
        if collision:
            raise ValueError(
                f"Raster names collide with shared root coordinates: {sorted(collision)}; "
                "rename these source or preprocessing groups"
            )
    for name in order:
        recipe = resolved.preprocessing[name]
        current = _select_variables(
            prepared[recipe.raster], recipe.variables, name=name
        ).copy(deep=True)
        for operation in recipe.operations:
            arguments = {
                parameter: prepared[raster].copy(deep=True)
                for parameter, raster in operation.inputs.items()
            }
            current = operation.apply(current, inputs=arguments)
            if not isinstance(current, xr.Dataset):
                raise TypeError(
                    f"Recipe {name!r} operation {operation.method or operation.call!r} must return a Dataset"
                )
        prepared[name] = current
    return stack(prepared)


def _select_variables(
    raster: xr.Dataset, variables: tuple[str, ...] | None, *, name: str
) -> xr.Dataset:
    """Select recipe variables before running its operations."""
    if variables is None:
        return raster
    if missing := set(variables) - raster.data_vars.keys():
        raise ValueError(
            f"Recipe {name!r} selects missing variables: {sorted(missing)}"
        )
    return raster[list(variables)]


def _select_rasters(
    requirements: Mapping[str, RasterRequirement],
    rasters: Mapping[str, xr.Dataset],
    *,
    scope: str,
) -> dict[str, xr.Dataset]:
    """Validate all selected rasters before any processing operation runs."""
    if missing := requirements.keys() - rasters.keys():
        raise ValueError(f"{scope} rasters are missing: {sorted(missing)}")
    selected = {}
    for name, requirement in requirements.items():
        raster = rasters[name]
        try:
            requirement.validate_raster(raster)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{scope} raster {name!r}: {error}") from error
        selected[name] = raster[list(requirement.variables)].copy(deep=True)
    return selected

"""EO extension: what each band is called and where in the spectrum it sits."""

from __future__ import annotations

import pystac
import xarray as xr
from pystac.extensions.eo import Band, EOExtension

from geosave_engine.geodata.attrs import AttrsModel, CFVariable, Spectral


def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Write one EO band per variable where the raster is spectral.

    A label or a DEM is not an optical band, so a raster none of whose
    variables carries `Spectral` writes nothing; its names stay in the file.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `read_raster` opens it.

    Examples:
        >>> write(asset, scene)
        >>> EOExtension.ext(asset).bands[0].common_name
        'red'
    """
    header = raster.gs.attrs
    variables = {name: header.data_vars[name] for name in raster.gs.variables}
    if all(variable.get(Spectral) is None for variable in variables.values()):
        return

    bands = []
    for name, variable in variables.items():
        facts = variable.get(Spectral)
        semantics = variable.get(CFVariable)
        bands.append(
            Band.create(
                name=name,
                common_name=None if facts is None else facts.common_name,
                center_wavelength=None if facts is None else facts.center_wavelength,
                full_width_half_max=None
                if facts is None
                else facts.full_width_half_max,
                description=None if semantics is None else semantics.long_name,
            )
        )
    EOExtension.ext(asset, add_if_missing=True).apply(bands)


def read(band: Band) -> list[AttrsModel]:
    """Read the attrs models one EO band states.

    Args:
        band: One band of an asset's EO listing.

    Returns:
        `Spectral` where the band states a common name or a wavelength, then
        `CFVariable` carrying its description as the long name. Empty where
        it states only its name.

    Examples:
        >>> read(Band.create(name="B04", common_name="red"))
        [Spectral(common_name='red', center_wavelength=None, full_width_half_max=None)]
    """
    facts = {
        "common_name": band.common_name,
        "center_wavelength": band.center_wavelength,
        "full_width_half_max": band.full_width_half_max,
    }
    models: list[AttrsModel] = []
    if any(value is not None for value in facts.values()):
        models.append(Spectral(**facts))
    if band.description is not None:
        models.append(CFVariable(long_name=band.description))
    return models

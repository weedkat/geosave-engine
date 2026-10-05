"""What a stacked DataArray's one attrs mapping cannot hold."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BeforeValidator, ConfigDict

from geosave_engine.geodata.attrs.model import AttrsModel, parse_collection_text


class StackedAttrs(AttrsModel):
    """Keep the attrs a stacked array's root cannot hold, on its band axis.

    The root of a stacked array holds what every band shares; this holds the
    rest, so `GeoArray.to_raster` restores each band and the Dataset root.

    Args:
        variable_attrs: Band name mapped to the attrs not every band shares.
        dataset_attrs: Attrs of the Dataset the bands were stacked from.

    Examples:
        >>> stacked = ds.gs.to_array()
        >>> stacked.gs.attrs.coords["band"].get(StackedAttrs).variable_attrs
        {'B04': {'scale_factor': 0.0001}, 'B03': {}}
    """

    NAME: ClassVar[str] = "stacked_attrs"
    model_config = ConfigDict(ser_json_inf_nan="constants")

    variable_attrs: Annotated[
        dict[str, dict[str, object]] | None,
        BeforeValidator(parse_collection_text),
    ] = None
    dataset_attrs: Annotated[
        dict[str, object] | None,
        BeforeValidator(parse_collection_text),
    ] = None

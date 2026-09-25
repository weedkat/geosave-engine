"""What a Zarr store cannot say about itself."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BeforeValidator

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.geodata.attrs.validate import parse_collection_text


class ZarrOrder(AttrsModel):
    """State the order a Zarr store's variables were written in.

    A Zarr group holds its members as a mapping, so the store records no order
    and two reads of one store can disagree. The order is written here instead,
    and `utils.io.zarr.read` restores it.

    Args:
        zarr_variable_order: Data variable names, in the order they were
            written.

    Examples:
        >>> ds.gs.attrs.root.get(ZarrOrder)
        ZarrOrder(zarr_variable_order=('B04', 'B03', 'B02'))
    """

    NAME: ClassVar[str] = "zarr"

    zarr_variable_order: Annotated[
        tuple[str, ...] | None, BeforeValidator(parse_collection_text)
    ] = None

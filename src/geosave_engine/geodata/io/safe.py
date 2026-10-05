"""Read an ESA SAFE product, a fixed tree whose leaves name their own band.

A product is a structure rather than an encoding, so each leaf opens through
the format module its suffix names, and this module owns only the tree, the
naming convention, and the sidecar metadata.

Examples:
    A Sentinel-2 Level-2A product::

        S2A_MSIL2A_20250601T103031_N0511_R108_T33UUP_....SAFE/
          MTD_MSIL2A.xml                 product metadata
          manifest.safe                  package manifest and checksums
          GRANULE/L2A_T33UUP_A051234_20250601T103031/
            MTD_TL.xml                   tile metadata, angle grids
            QI_DATA/                     cloud and quality masks
            IMG_DATA/
              R10m/T33UUP_20250601T103031_B04_10m.jp2
              R20m/T33UUP_20250601T103031_B05_20m.jp2
              R60m/T33UUP_20250601T103031_B01_60m.jp2

A leaf is named `T{tile}_{instant}_{band}_{resolution}m.jp2`, so its band and
instant are read from the filename rather than from metadata inside the file.
This is what separates a vendor product from the trees `layout` writes, whose
leaves name their own variables in their band descriptions.

The resolution groups sit on different grids, so a whole product is a stack
rather than a cube: `read_stack` gives one group per resolution, and `read`
takes the one resolution asked for. Implementing either also means translating
`MTD_MSIL2A.xml` through a header factory, the way `attrs/headers/gdal` and
`attrs/headers/geobox` translate their own sources.

Other vendors arrange products the same way over different encodings: Landsat
Collection 2 is GeoTIFF leaves beside an `MTL.xml`, and Element84's Sentinel-2
is COG leaves beside STAC JSON. A second product reader is what would make the
shared parts worth factoring out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset, DataTree

_UNBUILT = (
    "reading a SAFE product is not built yet; open its JP2 leaves with "
    "`io.gdal.read` and combine them yourself, or convert the product to a "
    "COG tree and read it with `io.read_tree`"
)


def read(
    source: str | PathLike[str], *, resolution: int = 10, **rio_options: Any
) -> Dataset:
    """Read one resolution group of a SAFE product as a cube.

    Args:
        source: Product directory ending in `.SAFE`.
        resolution: Ground sample distance in metres, naming the `R{n}m` group.
        **rio_options: Open options passed to every leaf.

    Returns:
        Cube holding that group's bands on its one grid.

    Raises:
        NotImplementedError: Always; reading a SAFE product is unbuilt.
    """
    raise NotImplementedError(_UNBUILT)


def read_stack(source: str | PathLike[str], **rio_options: Any) -> DataTree:
    """Read a whole SAFE product as one group per resolution.

    Args:
        source: Product directory ending in `.SAFE`.
        **rio_options: Open options passed to every leaf.

    Returns:
        Stack holding one group per `R{n}m` directory, each on its own grid.

    Raises:
        NotImplementedError: Always; reading a SAFE product is unbuilt.
    """
    raise NotImplementedError(_UNBUILT)

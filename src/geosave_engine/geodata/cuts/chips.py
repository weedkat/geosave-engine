"""Cut windows in space into chips of one size."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
from affine import Affine
from odc.geo.geobox import GeoBox
from tiler import Tiler

if TYPE_CHECKING:
    from geosave_engine.geodata import GeoDataFrame

type ChipMode = Literal["reflect", "edge", "constant", "wrap"]


def layout(
    shape: tuple[int, int],
    size: tuple[int, int],
    *,
    overlap: int = 0,
    mode: ChipMode = "reflect",
) -> tuple[Tiler, list[tuple[int, int]]]:
    """Lay chips over a window, the one way both cutting and merging use.

    Overlapping chips are laid over a halo around the window, so every pixel
    of it sits away from a chip edge at least once and a tapered merge leaves
    no seam at the border.

    Args:
        shape: Height and width of the window.
        size: Height and width of one chip.
        overlap: Pixels neighbouring chips share.
        mode: How a chip reaching past the window is filled.

    Returns:
        The Tiler laying the chips, and the halo before and after each axis.
        The same arguments always give the same layout, so a merge rebuilds
        it instead of storing it.

    Examples:
        >>> tiler, halo = layout((300, 500), (128, 128), overlap=32)
        >>> len(tiler), halo
        (24, [(16, 16), (16, 16)])
    """
    tiler = Tiler(shape, size, overlap=overlap, mode=mode, constant_value=float("nan"))
    halo = [(0, 0), (0, 0)]
    if overlap:
        padded, halo = tiler.calculate_padding()
        tiler.recalculate(data_shape=padded)
    return tiler, [(int(before), int(after)) for before, after in halo]


def chips(
    windows: gpd.GeoDataFrame,
    size: int | tuple[int, int],
    *,
    overlap: int = 0,
    mode: ChipMode = "reflect",
) -> GeoDataFrame:
    """Cut each window into chips of one size.

    Args:
        windows: Windows as `stacks` or `frames` lists them.
        size: Chip side in pixels, or height and width.
        overlap: Pixels neighbouring chips share.
        mode: How a chip reaching past its window is filled.

    Returns:
        One row per chip, windows in the order given. `id` gains
        `/chip-<n>`, `parent` names the window cut, `chip` is its number in
        that window's layout, `row_off`, `col_off`, `height`, `width`,
        `transform` and `geometry` state its own pixels, and `halo`,
        `overlap` and `mode` state the layout, which a read past the window's
        edge and a later merge both use.

    Examples:
        >>> cut = chips(stacks(items), 224, overlap=32)
        >>> cut.iloc[0][["id", "chip", "height", "width"]].tolist()
        ['s0/chip-0', 0, 224, 224]
    """
    side = (size, size) if isinstance(size, int) else (size[0], size[1])
    cut = []
    for _, window in windows.iterrows():
        tiler, halo = layout(
            (int(window["height"]), int(window["width"])),
            side,
            overlap=overlap,
            mode=mode,
        )
        # Unreferenced pixels have a layout and offsets, but no footprint.
        located = isinstance(window["crs"], str)
        grid = GeoBox(
            (int(window["height"]), int(window["width"])),
            Affine(*window["transform"]),
            window["crs"] if located else None,
        )
        for number in range(len(tiler)):
            near, far = tiler.get_tile_bbox(number)
            row = int(near[0]) - halo[0][0]
            column = int(near[1]) - halo[1][0]
            height, width = (int(edge) for edge in far - near)
            chip = grid.translate_pix(column, row).crop((height, width))
            cut.append(
                {
                    **window.to_dict(),
                    "id": f"{window['id']}/chip-{number}",
                    "parent": window["id"],
                    "chip": number,
                    "row_off": int(window["row_off"]) + row,
                    "col_off": int(window["col_off"]) + column,
                    "height": height,
                    "width": width,
                    "transform": list(chip.transform)[:6],
                    "halo": [list(widths) for widths in halo],
                    "overlap": overlap,
                    "mode": mode,
                    "geometry": chip.extent.to_crs("EPSG:4326").geom
                    if located
                    else None,
                }
            )
    # No window cut leaves an empty table, which still names its columns.
    columns = list(
        dict.fromkeys([*windows.columns, *["chip", "halo", "overlap", "mode"]])
    )
    return cast(
        "GeoDataFrame",
        gpd.GeoDataFrame(cut, columns=columns, geometry="geometry", crs=windows.crs),
    )

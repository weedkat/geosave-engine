"""Cut saved samples into the windows a model reads: frames in time, chips in space.

Every cut takes a table of windows and returns a table of windows, opening no
file. `stacks` lists an item table as one whole window per saved sample;
`frames` and `chips` split windows; `select` reads one window off an opened
sample; `merge` puts chip outputs back together.

    Examples:
        >>> windows = cuts.stacks(stac.table.read("data/train/items.parquet"))
        >>> windows = cuts.chips(cuts.frames(windows, 4, tolerance="10D"), 224)
        >>> chip = cuts.select(read_stack("data/train/s0"), windows.iloc[0])
"""

from .chips import chips, layout
from .frames import frames
from .merge import merge
from .read import select, select_pixels, select_times
from .stacks import stacks

__all__ = [
    "chips",
    "frames",
    "layout",
    "merge",
    "select",
    "select_pixels",
    "select_times",
    "stacks",
]

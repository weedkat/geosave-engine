"""Generic, domain-agnostic pipeline actions — no remote-sensing science here.

For actual derived-science features (spectral indices, cloud detection, etc)
see geosave_engine.geodata.features instead.
"""

from geosave_engine.geodata.transforms.logic import intersect, union
from geosave_engine.geodata.transforms.morphology import binary_open
from geosave_engine.geodata.transforms.scale import apply_scale

__all__ = [
    "apply_scale",
    "intersect",
    "union",
    "binary_open",
]

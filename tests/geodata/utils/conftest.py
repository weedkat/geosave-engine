"""Force a headless matplotlib backend for this test package.

geovis.plot() makes its own Figure via plt.subplots() when no ax is
given — needs a backend picked before matplotlib.pyplot is ever
imported, or it defaults to an interactive one (tkagg here) that needs a
real display.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402 — must follow the MPLBACKEND env var above
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every Figure a test opened — plot() never closes its own, pyplot keeps them alive otherwise."""
    yield
    plt.close("all")

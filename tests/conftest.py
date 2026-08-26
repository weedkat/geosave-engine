from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


@pytest.fixture
def dw_tif_path() -> Path:
    """Path to the DynamicWorld anchor used as a real test fixture."""
    return Path(__file__).parent / "data" / "dw_-22.7491991582_15.9791703445-20190223.tif"

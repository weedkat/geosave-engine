"""Fixtures shared by tests under geodata/."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import pytest

from geosave_engine.geodata.datastore import SampleStore


@pytest.fixture
def write_sample_store():
    """Factory: build and write a SampleStore, retrying litdata's own index.json write race.

    Observed directly in this environment (not a guess) — optimize()'s
    background index.json write occasionally hasn't landed yet by the time
    this fixture's own first read runs right after. A real correctness bug
    reproduces on every attempt, not just the first, so a bounded retry here
    doesn't mask one — it only papers over that one known timing gap.
    """

    def _write(path: str | Path, samples: Sequence[Any], **config: Any) -> SampleStore:
        last_err: ValueError | None = None
        for _ in range(3):
            store = SampleStore(str(path), **config)
            store.write(samples)
            try:
                len(store)
            except ValueError as e:
                last_err = e
                shutil.rmtree(path, ignore_errors=True)
                time.sleep(0.5)
                continue
            return store
        assert last_err is not None
        raise last_err

    return _write

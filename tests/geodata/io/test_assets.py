"""Asset readers retain file ownership through native stack construction."""

import pytest

from geosave_engine.geodata import io
from geosave_engine.geodata.io.assets import read
from tests.geodata.conftest import build_raster


@pytest.mark.parametrize("failure", [False, True])
def test_asset_readers_close_the_files_they_open(monkeypatch, failure):
    closed = []

    def open_raster(href, **options):
        if failure and href == "missing.tif":
            raise FileNotFoundError(href)
        raster = build_raster()
        raster.set_close(lambda: closed.append(href))
        return raster

    monkeypatch.setattr(io, "read_raster", open_raster)
    assets = {"first": {"href": "first.tif"}, "second": {"href": "missing.tif"}}
    if failure:
        with pytest.raises(FileNotFoundError):
            read(assets)
        assert closed == ["first.tif"]
    else:
        with read(assets):
            assert closed == []
        assert closed == ["first.tif", "missing.tif"]

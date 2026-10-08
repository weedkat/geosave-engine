import pytest

from geosave_engine.geodata import cuts, stack
from geosave_engine.geodata.transform.concat import concat_time
from geosave_engine.geodata.transform.warp import reproject

from tests.geodata.conftest import whole_windows

from .conftest import build, geobox


@pytest.mark.parametrize("operation", ["concat", "reproject", "frames"])
def test_rebuilt_stacks_preserve_root_attrs(operation):
    source = stack({"optical": build(geobox())})
    source.attrs = {"title": "training scene", "foreign": {"site": 7}}
    if operation == "concat":
        result = concat_time([source, source])
    elif operation == "reproject":
        result = reproject(source, source.gs.geobox.zoom_out(2))
        assert result.gs.geobox == source.gs.geobox.zoom_out(2)
    else:
        (window,) = whole_windows({"scene": source}).to_dict("records")
        result = cuts.select_times(source, window)
    assert result.attrs == source.attrs

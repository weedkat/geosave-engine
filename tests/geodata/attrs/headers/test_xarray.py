import xarray as xr

from geosave_engine.geodata import attrs


def test_xarray_factory_captures_root_variables_and_coordinates() -> None:
    source = xr.Dataset(
        {"red": ("x", [1, 2], {"_FillValue": 0, "source_band": "B04"})},
        coords={"x": ("x", [0, 1], {"axis_note": "east"})},
        attrs={"title": "source", "provider_id": "scene-001"},
    )

    header = attrs.create_header(source)

    assert header.root.to_attrs() == {
        "provider_id": "scene-001",
        "title": "source",
    }
    assert header.data_vars["red"].to_attrs() == {
        "source_band": "B04",
        "_FillValue": 0,
        "nodata": 0,
    }
    assert header.coords["x"].to_attrs() == {"axis_note": "east"}


def test_dataarray_factory_keeps_own_attrs_at_the_root() -> None:
    array = xr.DataArray(
        [1],
        dims="x",
        coords={"x": ("x", [0], {"axis": "X"})},
        attrs={"units": "1"},
    )

    header = attrs.create_header(array)

    assert header.root.get(attrs.CFVariable) == attrs.CFVariable(units="1")
    assert header.data_vars == {}
    assert header.coords["x"].to_attrs() == {"axis": "X"}


def test_datatree_factory_reads_only_the_selected_node() -> None:
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"title": "root"}),
            "/child": xr.Dataset({"red": ("x", [1], {"units": "1"})}),
        }
    )

    header = attrs.create_header(tree)

    assert header.root.to_attrs() == {"title": "root"}
    assert header.data_vars == {}
    assert header.coords == {}

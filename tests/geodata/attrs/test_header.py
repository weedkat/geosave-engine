import pytest

from geosave_engine.geodata.attrs import AttrsHeader, DroppedAttr


def test_merge_reports_losses_at_each_scope_and_keeps_unshared_variables():
    first = AttrsHeader.from_attrs(
        root={"title": "scene", "source": "first"},
        data_vars={"red": {"units": "1", "source": "first"}},
        coords={
            "x": {"axis": "X", "source": "first"},
            "y": {"axis": "Y"},
        },
    )
    second = AttrsHeader.from_attrs(
        root={"title": "scene", "source": "second"},
        data_vars={
            "red": {"units": "1", "source": "second"},
            "nir": {"long_name": "near infrared"},
        },
        coords={"x": {"axis": "X", "source": "second"}},
    )

    merged, dropped = AttrsHeader.merge([first, second])

    assert merged.root.to_attrs() == {"title": "scene"}
    assert list(merged.data_vars) == ["nir", "red"]
    assert merged.data_vars["nir"].to_attrs() == {"long_name": "near infrared"}
    assert merged.data_vars["red"].to_attrs() == {"units": "1"}
    assert list(merged.coords) == ["x", "y"]
    assert merged.coords["x"].to_attrs() == {"axis": "X"}
    assert merged.coords["y"].to_attrs() == {"axis": "Y"}
    assert dropped == {
        DroppedAttr(None, "source"),
        DroppedAttr("red", "source"),
        DroppedAttr("x", "source"),
    }
    assert first.root.to_attrs() == {"title": "scene", "source": "first"}
    assert second.data_vars["red"].to_attrs() == {"units": "1", "source": "second"}


def test_merge_rejects_names_used_as_both_a_variable_and_a_coordinate():
    first = AttrsHeader.from_attrs(data_vars={"x": {"units": "1"}})
    second = AttrsHeader.from_attrs(coords={"x": {"axis": "X"}})

    with pytest.raises(ValueError, match="coordinate attrs"):
        AttrsHeader.merge([first, second])


def test_merge_requires_at_least_one_header():
    with pytest.raises(ValueError, match="at least one header"):
        AttrsHeader.merge([])

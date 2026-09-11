from __future__ import annotations

import numpy as np
import pytest

from geosave_engine.geodata.transform import variables as transform_variables

from .conftest import build, geobox


def paired(times: int = 2):
    """Build two rasters on one grid carrying different variables."""
    box = geobox()
    optical = build(box, times=times)
    other = optical[["red"]].rename_vars({"red": "nir"})
    return box, optical, other


def test_merge_gathers_every_operand_variable() -> None:
    box, optical, other = paired()

    merged = transform_variables.merge([optical, other])

    assert merged.gs.variables == ("red", "nir")
    assert merged.gs.geobox == box
    assert merged.red.dtype == np.dtype("uint16")


def test_merge_keeps_each_variable_its_own_attrs() -> None:
    _, optical, other = paired()

    merged = transform_variables.merge([optical, other])

    assert merged.red.attrs["scale_factor"] == pytest.approx(1e-4)
    assert merged.gs.geobox == optical.gs.geobox


def test_merge_resolves_the_attrs_the_operands_agree_on() -> None:
    _, optical, other = paired()

    merged = transform_variables.merge([optical, other])

    assert merged.attrs["license"] == "CC0"


def test_merge_refuses_a_shared_variable_name() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="rename one side"):
        transform_variables.merge([build(box), build(box)])


def test_merge_refuses_a_different_time_axis() -> None:
    box = geobox()
    optical = build(box, times=2)
    other = build(box, times=3)[["red"]].rename_vars({"red": "nir"})

    with pytest.raises(ValueError, match="merging does not align"):
        transform_variables.merge([optical, other])


def test_merge_refuses_a_different_grid() -> None:
    box = geobox()
    other_box = geobox((400_000.0, 5_100_000.0, 400_320.0, 5_100_320.0))
    other = build(other_box)[["red"]].rename_vars({"red": "nir"})

    with pytest.raises(ValueError, match="reproject or resample"):
        transform_variables.merge([build(box), other])


def test_merge_refuses_no_rasters() -> None:
    with pytest.raises(ValueError, match="at least one raster"):
        transform_variables.merge([])


def test_rename_replaces_names_and_keeps_attrs() -> None:
    raster = build(geobox())

    renamed = transform_variables.rename(raster, {"red": "reflectance"})

    assert renamed.gs.variables == ("reflectance",)
    assert renamed.reflectance.attrs["scale_factor"] == pytest.approx(1e-4)
    assert raster.gs.variables == ("red",)


def test_rename_refuses_an_absent_variable() -> None:
    with pytest.raises(KeyError, match="are not data variables"):
        transform_variables.rename(build(geobox()), {"missing": "red"})


def test_rename_refuses_an_empty_replacement() -> None:
    with pytest.raises(ValueError, match="empty replacement name"):
        transform_variables.rename(build(geobox()), {"red": "  "})


def test_rename_refuses_shadowing_a_coordinate() -> None:
    with pytest.raises(ValueError, match="already name coordinates"):
        transform_variables.rename(build(geobox()), {"red": "y"})


def test_rename_refuses_creating_a_duplicate() -> None:
    _, optical, other = paired()
    both = transform_variables.merge([optical, other])

    with pytest.raises(ValueError, match="more than one data variable"):
        transform_variables.rename(both, {"red": "nir"})

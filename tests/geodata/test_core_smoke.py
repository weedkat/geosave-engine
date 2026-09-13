from typing import TYPE_CHECKING, assert_type

import numpy as np
import xarray as xr

import geosave_engine.geodata as gs

from geosave_engine.geodata import GeoAnchor, GeoArray, GeoRaster, GeoStack, GeoVector
from geosave_engine.geodata.attrs import CFVariable

from .conftest import build_raster


if TYPE_CHECKING:

    def _assert_accessor_types(
        array: gs.DataArray,
        dataset: gs.Dataset,
        tree: gs.DataTree,
    ) -> None:
        assert_type(array.gs, GeoArray)
        assert_type(dataset.gs, GeoRaster)
        assert_type(tree.gs, GeoStack)
        assert_type(dataset.isel(x=0).gs, GeoRaster)


def test_raster_accessor_survives_xarray_selection(raster: xr.Dataset) -> None:
    selected = raster.isel(x=slice(0, 1))

    assert isinstance(selected.gs, GeoRaster)
    assert selected.gs.variables == ("red", "nir")
    assert selected.gs.geobox.width == 1


def test_stack_accessor_reads_settled_structure(stack: xr.DataTree) -> None:
    assert isinstance(stack.gs, GeoStack)
    assert stack.gs.groups == ("optical", "infrared")


def test_typed_variable_attrs_round_trip(raster: xr.Dataset) -> None:
    metadata = CFVariable(long_name="red reflectance", units="1")

    stamped = raster.gs.rebase(metadata, target="red")

    assert stamped.gs.attrs.data_vars["red"].get(CFVariable) == metadata


def test_unbucketed_time_covers_only_its_own_instants() -> None:
    raster = build_raster(times=2).assign_coords(
        time=np.array(
            ["2025-06-01T10:23:11", "2025-06-11T10:24:02"], dtype="datetime64[ns]"
        )
    )

    start, end = raster.gs.timespan

    assert (start.hour, start.minute, start.second) == (10, 23, 11)
    assert (end.hour, end.minute, end.second) == (10, 24, 2)


def test_timeless_raster_has_no_timespan(raster: xr.Dataset) -> None:
    assert raster.gs.timespan is None


def test_anchor_and_vector_construction() -> None:
    vector = GeoVector.from_geometry("POINT (13 52)")

    anchor = GeoAnchor.from_geometry(vector.footprint, resolution=10)

    assert anchor.geobox.crs is not None
    assert anchor.geobox.crs.to_epsg() == 32633
    assert anchor.stem == "13.0000E_52.0000N_10mx10m_10m"


def test_odc_geometry_keeps_its_crs() -> None:
    projected = GeoVector.from_geometry(
        "POINT (500000 9000000)",
        crs="EPSG:32749",
    )

    restored = GeoVector.from_geometry(projected.footprint)

    assert restored.crs.to_epsg() == 32749


def test_geodata_types_are_xarray_at_runtime() -> None:
    assert gs.DataArray is xr.DataArray
    assert gs.Dataset is xr.Dataset
    assert gs.DataTree is xr.DataTree

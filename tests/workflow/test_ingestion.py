from copy import deepcopy
from datetime import timedelta
from threading import Lock

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest

from geosave_engine.geodata.attrs import ACDD, StacMetadata
from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.stac.source import StacSourceConfig
from geosave_engine.workflow.ingestion import acquire, stac_config
import geosave_engine.workflow.ingestion as ingestion
from geosave_engine.workflow.spec import RasterRequirement


def test_stac_projection_preserves_runtime_settings(spec):
    def sign_url(url):
        return url

    defaults = StacSourceConfig(
        bands=("unused",),
        chunks={"x": 2, "y": 3},
        dtype="uint16",
        item_properties=("platform",),
        asset_fields=("raster:bands",),
        patch_url=sign_url,
    )
    config = stac_config(spec.sources["optical"], defaults=defaults)
    assert tuple(config.bands) == ("nir", "red")
    assert config.model_dump(exclude={"bands"}) == defaults.model_dump(
        exclude={"bands"}
    )
    assert config.patch_url is sign_url
    config.chunks["x"] = 100
    assert defaults.chunks["x"] == 2
    assert defaults.bands == ("unused",)
    assert stac_config(spec.sources["optical"]).chunks is not None
    assert stac_config(spec.sources["optical"]).dtype is None


@pytest.mark.parametrize("bands", [None, ("red", "nir", "unused")])
def test_positional_stac_config_preserves_runtime_bands(bands):
    defaults = StacSourceConfig(bands=bands, chunks={"x": 2})
    selected = stac_config(RasterRequirement(channels=2), defaults=defaults)
    assert selected.model_dump() == defaults.model_dump()
    selected.chunks["x"] = 9
    assert defaults.chunks["x"] == 2


@pytest.mark.parametrize(
    "selector,names",
    [
        ({"variables": ["nir", "red"]}, ["nir", "red"]),
        ({"channels": 2}, ["red", "nir"]),
    ],
)
def test_acquire_selects_and_validates_real_stac_assets(local_stac, selector, names):
    sources, anchor = local_stac
    source = sources["optical"].set_config(bands=("red", "nir", "unused"))
    requirement = RasterRequirement(
        **selector,
        dims=("time", "y", "x"),
        dtypes=("uint16",),
        attrs={
            "data_vars": {"*": {"models": {"packing": {"required": ["scale_factor"]}}}}
        },
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = acquire(sources, anchor, requirements={"optical": requirement})
    selected = result.gs.rasters["optical"]
    assert tasks == []
    assert list(selected.data_vars) == names
    assert isinstance(selected.red.data, da.Array)
    assert selected.red.dtype == np.dtype("uint16")
    assert source.config.bands == ("red", "nir", "unused")
    del source.client.items[0].assets["nir"].extra_fields["raster:bands"][0]["scale"]
    with pytest.raises(ValueError, match="optical.*nir.*scale_factor"):
        acquire(sources, anchor, requirements={"optical": requirement})


def test_acquire_selects_required_sources_and_bands_without_computing(
    monkeypatch, spec, raw, source, anchor
):
    calls = []

    def load(bound_source, bound_anchor, *, config):
        calls.append((bound_source, bound_anchor, config))
        return raw["optical"]

    monkeypatch.setattr(ingestion, "_acquire_raster", load)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = acquire(
            {"optical": source, "unused": source},
            anchor,
            requirements=spec.sources,
        )
    optical = result.gs.rasters["optical"]
    assert tasks == []
    assert result.gs.groups == ("optical",)
    assert list(optical.data_vars) == ["nir", "red"]
    assert optical.red.data is raw["optical"].red.data
    assert len(calls) == 1
    assert calls[0][:2] == (source, anchor)
    assert tuple(calls[0][2].bands) == ("nir", "red")
    assert source.config.bands == ("unused",)


def test_acquire_without_requirements_uses_each_source_configuration(
    monkeypatch, raw, source, anchor
):
    class Signer:
        def __init__(self):
            self.lock = Lock()

        def __call__(self, url):
            return url

    signer = Signer()
    source.set_config(patch_url=signer)

    def load(bound_source, bound_anchor, *, config):
        assert config == source.config
        assert config is not source.config
        assert config.patch_url is signer
        config.chunks["x"] = 100
        assert source.config.chunks["x"] == 2
        return raw["optical"]

    monkeypatch.setattr(ingestion, "_acquire_raster", load)
    result = acquire({"optical": source}, anchor)
    assert result.gs.rasters["optical"].red.data is raw["optical"].red.data


def test_acquire_rejects_missing_bindings_before_acquisition(
    monkeypatch, spec, source, anchor
):
    def unexpected(*args, **kwargs):
        pytest.fail("acquisition ran before all bindings were validated")

    monkeypatch.setattr(ingestion, "_acquire_raster", unexpected)
    with pytest.raises(ValueError, match="optical"):
        acquire({"wrong": source}, anchor, requirements=spec.sources)


def test_acquire_rejects_empty_requirements_before_acquisition(
    monkeypatch, source, anchor
):
    def unexpected(*args, **kwargs):
        pytest.fail("acquisition ran with no source requirements")

    monkeypatch.setattr(ingestion, "_acquire_raster", unexpected)
    with pytest.raises(ValueError, match="At least one source requirement"):
        acquire({"optical": source}, anchor, requirements={})


def test_acquire_checks_loaded_rasters(monkeypatch, spec, raw, source, anchor):
    monkeypatch.setattr(
        ingestion,
        "_acquire_raster",
        lambda *args, **kwargs: raw["optical"].drop_vars("nir"),
    )
    with pytest.raises(ValueError, match="optical.*nir"):
        acquire({"optical": source}, anchor, requirements=spec.sources)


@pytest.mark.parametrize("use_requirements", [False, True])
def test_acquire_loads_local_stac_assets_without_mutating_a_reusable_source(
    local_stac, spec, use_requirements
):
    sources, anchor = local_stac
    source = sources["optical"]
    original_config = source.config
    original_query = source.query
    query_settings = deepcopy(source.query.to_search_params())
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = acquire(
            {"optical": source},
            anchor,
            requirements=spec.sources if use_requirements else None,
        )
    raw = result.gs.rasters["optical"]
    assert tasks == []
    assert list(raw.data_vars) == (["nir", "red"] if use_requirements else ["unused"])
    variable = raw.nir if use_requirements else raw.unused
    assert variable.dims == ("time", "y", "x")
    assert variable.dtype == np.dtype("uint16")
    assert isinstance(variable.data, da.Array)
    assert raw.gs.geobox == anchor.geobox
    assert raw.gs.attrs.root.get(ACDD).license == "CC-BY-4.0"
    assert raw.gs.attrs.root.get(StacMetadata).properties() == ("platform",)
    assert source.config is original_config
    assert source.config.bands == ("unused",)
    assert source.query is original_query
    assert source.query.to_search_params() == query_settings
    assert source.client.requests[0].datetime == "2025-01-15"
    assert source.client.requests[0].filter == query_settings["filter"]
    assert source.client.requests[0].max_items == 1
    np.testing.assert_array_equal(
        variable.compute()[0, 1:, :], 6000 if use_requirements else 1
    )


def test_acquisition_does_not_silently_squeeze_a_stac_time_axis(local_stac, spec):
    sources, anchor = local_stac
    requirement = RasterRequirement(
        **{
            **spec.sources["optical"].model_dump(),
            "dims": ("y", "x"),
        }
    )
    with pytest.raises(ValueError, match="optical.*dimensions"):
        acquire(sources, anchor, requirements={"optical": requirement})


def test_acquisition_validates_loaded_metadata(local_stac):
    sources, anchor = local_stac
    requirement = RasterRequirement(
        variables=("nir", "red"),
        dims=("time", "y", "x"),
        require_crs=True,
        attrs={
            "data_vars": {"*": {"models": {"packing": {"required": ("scale_factor",)}}}}
        },
    )
    nir = sources["optical"].client.items[0].assets["nir"]
    del nir.extra_fields["raster:bands"][0]["scale"]
    with pytest.raises(ValueError, match="optical.*nir.*packing.*scale_factor"):
        acquire(
            {"optical": sources["optical"]},
            anchor,
            requirements={"optical": requirement},
        )


def test_acquisition_loads_native_stac_properties_as_time_coordinates(local_stac):
    sources, anchor = local_stac
    source = sources["optical"]
    first = source.client.items[0]
    first.properties["view:sun_azimuth"] = 10.0
    second = deepcopy(first)
    second.id = "optical-scene-2"
    second.datetime = first.datetime + timedelta(days=1)
    second.properties["view:sun_azimuth"] = 20.0
    source.client.items.append(second)
    source.set_config(
        with_properties=(
            {
                "key": "view:sun_azimuth",
                "name": "sun_azimuth",
                "dtype": "float32",
                "units": "degree",
            },
        )
    )
    requirement = RasterRequirement(
        variables=("nir", "red"),
        dims=("time", "y", "x"),
        require_crs=True,
    )

    result = acquire(
        {"optical": source},
        anchor,
        requirements={"optical": requirement},
    )

    optical = result.gs.rasters["optical"]
    assert list(optical.data_vars) == ["nir", "red"]
    assert "sun_azimuth" in optical.coords
    assert optical.sun_azimuth.dims == ("time",)
    assert optical.sun_azimuth.dtype == np.dtype("float32")
    assert optical.sun_azimuth.attrs["units"] == "degree"
    np.testing.assert_array_equal(optical.sun_azimuth, [10.0, 20.0])
    assert isinstance(optical.nir.data, da.Array)
    assert optical.nir.chunks == ((1, 1), (2, 2), (2, 2))


def test_empty_stac_search_preserves_native_error(local_stac, spec):
    sources, anchor = local_stac
    sources["optical"].client.items.clear()
    with pytest.raises(AnchorFetchError):
        acquire(
            {"optical": sources["optical"]},
            anchor,
            requirements=spec.sources,
        )

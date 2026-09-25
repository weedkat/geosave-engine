"""Numerical prediction contracts exercised with real, small raster models."""

import dask.array as da
from dask.callbacks import Callback
import numpy as np
from odc.geo.geobox import GeoBox
import pandas as pd
import pytest
import torch
from torch import nn
from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.core.stack import stack
from geosave_engine.workflow.spec import (
    InferenceSpec,
    ModelSpec,
    NormalizationSpec,
    RasterRequirement,
    SegmentationSpec,
    TensorInputSpec,
    TilingSpec,
    TimeWindowSpec,
)


@pytest.fixture
def temporal_scene():
    box = GeoBox.from_bbox((0, 0, 60, 60), crs="EPSG:32633", resolution=10)
    values = np.broadcast_to(np.array([2, 4, 6, 8])[:, None, None], (4, 6, 6))
    return stack(
        {
            "series": raster(
                {
                    "red": da.from_array(values, chunks=(1, 3, 3)),
                    "nir": da.full((4, 6, 6), 6, chunks=(1, 3, 3)),
                },
                box,
                time=pd.date_range("2025-01-01", periods=4).values,
            ),
            "terrain": raster({"height": da.full((6, 6), 30, chunks=(3, 3))}, box),
        }
    )


@pytest.fixture
def temporal_settings():
    return InferenceSpec(
        inputs={
            "image": TensorInputSpec(
                raster="series", variables=("nir", "red"), layout="TCHW"
            ),
            "elevation": TensorInputSpec(raster="terrain"),
        },
        time_window=TimeWindowSpec(size=2, stride=2, tolerance="1D"),
        tiling=TilingSpec(raster="series", tile_shape=(4, 4), overlap=2, window="hann"),
    )


class TemporalModel(nn.Module):
    def forward(self, image, elevation, offset=0):
        score = image[:, :, 1].mean(dim=1) + elevation[:, 0] + offset
        return torch.stack((-score, score), dim=1)


def test_temporal_multimodal_batches_reconstruct_each_window(
    temporal_scene, temporal_settings
):
    from geosave_engine.workflow.inference import infer

    model = TemporalModel()
    outputs = infer(
        temporal_scene,
        model=model,
        settings=temporal_settings,
        context={"offset": 5},
        batch_size=3,
    )
    assert len(outputs) == 2
    for result, expected in zip(outputs, (38, 42), strict=True):
        assert result.dims == ("band", "y", "x")
        assert result.gs.geobox == temporal_scene.gs.geobox
        np.testing.assert_allclose(result.isel(band=1), expected)
        np.testing.assert_allclose(result.isel(band=0), -expected)
    assert outputs[0].attrs["time_coverage_start"].startswith("2025-01-01")
    assert outputs[0].attrs["time_coverage_end"].startswith("2025-01-02")
    assert outputs[1].attrs["time_coverage_start"].startswith("2025-01-03")
    assert model.training


def test_unbound_reference_dates_do_not_constrain_model_windows(
    temporal_scene, temporal_settings
):
    from geosave_engine.workflow.inference import infer

    rasters = temporal_scene.gs.rasters
    rasters["reference"] = rasters["series"].assign_coords(
        time=pd.date_range("2026-01-01", periods=4).values
    )
    settings = temporal_settings.model_copy(
        update={
            "tiling": temporal_settings.tiling.model_copy(
                update={"raster": "reference"}
            )
        }
    )
    outputs = infer(stack(rasters), model=TemporalModel(), settings=settings)
    assert len(outputs) == 2
    assert outputs[0].attrs["time_coverage_start"].startswith("2025-01-01")
    np.testing.assert_allclose(outputs[0].isel(band=1), 33)


def test_unselected_temporal_variables_do_not_change_static_model_input(
    temporal_scene, temporal_settings
):
    from geosave_engine.workflow.inference import infer

    rasters = temporal_scene.gs.rasters
    rasters["terrain"]["unused"] = rasters["terrain"].height.expand_dims(
        time=pd.date_range("2026-01-01", periods=3).values
    )
    settings = temporal_settings.model_copy(
        update={
            "inputs": {
                **temporal_settings.inputs,
                "elevation": TensorInputSpec(raster="terrain", variables=("height",)),
            }
        }
    )
    outputs = infer(stack(rasters), model=TemporalModel(), settings=settings)
    assert len(outputs) == 2
    np.testing.assert_allclose(outputs[1].isel(band=1), 37)


def test_static_raster_dates_do_not_change_temporal_output_coverage(
    temporal_scene, temporal_settings
):
    from geosave_engine.workflow.inference import infer

    rasters = temporal_scene.gs.rasters
    rasters["terrain"] = rasters["terrain"].assign_coords(
        time=np.datetime64("2000-01-01")
    )
    outputs = infer(stack(rasters), model=TemporalModel(), settings=temporal_settings)
    assert outputs[0].attrs["time_coverage_start"].startswith("2025-01-01")
    assert outputs[1].attrs["time_coverage_start"].startswith("2025-01-03")


def test_normalization_uses_selected_channel_order_and_preserves_source(raw):
    from geosave_engine.workflow.inference import infer

    settings = InferenceSpec(
        inputs={
            "image": TensorInputSpec(
                raster="optical",
                variables=("nir", "red"),
                normalize=NormalizationSpec(mean=(2, 1), std=(2, 1)),
            )
        },
        tiling=TilingSpec(raster="optical", tile_shape=(2, 2)),
    )

    class Sum(nn.Module):
        def forward(self, image):
            return image.sum(1, keepdim=True)

    (result,) = infer(stack(raw), model=Sum(), settings=settings, batch_size=3)
    np.testing.assert_allclose(result, 3)
    np.testing.assert_allclose(raw["optical"].nir.compute(), 6)


def test_context_extraction_precedes_tile_materialization(
    temporal_scene, temporal_settings
):
    from geosave_engine.workflow.inference import infer

    events = []

    def context(tile):
        events.append("context")
        assert "time" in tile["series"].coords
        return {"offset": torch.tensor(5.0)}

    class Model(TemporalModel):
        def forward(self, image, elevation, offset):
            return super().forward(image, elevation, offset[:, None, None])

    with Callback(pretask=lambda *args: events.append("pixels")):
        outputs = infer(
            temporal_scene,
            model=Model(),
            settings=temporal_settings,
            model_context=context,
            batch_size=2,
        )
    assert events[0] == "context"
    assert "pixels" in events
    np.testing.assert_allclose(outputs[0].isel(band=1), 38)


def test_time_aligned_sun_coordinates_and_foreign_attrs_reach_model_context(
    temporal_scene, temporal_settings
):
    from geosave_engine.workflow import infer, preprocess

    raw = temporal_scene.gs.rasters
    raw["series"] = raw["series"].assign_coords(
        sun_azimuth=(
            "time",
            [100.0, 120.0, 140.0, 160.0],
            {
                "units": "degree",
                "provider_note": "sample observations",
            },
        )
    )
    spec = ModelSpec(
        sources={
            "series": RasterRequirement(
                variables=("nir", "red"),
                attrs={
                    "coords": {
                        "sun_azimuth": {
                            "models": {"coordinate": {"equals": {"units": "degree"}}},
                            "foreign": {
                                "equals": {"provider_note": "sample observations"}
                            },
                        }
                    }
                },
            ),
            "terrain": RasterRequirement(variables=("height",)),
        },
        inference=temporal_settings,
    )
    reads = []
    with Callback(pretask=lambda *args: reads.append(args)):
        prepared = preprocess(raw, spec=spec)
    assert reads == []

    def extract_context(tile):
        angles = tile["series"].sun_azimuth
        assert angles.dims == ("time",)
        assert angles.attrs == raw["series"].sun_azimuth.attrs
        return {"sun_azimuth": torch.tensor(angles.values)}

    class SolarModel(TemporalModel):
        def forward(self, image, elevation, sun_azimuth):
            return super().forward(image, elevation, sun_azimuth.mean(1)[:, None, None])

    outputs = infer(
        prepared,
        model=SolarModel(),
        settings=spec.inference,
        model_context=extract_context,
        batch_size=3,
    )
    np.testing.assert_allclose(outputs[0].isel(band=1), 143)
    np.testing.assert_allclose(outputs[1].isel(band=1), 187)


def test_context_collisions_fail_before_reading_pixels(
    temporal_scene, temporal_settings
):
    from geosave_engine.workflow.inference import infer

    reads = []
    with Callback(pretask=lambda *args: reads.append(args)):
        with pytest.raises(ValueError, match="image"):
            infer(
                temporal_scene,
                model=TemporalModel(),
                settings=temporal_settings,
                context={"image": torch.zeros(1)},
            )
    assert reads == []


def test_model_failure_restores_each_module_training_state(raw, spec):
    from geosave_engine.workflow.inference import infer

    failure = RuntimeError("model failed")

    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Dropout()

        def forward(self, image):
            assert not self.training and not self.child.training
            assert not torch.is_grad_enabled()
            raise failure

    model = Broken()
    model.child.eval()
    with pytest.raises(RuntimeError) as caught:
        infer(stack(raw), model=model, settings=spec.inference)
    assert caught.value is failure
    assert model.training and not model.child.training


@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 1, 1), (1, 2, 4, 4, 1)])
def test_bad_model_output_shape_is_rejected(raw, spec, shape):
    from geosave_engine.workflow.inference import infer

    class Bad(nn.Module):
        def forward(self, image):
            return torch.ones(shape)

    with pytest.raises(ValueError, match="output|shape"):
        infer(stack(raw), model=Bad(), settings=spec.inference)


def test_segmentation_preserves_grid_and_masks_nonfinite_logits():
    from geosave_engine.workflow.postprocessing import postprocess

    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32633", resolution=10)
    logits = array(
        np.array([[[0, 0], [np.nan, 3]], [[3, 0], [np.nan, 0]]]), box, band=[0, 1]
    )
    logits.attrs["time_coverage_start"] = "2025-01-01T00:00:00"
    (output,) = postprocess(
        (logits,),
        settings=SegmentationSpec(
            classes=("background", "water"), thresholds=(0.8, 0.8)
        ),
    )
    np.testing.assert_array_equal(output.prediction, [[1, 255], [255, 0]])
    assert np.isnan(output.confidence.values[1, 0])
    assert output.gs.geobox == box
    assert output.prediction.attrs["_FillValue"] == 255
    assert output.prediction.attrs["flag_meanings"] == "background water"
    assert output.attrs["time_coverage_start"] == "2025-01-01T00:00:00"


def test_uninterpreted_outputs_preserve_logits(raw):
    from geosave_engine.workflow.postprocessing import postprocess

    logits = raw["optical"].nir.expand_dims(band=[0]).rename("scores")
    (output,) = postprocess((logits,), settings=None)
    assert output.logits.identical(logits.rename("logits"))

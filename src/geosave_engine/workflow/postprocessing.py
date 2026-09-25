"""Interpret spatially merged predictions with native geospatial metadata."""

import numpy as np
import torch
import xarray as xr

from geosave_engine.geodata.attrs import Legend, Nodata
from geosave_engine.ml.postprocessing.segmentation import apply_thresholds

from .spec import OperationSpec, SegmentationSpec


def postprocess(
    outputs: tuple[xr.DataArray, ...],
    *,
    settings: SegmentationSpec | OperationSpec | None = None,
) -> tuple[xr.Dataset, ...]:
    """Interpret each window after tile merging, retaining its spatial grid.

    Args:
        outputs: Merged model outputs in temporal-window order.
        settings: Segmentation settings or an installed Dataset-returning callable.

    Returns:
        One Dataset per window, with logits or interpreted prediction variables.

    Raises:
        ValueError: Segmentation channels do not match the configured classes.
        TypeError: A custom operation returns something other than a Dataset.
    """
    if isinstance(settings, OperationSpec):
        settings = OperationSpec.model_validate(settings)
        if settings.inputs:
            raise ValueError("Postprocessing operations cannot reference input rasters")
        settings.resolve(xr.DataArray())
        results = []
        for output in outputs:
            result = settings.apply(output.copy(deep=True))
            if not isinstance(result, xr.Dataset):
                raise TypeError(
                    "A postprocessing operation must return an xarray.Dataset"
                )
            results.append(result)
        return tuple(results)
    if settings is None:
        return tuple(
            output.rename("logits").to_dataset(promote_attrs=True) for output in outputs
        )
    settings = SegmentationSpec.model_validate(settings)
    results = []
    for output in outputs:
        if (
            output.ndim != 3
            or output.dims[0] != "band"
            or output.sizes["band"] != len(settings.classes)
        ):
            raise ValueError(
                "Segmentation output channels must match the configured classes"
            )
        values = np.asarray(output.values)
        invalid = ~np.isfinite(values).all(axis=0)
        tensor = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
        thresholds = torch.tensor(settings.thresholds or (0.0,) * len(settings.classes))
        labels, confidence = apply_thresholds(
            tensor,
            thresholds,
            settings.ignore_index,
            torch.from_numpy(invalid).unsqueeze(0),
        )
        template = output.isel(band=0, drop=True)
        prediction = template.copy(data=labels[0].numpy()).rename("prediction")
        prediction.attrs = {
            **Nodata(fill_value=settings.ignore_index).to_attrs(),
            **Legend(class_map=dict(enumerate(settings.classes))).to_attrs(),
        }
        certainty = template.copy(data=confidence[0].numpy()).rename("confidence")
        certainty = certainty.where(~invalid)
        certainty.attrs = Nodata(fill_value=np.nan).to_attrs()
        results.append(
            xr.Dataset(
                {"prediction": prediction, "confidence": certainty},
                attrs=dict(output.attrs),
            )
        )
    return tuple(results)
